#!/usr/bin/env python
"""Graph encoder/decoder helpers used by the conditional diffusion pipeline."""

import copy
import numpy as np
import networkx as nx
import random
import pulp
import dill as pickle
from .timeit import timeit
import torch
from typing import List, Tuple, Optional, Any, Sequence, Dict, Union
from .low_rank_mlp import LowRankMLP
from .conditional_node_generator_base import ConditionalNodeGeneratorBase

def scaled_slerp(v0: np.ndarray, v1: np.ndarray, t: float) -> np.ndarray:
    """Interpolate between vectors on the hypersphere while blending magnitudes linearly."""
    # Compute magnitudes
    mag0 = np.linalg.norm(v0)
    mag1 = np.linalg.norm(v1)

    # Normalize directions (guard against zero)
    v0_unit = v0 / mag0 if mag0 != 0 else v0
    v1_unit = v1 / mag1 if mag1 != 0 else v1

    # Compute angle between
    dot = np.clip(np.dot(v0_unit, v1_unit), -1.0, 1.0)
    theta = np.arccos(dot)

    # Slerp the direction
    if theta < 1e-6:
        # Nearly colinear: fall back to linear interpolation + renormalization
        direction = (1 - t) * v0_unit + t * v1_unit
        norm = np.linalg.norm(direction)
        direction = direction / norm if norm != 0 else direction
    else:
        sin_theta = np.sin(theta)
        direction = (
            np.sin((1 - t) * theta) * v0_unit +
            np.sin(t * theta) * v1_unit
        ) / sin_theta

    # Linearly interpolate magnitudes
    mag = (1 - t) * mag0 + t * mag1
    return direction * mag


def scaled_slerp_average(vectors: np.ndarray) -> np.ndarray:
    """Compute a magnitude-aware mean direction for a batch of vectors."""
    vs = np.asarray(vectors, dtype=float)             # (B, D)
    mags = np.linalg.norm(vs, axis=1)                # (B,)
    unit_vs = np.zeros_like(vs)                      # (B, D)
    nonzero = mags > 0
    unit_vs[nonzero] = vs[nonzero] / mags[nonzero, None]

    avg_dir = unit_vs.sum(axis=0)                    # (D,)
    norm = np.linalg.norm(avg_dir)
    if norm > 0:
        avg_dir /= norm

    avg_mag = mags.mean()
    return avg_dir * avg_mag                         # (D,)


# Suppress numpy warnings for invalid operations and divisions
np.seterr(invalid='ignore', divide='ignore')

# =============================================================================
# DecompositionalNodeEncoderDecoder Class
# =============================================================================

class DecompositionalNodeEncoderDecoder(object):
    """Trainable toolkit for vectorising graphs and decoding them back into structure."""
    
    def __init__(
        self,
        adjacency_matrix_classifier: LowRankMLP,
        node_label_classifier: LowRankMLP,
        edge_label_classifier: LowRankMLP,       
        verbose: bool = True,
        negative_sample_factor: int = 1,
        existence_threshold: float = 0.5,
        num_augmentation_iterations: int = 0,
        augmentation_noise: float = 1e-2,
        enforce_connectivity: bool = True,
        degree_slack_penalty: float = 1e6,
        warm_start_mst: bool = True
    ) -> None:
        """Create deep copies of the supplied classifiers and store training hyper-parameters."""
        self.adjacency_matrix_classifier = copy.deepcopy(adjacency_matrix_classifier)
        self.node_label_classifier      = copy.deepcopy(node_label_classifier)
        self.edge_label_classifier      = copy.deepcopy(edge_label_classifier)
        self.verbose                    = verbose
        self.negative_sample_factor     = negative_sample_factor
        self.existence_threshold        = existence_threshold
        self.num_augmentation_iterations= num_augmentation_iterations
        self.augmentation_noise         = augmentation_noise
        self.enforce_connectivity       = enforce_connectivity
        self.degree_slack_penalty       = degree_slack_penalty
        self.warm_start_mst             = warm_start_mst

    def optimize_adjacency_matrix(
        self,
        prob_matrix: np.ndarray,
        target_degrees: List[int],
        timeLimit: int = 60,
        verbose: bool = False,
        alpha: float = 0.7,
        connectivity: Optional[bool] = None
    ) -> np.ndarray:
        """Optimise a binary adjacency matrix subject to degree and connectivity targets."""
        n = prob_matrix.shape[0]
        # Smooth probabilities
        if alpha != 1.0:
            prob_matrix = np.power(prob_matrix, alpha)

        # Connectivity setting
        if connectivity is None:
            connectivity = self.enforce_connectivity

        # Build LP
        prob = pulp.LpProblem("AdjacencyMatrixOptimization", pulp.LpMaximize)

        # Decision vars
        x = {(i, j): pulp.LpVariable(f"x_{i}_{j}", cat="Binary")
             for i in range(n) for j in range(i+1, n)}
        u = {i: pulp.LpVariable(f"u_{i}", lowBound=0, cat="Integer") for i in range(n)}
        v = {i: pulp.LpVariable(f"v_{i}", lowBound=0, cat="Integer") for i in range(n)}

        # Objective
        prob += (
            pulp.lpSum(prob_matrix[i, j] * x[(i, j)] for i in range(n) for j in range(i+1, n))
            - self.degree_slack_penalty * pulp.lpSum(u[i] + v[i] for i in range(n))
        )

        # Degree constraints
        for i in range(n):
            incident = [x[(i,j)] for j in range(i+1, n)] + [x[(j,i)] for j in range(i) if (j,i) in x]
            prob += (pulp.lpSum(incident) + u[i] - v[i] == target_degrees[i]), f"Degree_{i}"

        # Connectivity via flow
        if connectivity:
            directed_edges = [(i,j) for (i,j) in x] + [(j,i) for (i,j) in x]
            f_vars = {(u_, v_): pulp.LpVariable(f"f_{u_}_{v_}", lowBound=0, cat="Continuous")
                      for u_,v_ in directed_edges}
            M = n-1
            root = 0
            for v_idx in range(n):
                inflow  = pulp.lpSum(f_vars[(u_,v2)] for (u_,v2) in directed_edges if v2==v_idx)
                outflow = pulp.lpSum(f_vars[(v2,w)] for (v2,w) in directed_edges if v2==v_idx)
                prob += ((outflow-inflow)==M if v_idx==root else (inflow-outflow)==1), f"Flow_{v_idx}"
            for u_,v_ in directed_edges:
                i,j = min(u_,v_), max(u_,v_)
                prob += (f_vars[(u_,v_)] <= M * x[(i,j)]), f"FlowCouple_{u_}_{v_}"

        # Warm-start with MST
        if self.warm_start_mst:
            G = nx.Graph()
            G.add_nodes_from(range(n))
            for i in range(n):
                for j in range(i+1, n):
                    G.add_edge(i, j, weight=prob_matrix[i,j])
            T = nx.maximum_spanning_tree(G)
            # Initialize x
            for (i,j), var in x.items():
                var.start = 1 if T.has_edge(i,j) else 0

        # Solve
        solver = pulp.PULP_CBC_CMD(timeLimit=timeLimit, msg=verbose)
        prob.solve(solver)

        # Build adjacency
        adj = np.zeros((n,n), dtype=int)
        for (i,j), var in x.items():
            adj[i,j] = adj[j,i] = int(pulp.value(var))
        return adj

    def graphs_to_adjacency_matrices(self, graphs: List[nx.Graph]) -> List[np.ndarray]:
        """Convert each NetworkX graph into a dense adjacency array."""
        adj_mtx_list = []
        for graph in graphs:
            # Convert graph to a numpy array with integer type.
            adj_mtx = nx.to_numpy_array(graph, dtype=int)
            adj_mtx_list.append(adj_mtx)
        return adj_mtx_list

    def adj_mtx_to_targets(
        self,
        adj_mtx_list: List[np.ndarray],
        node_encodings_list: List[np.ndarray],
        locality_sample_fraction: float,
        force_bi_directional_edges: bool = True,
        is_training: bool = False,
        horizon: int = 1
    ) -> Tuple[np.ndarray, List[Tuple[int, int, int]]]:
        """Label node pairs as local or non-local using shortest-path distance within each graph."""
        if horizon < 1:
            raise ValueError("horizon must be >= 1")

        # Collect all targets and pairs first
        all_targets = []
        all_pairs = []
        
        for g_idx, (adj_mtx, encodings) in enumerate(zip(adj_mtx_list, node_encodings_list)):
            n_nodes = adj_mtx.shape[0]
            # Treat graphs as simple, unweighted, undirected structures
            G = nx.from_numpy_array(adj_mtx, create_using=nx.Graph)
            for i in range(n_nodes):
                # Collect positive neighbours for node i up to the specified horizon
                lengths = nx.single_source_shortest_path_length(G, i, cutoff=horizon)
                pos_neighbors = [j for j, dist in lengths.items() if j != i and dist <= horizon]
                
                # Add positive examples (both directions if force_bi_directional_edges)
                for j in pos_neighbors:
                    all_targets.append(1)
                    all_pairs.append((g_idx, i, j))
                    if force_bi_directional_edges:
                        all_targets.append(1)  
                        all_pairs.append((g_idx, j, i))
                
                # Determine number of negative samples
                num_pos = len(pos_neighbors) * (2 if force_bi_directional_edges else 1)
                num_neg_samples = int(round(self.negative_sample_factor * num_pos))
                if num_neg_samples <= 0:
                    continue
                
                # Build candidate list for negative examples
                candidate_indices = [k for k in range(n_nodes) if k != i and k not in lengths]
                if not candidate_indices:
                    continue
                
                # Get distances and sort candidates
                distances = np.array([np.linalg.norm(encodings[i] - encodings[k]) for k in candidate_indices])
                sorted_candidate_indices = np.argsort(distances)
                selected_negatives = [candidate_indices[idx] for idx in sorted_candidate_indices[:num_neg_samples]]
                
                # Add negative examples (both directions if force_bi_directional_edges)
                for k in selected_negatives:
                    all_targets.append(0)
                    all_pairs.append((g_idx, i, k))
                    if force_bi_directional_edges:
                        all_targets.append(0)
                        all_pairs.append((g_idx, k, i))
        
        # Apply locality-pair sampling during training when requested
        if is_training and locality_sample_fraction < 1.0:
            num_pairs = len(all_pairs)
            num_pairs_to_use = int(round(num_pairs * locality_sample_fraction))

            if self.verbose and num_pairs > 0:
                print(f"adj_mtx_to_targets: Sampling {num_pairs_to_use} locality pairs ({locality_sample_fraction:.2%}) from {num_pairs} total pairs.")

            if 0 < num_pairs_to_use < num_pairs:
                indices = np.random.choice(num_pairs, num_pairs_to_use, replace=False)
                all_targets = [all_targets[i] for i in indices]
                all_pairs = [all_pairs[i] for i in indices]
            elif num_pairs_to_use == 0 and num_pairs > 0:
                if self.verbose:
                    print(f"adj_mtx_to_targets: Warning - num_pairs_to_use is 0 with locality_sample_fraction={locality_sample_fraction} and num_pairs={num_pairs}. No locality pairs will be used.")
                return np.array([]), []
            elif num_pairs_to_use == 0 and num_pairs == 0:
                return np.array([]), []

        return np.array(all_targets), all_pairs

    def compute_edge_supervision(
        self, 
        graphs: List[nx.Graph], 
        node_encodings_list: List[np.ndarray],
        locality_sample_fraction: float,
        horizon: int = 1
    ) -> Tuple[np.ndarray, List[Tuple[int, int, int]]]:
        """Compute locality supervision pairs for training."""
        adj = self.graphs_to_adjacency_matrices(graphs)
        return self.adj_mtx_to_targets(
            adj,
            node_encodings_list,
            locality_sample_fraction=locality_sample_fraction,
            is_training=True,
            horizon=horizon
        )

    def encodings_and_adj_mtx_to_dataset(
        self,
        node_encodings_list: List[np.ndarray],
        adj_mtx_list: List[np.ndarray],
        locality_sample_fraction: float,
        horizon: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Assemble feature/label pairs for the locality classifier derived from adjacencies."""
        y, pair_indices = self.adj_mtx_to_targets(
            adj_mtx_list,
            node_encodings_list,
            locality_sample_fraction=locality_sample_fraction,
            is_training=True,
            horizon=horizon
        )
        X = self.encodings_to_instances(node_encodings_list, pair_indices)
        return X, y

    def encodings_to_instances(
        self,
        node_encodings_list: List[np.ndarray],
        pair_indices: Optional[List[Tuple[int, int, int]]] = None,
        use_graph_encoding: bool = False
    ) -> np.ndarray:
        """Stack node-pair feature vectors, optionally augmented with a graph-level summary."""
        instances = []
        if pair_indices is not None:
            # Use provided pair indices.
            for g_idx, i, j in pair_indices:
                encodings = node_encodings_list[g_idx]
                if use_graph_encoding: 
                    graph_encoding = np.sum(encodings, axis=0)
                    instance = np.hstack([graph_encoding, encodings[i], encodings[j]])
                else:
                    instance = np.hstack([encodings[i], encodings[j]])
                instances.append(instance)
        else:
            # Evaluate all pairs (i, j) with i != j for every graph.
            for g_idx, encodings in enumerate(node_encodings_list):
                if use_graph_encoding: 
                    graph_encoding = np.sum(encodings, axis=0)
                n_nodes = encodings.shape[0]
                for i in range(n_nodes):
                    for j in range(n_nodes):
                        if i != j:
                            if use_graph_encoding:
                                instance = np.hstack([graph_encoding, encodings[i], encodings[j]])
                            else:
                                instance = np.hstack([encodings[i], encodings[j]])
                            instances.append(instance)
        return np.vstack(instances)
        
    @timeit
    def encodings_and_graphs_to_node_label_dataset(
        self,
        node_encodings_list: List[np.ndarray],
        graphs: List[nx.Graph]
    ) -> Tuple[np.ndarray, List[Any]]:
        """Produce node-level training examples paired with labels from the graphs."""
        instances = []
        node_labels = []
        # Process each graph and its corresponding encoding matrix.
        for graph, encodings in zip(graphs, node_encodings_list):
            for i, u in enumerate(list(graph.nodes())):
                instances.append(encodings[i])
                # Assume node label is stored under the key 'label'.
                node_labels.append(graph.nodes[u]['label'])
        return np.vstack(instances), node_labels

    def encodings_and_graphs_to_edge_label_dataset(
        self,
        node_encodings_list: List[np.ndarray],
        graphs: List[nx.Graph]
    ) -> Tuple[np.ndarray, List[Any]]:
        """Produce edge-level feature vectors paired with labels drawn from the graphs."""
        instances = []
        edge_labels = []
        for graph, encodings in zip(graphs, node_encodings_list):
            # Iterate over nodes to consider each potential edge.
            for i, u in enumerate(list(graph.nodes())):
                for j, v in enumerate(list(graph.nodes())):
                    # If an edge exists between the nodes, create an instance.
                    if graph.has_edge(u, v):
                        instance = np.hstack([encodings[i], encodings[j]])
                        instances.append(instance)
                        edge_labels.append(graph.edges[u, v]['label'])
        if instances:
            instances = np.vstack(instances)
        return instances, edge_labels

    def encodings_and_adj_mtx_to_edge_dataset(
        self,
        node_encodings_list: List[np.ndarray],
        adj_mtx_list: List[np.ndarray]
    ) -> np.ndarray:
        """Collect node-pair feature vectors wherever the adjacency indicates a connection."""
        instances = []
        for encodings, adj_mtx in zip(node_encodings_list, adj_mtx_list):
            n_nodes = encodings.shape[0]
            # Iterate over all pairs of nodes.
            for i in range(n_nodes):
                for j in range(n_nodes):
                    if adj_mtx[i, j] != 0:
                        instance = np.hstack([encodings[i], encodings[j]])
                        instances.append(instance)
        if instances:
            instances = np.vstack(instances)
        return instances
    
    @timeit
    def node_label_classifier_fit(
        self,
        node_encodings_list: List[np.ndarray],
        graphs: List[nx.Graph]
    ) -> None:
        """Fit the node-label classifier or cache the constant label shortcut."""
        X, y = self.encodings_and_graphs_to_node_label_dataset(node_encodings_list, graphs)
        unique_labels = np.unique(y)
        # If only one label is present, skip training.
        if len(unique_labels) == 1:
            if self.verbose:
                print("Only one node label found: {}. Skipping training and storing the label.".format(unique_labels[0]))
            self.single_node_label = unique_labels[0]
        else:
            if self.verbose:
                print('Training node label predictor on {} instances with {} features'.format(X.shape[0], X.shape[1]))
            self.node_label_classifier.fit(X, y)
            self.single_node_label = None

    @timeit
    def edge_label_classifier_fit(
        self,
        node_encodings_list: List[np.ndarray],
        graphs: List[nx.Graph]
    ) -> None:
        """Fit the edge-label classifier or remember the sole label when degenerate."""
        X, y = self.encodings_and_graphs_to_edge_label_dataset(node_encodings_list, graphs)
        unique_labels = np.unique(y)
        # If only one edge label is found, store it and skip training.
        if len(unique_labels) == 1:
            if self.verbose:
                print("Only one edge label found: {}. Skipping training and storing the label.".format(unique_labels[0]))
            self.single_edge_label = unique_labels[0]
        else:
            if self.verbose:
                print('Training edge label predictor on {} instances with {} features'.format(X.shape[0], X.shape[1]))
            self.edge_label_classifier.fit(X, y)
            self.single_edge_label = None

    @timeit    
    def adjacency_matrix_classifier_fit(
        self,
        node_encodings_list: List[np.ndarray],
        graphs: List[nx.Graph],
        locality_sample_fraction: float # <-- New parameter
    ) -> None:
        """Train the adjacency classifier on features derived from the provided graphs."""
        # Convert graphs to corresponding adjacency matrices.
        adj_mtx_list = self.graphs_to_adjacency_matrices(graphs)
        # Build the training dataset.
        X, y = self.encodings_and_adj_mtx_to_dataset(
            node_encodings_list,
            adj_mtx_list,
            locality_sample_fraction=locality_sample_fraction,
            horizon=1
        )
        if self.verbose:
            print('Training adjacency matrix predictor on {} instances with {} features'.format(X.shape[0], X.shape[1]))
        self.adjacency_matrix_classifier.fit(X, y)

    @timeit
    def fit(
        self,
        graphs: List[nx.Graph],
        node_encodings_list: List[np.ndarray],
        locality_sample_fraction_for_adj_mtx: float  # <-- New parameter (unchanged)
    ) -> 'DecompositionalNodeEncoderDecoder':
        """Train the node, edge, and adjacency predictors on the supplied dataset."""

        # ------------------------------------------------------------------
        # 1. Build augmented dataset
        # ------------------------------------------------------------------
        combined_graphs: List[nx.Graph] = list(graphs)                 # shallow copy is fine
        combined_encodings: List[np.ndarray] = list(node_encodings_list)

        if self.num_augmentation_iterations > 0:
            noise_list = np.linspace(0, self.augmentation_noise, self.num_augmentation_iterations + 1).tolist()

            for noise in noise_list:
                if self.verbose:
                    print(f'<<Generating noisy encodings (noise={noise})>>')

                for enc in node_encodings_list:
                    perturbed = enc + np.random.rand(*enc.shape) * noise
                    combined_encodings.append(perturbed)

                # Point to the same graphs for every new encoding set
                combined_graphs.extend(graphs)

        # ------------------------------------------------------------------
        # 2. Train once on the enlarged dataset
        # ------------------------------------------------------------------
        self.node_label_classifier_fit(combined_encodings, combined_graphs)
        self.edge_label_classifier_fit(combined_encodings, combined_graphs)
        self.adjacency_matrix_classifier_fit(
            combined_encodings, combined_graphs,
            locality_sample_fraction=locality_sample_fraction_for_adj_mtx
        )

        return self

    def constrained_node_encodings_list(
        self,
        original_node_encodings_list: List[np.ndarray]
    ) -> List[np.ndarray]:
        """Return copies of the encodings with negative entries clipped to zero."""
        constrained = []
        for encoding in original_node_encodings_list:
            new_enc = encoding.copy()
            # Replace negative values with 0.
            new_enc[new_enc < 0] = 0
            constrained.append(new_enc)
        return constrained

    def get_degrees(
        self,
        encodings: np.ndarray,
        n_nodes: int,
        threshold: float = 0.5
    ) -> List[int]:
        """Derive integer degree targets from encodings while masking non-existent nodes."""
        degs = np.rint(encodings[:n_nodes, 1])
        existent = encodings[:n_nodes, 0] >= threshold
        # For existent nodes enforce at least 1, for non-existent nodes set to 0.
        degs = np.where(existent, np.maximum(degs, 1), 0)
        return degs.astype(int).tolist()

    def decode_adjacency_matrix(
        self,
        original_node_encodings_list: List[np.ndarray],
        existence_threshold: float = 0.5
    ) -> List[np.ndarray]:
        """Project predicted adjacency probabilities onto valid binary graphs."""
        # Constrain encodings to be non-negative.
        node_encodings_list = self.constrained_node_encodings_list(original_node_encodings_list)
        
        # Calculate the number of instances per graph (all ordered pairs except self-pairs).
        sizes = [enc.shape[0]**2 - enc.shape[0] for enc in node_encodings_list]
        # Generate instances for adjacency matrix prediction.
        X = self.encodings_to_instances(node_encodings_list)
        predicted_probs = self.adjacency_matrix_classifier.predict_proba(X)[:, -1]
        # Split predicted probabilities for each graph based on the sizes computed.
        predicted_probs_list = np.split(predicted_probs, np.cumsum(sizes)[:-1])
        
        adj_mtx_list = []
        # Process each graph's predictions.
        for prob_list, encodings in zip(predicted_probs_list, node_encodings_list):
            n_nodes = encodings.shape[0]
            idx = 0
            prob_matrix = np.zeros((n_nodes, n_nodes))
            # Reconstruct the probability matrix from the flat list.
            for i in range(n_nodes):
                for j in range(n_nodes):
                    if i != j:
                        prob_matrix[i, j] = prob_list[idx]
                        idx += 1
            # Zero out probabilities for edges where either node is non-existent.
            existent = encodings[:, 0] >= existence_threshold
            for i in range(n_nodes):
                for j in range(n_nodes):
                    if not (existent[i] and existent[j]):
                        prob_matrix[i, j] = 0
            # Ensure the matrix is symmetric.
            prob_matrix = (prob_matrix + prob_matrix.T) / 2
            # Extract target degrees from encodings (modified to set degree=0 for non-existent nodes).
            target_degrees = self.get_degrees(encodings, n_nodes, threshold=existence_threshold)
            # Optimize the probability matrix into a binary adjacency matrix.
            adj = self.optimize_adjacency_matrix(prob_matrix, target_degrees)
            adj_mtx_list.append(adj)
        return adj_mtx_list

    def decode_node_labels(
        self,
        node_encodings_list: List[np.ndarray]
    ) -> List[np.ndarray]:
        """Predict node-level labels for each encoding matrix."""
        # If a single node label exists, return it for all nodes.
        if hasattr(self, 'single_node_label') and self.single_node_label is not None:
            predicted_node_labels_list = []
            for enc in node_encodings_list:
                n = enc.shape[0]
                predicted_node_labels_list.append(np.array([self.single_node_label] * n))
            return predicted_node_labels_list

        # Otherwise, predict labels for all nodes.
        X = np.vstack(node_encodings_list)
        predicted_node_labels = self.node_label_classifier.predict(X)
        sizes = [enc.shape[0] for enc in node_encodings_list]
        predicted_node_labels_list = np.split(predicted_node_labels, np.cumsum(sizes)[:-1])
        return predicted_node_labels_list

    def decode_edge_labels(
        self,
        node_encodings_list: List[np.ndarray],
        adj_mtx_list: List[np.ndarray]
    ) -> List[np.ndarray]:
        """Predict edge labels for every edge present in the supplied adjacency matrices."""
        # If a single edge label exists, return it for all edges.
        if hasattr(self, 'single_edge_label') and self.single_edge_label is not None:
            predicted_edge_labels_list = []
            for adj in adj_mtx_list:
                n_edges = int(np.sum(adj))
                predicted_edge_labels_list.append(np.array([self.single_edge_label] * n_edges))
            return predicted_edge_labels_list

        # Create instances based on encodings and adjacency matrices.
        X = self.encodings_and_adj_mtx_to_edge_dataset(node_encodings_list, adj_mtx_list)
        if len(X) < 1:
            return [[] for _ in node_encodings_list]
        predicted_edge_labels = self.edge_label_classifier.predict(X)
        sizes = [np.sum(adj) for adj in adj_mtx_list]
        predicted_edge_labels_list = np.split(predicted_edge_labels, np.cumsum(sizes)[:-1])
        return predicted_edge_labels_list

    @timeit
    def decode(
        self,
        original_node_encodings_list: List[np.ndarray]
    ) -> List[nx.Graph]:
        """Reconstruct labelled graphs from node encodings, respecting existence masks."""
        # Step 1: Decode the adjacency matrices using the modified method that accounts for node existence.
        adj_mtx_list = self.decode_adjacency_matrix(original_node_encodings_list, existence_threshold=self.existence_threshold)
        
        # Step 2: Decode node labels (this method remains unchanged).
        predicted_node_labels_list = self.decode_node_labels(original_node_encodings_list)
        
        # Step 3: Decode edge labels based on the updated adjacency matrices.
        predicted_edge_labels_list = self.decode_edge_labels(original_node_encodings_list, adj_mtx_list)
        
        graphs = []
        # Step 4: Reconstruct each graph and filter out non-existent nodes.
        for encodings, node_labels, edge_labels, adj_mtx in zip(
                original_node_encodings_list, predicted_node_labels_list, predicted_edge_labels_list, adj_mtx_list):
            # Create the initial graph from the predicted adjacency matrix.
            graph = nx.from_numpy_array(adj_mtx)
            
            # Assign node labels. (Assumes ordering in node_labels matches node indices.)
            node_label_map = {i: label for i, label in enumerate(node_labels)}
            nx.set_node_attributes(graph, node_label_map, 'label')
            
            # If edges exist, assign edge labels.
            if np.sum(adj_mtx) > 0:
                n_nodes = graph.number_of_nodes()
                edge_idx = 0
                edge_attr = {}
                for i in range(n_nodes):
                    for j in range(n_nodes):
                        if adj_mtx[i, j] != 0:
                            edge_attr[(i, j)] = edge_labels[edge_idx]
                            edge_idx += 1
                nx.set_edge_attributes(graph, edge_attr, 'label')
            
            # Filter out nodes that do not meet the existence threshold.
            existent_indices = np.where(encodings[:, 0] >= self.existence_threshold)[0]
            # Create a subgraph that includes only existent nodes.
            filtered_graph = graph.subgraph(existent_indices).copy()
            graphs.append(filtered_graph)
        
        return graphs
    
    def save(self, filename: str = 'generative_model.obj') -> None:
        """Serialise the current object to `filename` using pickle."""
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
    
    def load(self, filename: str = 'generative_model.obj') -> 'DecompositionalNodeEncoderDecoder':
        """Load a previously saved instance from disk and return it."""
        with open(filename, 'rb') as f:
            self = pickle.load(f)
        return self

# =============================================================================
# ConditionalNodeGeneratorModel Class
# =============================================================================
class ConditionalNodeGeneratorModel(object):
    """Thin wrapper that delegates training and inference to a conditional node generator."""
    def __init__(
            self, 
            conditional_node_generator: Optional[ConditionalNodeGeneratorBase] = None,
            verbose: bool = True
            ) -> None:
        """Clone the supplied generator and persist the verbosity preference."""
        self.conditional_node_generator = copy.deepcopy(conditional_node_generator)
        self.verbose = verbose

    @timeit
    def fit(
        self,
        node_encodings_list: List[np.ndarray],
        conditional_graph_encodings: Any,
        edge_pairs: Optional[List[Tuple[int, int, int]]] = None,
        edge_targets: Optional[np.ndarray] = None,
        node_mask: Optional[np.ndarray] = None
    ) -> 'ConditionalNodeGeneratorModel':
        if self.verbose:
            print(f"Training conditional model on {len(node_encodings_list)} graphs with {node_encodings_list[0].shape[0]} nodes each.")

        if edge_pairs is not None and edge_targets is not None:
            if self.verbose:
                print(f"Using locality supervision with {len(edge_pairs)} labelled pairs.")
            self.conditional_node_generator.setup(
                node_encodings_list=node_encodings_list,
                conditional_graph_encodings=conditional_graph_encodings,
                edge_pairs=edge_pairs,
                edge_targets=edge_targets,
                node_mask=node_mask
            )
            self.conditional_node_generator.fit(
                node_encodings_list=node_encodings_list,
                conditional_graph_encodings=conditional_graph_encodings,
                edge_pairs=edge_pairs,
                edge_targets=edge_targets,
                node_mask=node_mask
            )
        else:
            self.conditional_node_generator.setup(
                node_encodings_list=node_encodings_list,
                conditional_graph_encodings=conditional_graph_encodings,
                node_mask=node_mask
            )
            self.conditional_node_generator.fit(
                node_encodings_list=node_encodings_list,
                conditional_graph_encodings=conditional_graph_encodings,
                node_mask=node_mask
            )

        return self

    @timeit
    def predict(
        self,
        conditional_graph_encodings: Any,
        desired_class: Optional[Union[int, Sequence[int]]] = None
    ) -> List[np.ndarray]:
        if self.verbose:
            print(f"Predicting node matrices for {len(conditional_graph_encodings)} graphs...")
        predicted_node_encodings_list = self.conditional_node_generator.predict(conditional_graph_encodings, desired_class=desired_class)
        return predicted_node_encodings_list
    
# =============================================================================
# DecompositionalEncoderDecoder Class 
# =============================================================================

class DecompositionalEncoderDecoder(object):
    """End-to-end manager that vectorises graphs, trains generators, and rebuilds structures."""
    def __init__(
            self,
            graph_vectorizer: Any = None,
            node_graph_vectorizer: Any = None,
            conditioning_to_node_embeddings_generator: Optional[ConditionalNodeGeneratorModel] = None,
            node_embeddings_to_graph_generator: Optional[DecompositionalNodeEncoderDecoder] = None,
            verbose: bool = True,
            use_locality_supervision: bool = False,
            locality_sample_fraction: float = 1.0,
            locality_horizon: int = 1
            ) -> None:
        """Store the collaborating components and configuration used for the pipeline."""
        self.graph_vectorizer = graph_vectorizer
        self.node_graph_vectorizer = node_graph_vectorizer
        self.conditioning_to_node_embeddings_generator = conditioning_to_node_embeddings_generator
        self.node_embeddings_to_graph_generator = node_embeddings_to_graph_generator
        self.verbose = verbose
        self.use_locality_supervision = use_locality_supervision
        if not 0.0 < locality_sample_fraction <= 1.0:
            raise ValueError("locality_sample_fraction must be between 0.0 (exclusive) and 1.0 (inclusive)")
        self.locality_sample_fraction = locality_sample_fraction
        if locality_horizon < 1:
            raise ValueError("locality_horizon must be >= 1")
        self.locality_horizon = locality_horizon

    def toggle_verbose(self) -> None:
        """Flip verbosity for this instance and any nested generators."""
        self.verbose = not self.verbose
        if self.conditioning_to_node_embeddings_generator is not None:
            self.conditioning_to_node_embeddings_generator.verbose = self.verbose
        if self.node_embeddings_to_graph_generator is not None:
            self.node_embeddings_to_graph_generator.verbose = self.verbose

    @timeit
    def fit(
        self,
        graphs: List[nx.Graph],
        train_conditioning_to_node_embeddings_generator: bool = True,
        train_node_embeddings_to_graph_generator: bool = True
    ) -> 'DecompositionalEncoderDecoder':
        if self.verbose:
            print(f"Fitting model on {len(graphs)} graphs")

        # Fit vectorizers
        self.graph_vectorizer.fit(graphs)
        self.node_graph_vectorizer.fit(graphs)

        # Generate encodings
        node_encodings_list, conditional_graph_encodings = self.encode(graphs)

        if train_conditioning_to_node_embeddings_generator:
            edge_pairs_for_cond_gen = None
            edge_targets_for_cond_gen = None
            node_mask_for_cond_gen = None # Assuming node_mask might be needed by ConditionalNodeGeneratorModel

            if self.use_locality_supervision:
                if self.node_embeddings_to_graph_generator is None:
                    raise RuntimeError("Locality supervision requested but node_embeddings_to_graph_generator is None.")
                if self.verbose:
                    print(f"Using locality supervision (horizon={self.locality_horizon}) for training the conditioning→node generator.")

                edge_targets_for_cond_gen, edge_pairs_for_cond_gen = self.node_embeddings_to_graph_generator.compute_edge_supervision(
                    graphs,
                    node_encodings_list,
                    locality_sample_fraction=self.locality_sample_fraction,
                    horizon=self.locality_horizon
                )
            
            self.conditioning_to_node_embeddings_generator.fit(
                node_encodings_list=node_encodings_list,
                conditional_graph_encodings=conditional_graph_encodings,
                edge_pairs=edge_pairs_for_cond_gen,
                edge_targets=edge_targets_for_cond_gen,
                node_mask=node_mask_for_cond_gen # Pass if available/needed
            )

        if train_node_embeddings_to_graph_generator:
            self.node_embeddings_to_graph_generator.fit(graphs, node_encodings_list, locality_sample_fraction_for_adj_mtx=self.locality_sample_fraction)

        return self

    @timeit
    def node_encode(self, graphs: List[nx.Graph]) -> List[np.ndarray]:
        """Transform graphs into per-node embedding matrices."""
        if self.verbose:
            print(f"Node encoding {len(graphs)} graphs")
        return self.node_graph_vectorizer.transform(graphs)

    @timeit
    def graph_encode(self, graphs: List[nx.Graph]) -> List[np.ndarray]:
        """Transform graphs into global conditioning vectors."""
        if self.verbose:
            print(f"Encoding {len(graphs)} graphs")
        return self.graph_vectorizer.transform(graphs)

    def encode(self, graphs: List[nx.Graph]) -> Tuple[List[np.ndarray], Any]:
        """Produce both node-level encodings and their matching conditioning vectors."""
        node_encs = self.node_encode(graphs)
        cond_encs = self.graph_encode(graphs)
        return node_encs, cond_encs

    def decode(self, conditioning_vectors: Any, desired_class: Optional[Union[int, Sequence[int]]] = None) -> List[nx.Graph]:
        """Decode conditioning vectors into reconstructed graphs."""
        if self.verbose:
            print(f"Decoding {len(conditioning_vectors)} conditioning vectors")
            if desired_class is not None:
                print(f"Using classifier guidance toward class(es): {desired_class}")
        
        node_feats = self.conditioning_to_node_embeddings_generator.predict(
            conditioning_vectors, desired_class=desired_class
        )
        return self.node_embeddings_to_graph_generator.decode(node_feats)

    @timeit
    def sample(self, n_samples: int = 1, desired_class: Optional[Union[int, Sequence[int]]] = None) -> List[nx.Graph]:
        """Generate random graphs by sampling conditioning vectors from the prior."""
        if self.verbose:
            print(f"Sampling {n_samples} graphs")
            if desired_class is not None:
                print(f"Using classifier guidance toward class(es): {desired_class}")
        
        node_feats = self.conditioning_to_node_embeddings_generator.predict(
            self._sample_conditions(n_samples), desired_class=desired_class
        )
        return self.node_embeddings_to_graph_generator.decode(node_feats)

    @timeit
    def conditional_sample(
        self,
        graphs: List[nx.Graph],
        n_samples: int = 1,
        desired_class: Optional[Union[int, Sequence[int]]] = None
    ) -> List[List[nx.Graph]]:
        """Sample multiple graphs per input by conditioning on each graph's encoding."""
        _, cond_encs = self.encode(graphs)
        cond_encs = [[cond_enc]*n_samples for cond_enc in cond_encs]
        
        results = []
        for i in range(len(graphs)):
            y_i = cond_encs[i]
            node_feats = self.conditioning_to_node_embeddings_generator.predict(
                y_i, desired_class=desired_class
            )
            decoded = self.node_embeddings_to_graph_generator.decode(node_feats)
            results.append(decoded)
        
        return results

    def sample_from(self, graphs, n_samples=1):
        sampled_seed_graphs = random.choices(graphs, k=n_samples)
        reconstructed_graphs_list = self.conditional_sample(sampled_seed_graphs, n_samples=1)
        sampled_graphs = [reconstructed_graphs[0] for reconstructed_graphs in reconstructed_graphs_list]
        return sampled_graphs

    def interpolate(
        self,
        G1: nx.Graph,
        G2: nx.Graph,
        n_steps: int = 10,
        t_start: float = 0.0,
        t_end: float = 1.0
    ) -> List[nx.Graph]:
        """Interpolate between two graphs by SLERP-ing their condition vectors."""
        ts = np.linspace(t_start, t_end, n_steps)
        results = []
        emb1 = self.graph_vectorizer.transform([G1])[0]
        emb2 = self.graph_vectorizer.transform([G2])[0]
        for t in ts:
            emb_t = scaled_slerp(emb1, emb2, t)
            results.append(self.decode([emb_t])[0])
        
        return results

    def mean(
        self,
        graphs: List[nx.Graph]
    ) -> nx.Graph:
        """Compute a geometric mean graph via the SLERP barycentre of encodings."""
        Y = np.vstack(self.graph_vectorizer.transform(graphs))
        centroid = scaled_slerp_average(Y)
        return self.decode([centroid])[0]
    
    @timeit
    def fit_classifier(self, graphs, targets, epochs=20, lr=1e-3):
        """Train the optional guidance classifier on supplied graphs and labels."""
        # --- Step 1: Encode inputs ---
        node_encs = self.node_graph_vectorizer.transform(graphs)  # List of node arrays
        cond_vecs = self.graph_vectorizer.transform(graphs)       # 2D array

        # --- Step 2: Infer number of classes ---
        targets_np = targets.cpu().numpy() if isinstance(targets, torch.Tensor) else np.array(targets)
        num_classes = int(np.max(targets_np)) + 1

        # --- Step 3: Access underlying model ---
        model = self.conditioning_to_node_embeddings_generator.conditional_node_generator.model

        # --- Step 4: Ensure guidance classifier is initialized correctly ---
        if not hasattr(model, "guidance_classifier") or model.guidance_classifier is None:
            model.set_guidance_classifier(num_classes)
        else:
            try:
                current_dim = model.guidance_classifier.net[-1].out_features
            except AttributeError:
                current_dim = None
            if current_dim != num_classes:
                print(f"Resetting guidance classifier (was {current_dim}, now {num_classes})")
                model.set_guidance_classifier(num_classes)

        # --- Step 5: Train the classifier with internal validation and plot ---
        model.train_guidance_classifier(
            node_feats=node_encs,
            cond_vecs=cond_vecs,
            labels=targets_np,
            epochs=epochs,
            lr=lr,
            verbose=self.verbose
        )
