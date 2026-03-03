#!/usr/bin/env python
"""Graph encoder/decoder helpers used by the maintained conditional graph-generation pipeline."""

import copy
from dataclasses import dataclass
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


@dataclass(frozen=True)
class SupervisionChannelPlan:
    """Description of how one prediction channel should be handled during training."""

    name: str
    mode: str
    reason: str
    constant_value: Optional[Any] = None
    horizon: Optional[int] = None
    enabled: bool = False


@dataclass(frozen=True)
class SupervisionPlan:
    """Single source of truth for supervision decisions in the orchestration layer."""

    node_labels: SupervisionChannelPlan
    edge_labels: SupervisionChannelPlan
    direct_edges: SupervisionChannelPlan
    auxiliary_locality: SupervisionChannelPlan

    def as_dict(self) -> Dict[str, SupervisionChannelPlan]:
        return {
            "node_labels": self.node_labels,
            "edge_labels": self.edge_labels,
            "direct_edges": self.direct_edges,
            "auxiliary_locality": self.auxiliary_locality,
        }

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
    """Trainable toolkit for vectorising graphs and decoding them back into structure.

    Args:
        adjacency_matrix_classifier (LowRankMLP): Model that predicts adjacency
            probabilities between node pairs.
        node_label_classifier (LowRankMLP): Model used to score categorical node
            labels.
        edge_label_classifier (LowRankMLP): Model that predicts edge label
            distributions.
        verbose (bool): Toggle for printing solver and optimisation diagnostics.
        negative_sample_factor (int): Multiplier controlling how many non-edges to
            sample when building training batches.
        existence_threshold (float): Probability threshold above which edges or nodes
            are treated as present.
        num_augmentation_iterations (int): Number of noise augmentation passes applied
            to training graphs.
        augmentation_noise (float): Standard deviation of Gaussian noise injected
            during augmentation.
        enforce_connectivity (bool): Require the decoded graph to remain connected when
            solving the adjacency optimisation.
        degree_slack_penalty (float): Penalty weight for allowing slack in degree
            constraints during optimisation.
        warm_start_mst (bool): Whether to initialise the linear program with a maximum
            spanning tree warm start.
    """
    
    def __init__(
        self,
        adjacency_matrix_classifier: Optional[LowRankMLP] = None,
        node_label_classifier: Optional[LowRankMLP] = None,
        edge_label_classifier: Optional[LowRankMLP] = None,
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
        horizon: int = 1,
        supervision_name: str = "locality",
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
                print(
                    f"adj_mtx_to_targets[{supervision_name}, horizon={horizon}]: "
                    f"sampling {num_pairs_to_use} pairs ({locality_sample_fraction:.2%}) "
                    f"from {num_pairs} total pairs."
                )

            if 0 < num_pairs_to_use < num_pairs:
                indices = np.random.choice(num_pairs, num_pairs_to_use, replace=False)
                all_targets = [all_targets[i] for i in indices]
                all_pairs = [all_pairs[i] for i in indices]
            elif num_pairs_to_use == 0 and num_pairs > 0:
                if self.verbose:
                    print(
                        f"adj_mtx_to_targets[{supervision_name}, horizon={horizon}]: "
                        f"warning - num_pairs_to_use is 0 with locality_sample_fraction="
                        f"{locality_sample_fraction} and num_pairs={num_pairs}. No pairs will be used."
                    )
                return np.array([]), []
            elif num_pairs_to_use == 0 and num_pairs == 0:
                return np.array([]), []

        return np.array(all_targets), all_pairs

    def compute_edge_supervision(
        self, 
        graphs: List[nx.Graph], 
        node_encodings_list: List[np.ndarray],
        locality_sample_fraction: float,
        horizon: int = 1,
        supervision_name: str = "locality",
    ) -> Tuple[np.ndarray, List[Tuple[int, int, int]]]:
        """Compute locality supervision pairs for training."""
        adj = self.graphs_to_adjacency_matrices(graphs)
        return self.adj_mtx_to_targets(
            adj,
            node_encodings_list,
            locality_sample_fraction=locality_sample_fraction,
            is_training=True,
            horizon=horizon,
            supervision_name=supervision_name,
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
                    if graph.has_edge(u, v) and "label" in graph.edges[u, v]:
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
        """Fit the node-label classifier when the target is not constant."""
        X, y = self.encodings_and_graphs_to_node_label_dataset(node_encodings_list, graphs)
        unique_labels = np.unique(y)
        # If only one label is present, skip training.
        if len(unique_labels) == 1:
            if self.verbose:
                print("Node-label channel is constant in the training data: {}.".format(unique_labels[0]))
            return

        if self.node_label_classifier is None:
            return

        if self.verbose:
            print('Training node label predictor on {} instances with {} features'.format(X.shape[0], X.shape[1]))
        self.node_label_classifier.fit(X, y)

    @timeit
    def edge_label_classifier_fit(
        self,
        node_encodings_list: List[np.ndarray],
        graphs: List[nx.Graph]
    ) -> None:
        """Fit the edge-label classifier when the target is not constant."""
        X, y = self.encodings_and_graphs_to_edge_label_dataset(node_encodings_list, graphs)
        if len(y) == 0:
            if self.verbose:
                print("Edge-label channel is disabled: no usable edge labels were found in the training data.")
            return
        unique_labels = np.unique(y)
        # If only one edge label is found, store it and skip training.
        if len(unique_labels) == 1:
            if self.verbose:
                print("Edge-label channel is constant in the training data: {}.".format(unique_labels[0]))
            return

        if self.edge_label_classifier is None:
            return

        if self.verbose:
            print('Training edge label predictor on {} instances with {} features'.format(X.shape[0], X.shape[1]))
        self.edge_label_classifier.fit(X, y)

    @timeit    
    def adjacency_matrix_classifier_fit(
        self,
        node_encodings_list: List[np.ndarray],
        graphs: List[nx.Graph],
        locality_sample_fraction: float # <-- New parameter
    ) -> None:
        """Train the adjacency classifier on features derived from the provided graphs."""
        if self.adjacency_matrix_classifier is None:
            return
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
        locality_sample_fraction_for_adj_mtx: float,
        train_adjacency_matrix_classifier: bool = True,
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
        if train_adjacency_matrix_classifier:
            self.adjacency_matrix_classifier_fit(
                combined_encodings, combined_graphs,
                locality_sample_fraction=locality_sample_fraction_for_adj_mtx
            )
        elif self.verbose:
            print("Adjacency channel delegated to generator-owned direct-edge prediction.")

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
        predicted_edge_probability_matrices: Optional[List[np.ndarray]] = None,
        existence_threshold: float = 0.5
    ) -> List[np.ndarray]:
        """Project predicted adjacency probabilities onto valid binary graphs."""
        # Constrain encodings to be non-negative.
        node_encodings_list = self.constrained_node_encodings_list(original_node_encodings_list)
        channel_plan = self._plan_channel("direct_edges")

        predicted_probs_list: List[np.ndarray]
        if predicted_edge_probability_matrices is None:
            if self.adjacency_matrix_classifier is None:
                if channel_plan is not None and channel_plan.mode == "disabled":
                    raise RuntimeError(
                        "decode_adjacency_matrix cannot reconstruct graph structure because direct_edges are "
                        "disabled in the supervision plan and no adjacency_matrix_classifier is available."
                    )
                raise RuntimeError(
                    "decode_adjacency_matrix requires either predicted_edge_probability_matrices "
                    "or a fitted adjacency_matrix_classifier."
                )
            # Calculate the number of instances per graph (all ordered pairs except self-pairs).
            sizes = [enc.shape[0]**2 - enc.shape[0] for enc in node_encodings_list]
            # Generate instances for adjacency matrix prediction.
            X = self.encodings_to_instances(node_encodings_list)
            predicted_probs = self.adjacency_matrix_classifier.predict_proba(X)[:, -1]
            # Split predicted probabilities for each graph based on the sizes computed.
            predicted_probs_list = np.split(predicted_probs, np.cumsum(sizes)[:-1])
        else:
            if len(predicted_edge_probability_matrices) != len(node_encodings_list):
                raise ValueError(
                    "predicted_edge_probability_matrices must align with original_node_encodings_list "
                    f"(got {len(predicted_edge_probability_matrices)} matrices for {len(node_encodings_list)} graphs)."
                )
            predicted_probs_list = []
            for encodings, prob_matrix in zip(node_encodings_list, predicted_edge_probability_matrices):
                n_nodes = encodings.shape[0]
                prob_matrix = np.asarray(prob_matrix, dtype=float)
                if prob_matrix.shape != (n_nodes, n_nodes):
                    raise ValueError(
                        "Each predicted edge-probability matrix must have shape (n_nodes, n_nodes); "
                        f"received {prob_matrix.shape} for n_nodes={n_nodes}."
                    )
                mask = ~np.eye(n_nodes, dtype=bool)
                predicted_probs_list.append(prob_matrix[mask])
        
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
        channel_plan = self._plan_channel("node_labels")
        if channel_plan is not None and channel_plan.mode == "constant":
            predicted_node_labels_list = []
            for enc in node_encodings_list:
                n = enc.shape[0]
                predicted_node_labels_list.append(np.array([channel_plan.constant_value] * n, dtype=object))
            return predicted_node_labels_list

        if self.node_label_classifier is None:
            if channel_plan is not None and channel_plan.mode == "disabled":
                predicted_node_labels_list = []
                for enc in node_encodings_list:
                    n = enc.shape[0]
                    predicted_node_labels_list.append(np.array([None] * n, dtype=object))
                return predicted_node_labels_list
            raise RuntimeError(
                "decode_node_labels requires predicted node labels from the conditioning generator "
                "or a fitted node_label_classifier."
            )

        # Otherwise, predict labels for all nodes.
        X = np.vstack(node_encodings_list)
        predicted_node_labels = self.node_label_classifier.predict(X)
        sizes = [enc.shape[0] for enc in node_encodings_list]
        predicted_node_labels_list = np.split(predicted_node_labels, np.cumsum(sizes)[:-1])
        return predicted_node_labels_list

    def decode_edge_labels(
        self,
        node_encodings_list: List[np.ndarray],
        adj_mtx_list: List[np.ndarray],
        predicted_edge_label_matrices: Optional[List[np.ndarray]] = None,
    ) -> List[np.ndarray]:
        """Predict edge labels for every edge present in the supplied adjacency matrices."""
        channel_plan = self._plan_channel("edge_labels")
        if predicted_edge_label_matrices is not None:
            if len(predicted_edge_label_matrices) != len(adj_mtx_list):
                raise ValueError(
                    "predicted_edge_label_matrices must align with adj_mtx_list "
                    f"(got {len(predicted_edge_label_matrices)} matrices for {len(adj_mtx_list)} graphs)."
                )
            predicted_edge_labels_list = []
            for adj_mtx, edge_label_matrix in zip(adj_mtx_list, predicted_edge_label_matrices):
                edge_label_matrix = np.asarray(edge_label_matrix, dtype=object)
                if edge_label_matrix.shape != adj_mtx.shape:
                    raise ValueError(
                        "Each predicted edge-label matrix must have the same shape as its adjacency matrix; "
                        f"received {edge_label_matrix.shape} and {adj_mtx.shape}."
                    )
                edge_labels = []
                n_nodes = adj_mtx.shape[0]
                for i in range(n_nodes):
                    for j in range(n_nodes):
                        if adj_mtx[i, j] != 0:
                            edge_labels.append(edge_label_matrix[i, j])
                predicted_edge_labels_list.append(np.asarray(edge_labels, dtype=object))
            return predicted_edge_labels_list

        if channel_plan is not None and channel_plan.mode == "constant":
            predicted_edge_labels_list = []
            for adj in adj_mtx_list:
                n_edges = int(np.sum(adj))
                predicted_edge_labels_list.append(np.array([channel_plan.constant_value] * n_edges, dtype=object))
            return predicted_edge_labels_list

        if self.edge_label_classifier is None:
            if channel_plan is not None and channel_plan.mode == "disabled":
                return [np.asarray([], dtype=object) for _ in node_encodings_list]
            if int(self.verbose) >= 2:
                print(
                    "No edge-label predictor is available; returning graphs without edge labels."
                )
            return [np.asarray([], dtype=object) for _ in node_encodings_list]

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
        original_node_encodings_list: List[np.ndarray],
        predicted_node_labels_list: Optional[List[np.ndarray]] = None,
        predicted_edge_probability_matrices: Optional[List[np.ndarray]] = None,
        predicted_edge_label_matrices: Optional[List[np.ndarray]] = None,
    ) -> List[nx.Graph]:
        """Reconstruct labelled graphs from node encodings, respecting existence masks."""
        # Step 1: Decode the adjacency matrices using the modified method that accounts for node existence.
        adj_mtx_list = self.decode_adjacency_matrix(
            original_node_encodings_list,
            predicted_edge_probability_matrices=predicted_edge_probability_matrices,
            existence_threshold=self.existence_threshold,
        )
        
        # Step 2: Decode node labels unless they were already predicted by the conditional generator.
        if predicted_node_labels_list is None:
            predicted_node_labels_list = self.decode_node_labels(original_node_encodings_list)
        
        # Step 3: Decode edge labels based on the updated adjacency matrices.
        predicted_edge_labels_list = self.decode_edge_labels(
            original_node_encodings_list,
            adj_mtx_list,
            predicted_edge_label_matrices=predicted_edge_label_matrices,
        )
        
        graphs = []
        # Step 4: Reconstruct each graph and filter out non-existent nodes.
        for encodings, node_labels, edge_labels, adj_mtx in zip(
                original_node_encodings_list, predicted_node_labels_list, predicted_edge_labels_list, adj_mtx_list):
            # Create the initial graph from the predicted adjacency matrix.
            graph = nx.from_numpy_array(adj_mtx)
            
            # Assign node labels. (Assumes ordering in node_labels matches node indices.)
            if len(node_labels) > 0 and not all(label is None for label in node_labels):
                node_label_map = {i: label for i, label in enumerate(node_labels)}
                nx.set_node_attributes(graph, node_label_map, 'label')
            
            # If edges exist, assign edge labels.
            if np.sum(adj_mtx) > 0 and len(edge_labels) > 0:
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
        edge_label_pairs: Optional[List[Tuple[int, int, int]]] = None,
        edge_label_targets: Optional[np.ndarray] = None,
        auxiliary_edge_pairs: Optional[List[Tuple[int, int, int]]] = None,
        auxiliary_edge_targets: Optional[np.ndarray] = None,
        node_mask: Optional[np.ndarray] = None,
        node_label_targets: Optional[List[np.ndarray]] = None,
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
                edge_label_pairs=edge_label_pairs,
                edge_label_targets=edge_label_targets,
                auxiliary_edge_pairs=auxiliary_edge_pairs,
                auxiliary_edge_targets=auxiliary_edge_targets,
                node_mask=node_mask,
                node_label_targets=node_label_targets,
            )
            self.conditional_node_generator.fit(
                node_encodings_list=node_encodings_list,
                conditional_graph_encodings=conditional_graph_encodings,
                edge_pairs=edge_pairs,
                edge_targets=edge_targets,
                edge_label_pairs=edge_label_pairs,
                edge_label_targets=edge_label_targets,
                auxiliary_edge_pairs=auxiliary_edge_pairs,
                auxiliary_edge_targets=auxiliary_edge_targets,
                node_mask=node_mask,
                node_label_targets=node_label_targets,
            )
        else:
            self.conditional_node_generator.setup(
                node_encodings_list=node_encodings_list,
                conditional_graph_encodings=conditional_graph_encodings,
                edge_label_pairs=edge_label_pairs,
                edge_label_targets=edge_label_targets,
                auxiliary_edge_pairs=auxiliary_edge_pairs,
                auxiliary_edge_targets=auxiliary_edge_targets,
                node_mask=node_mask,
                node_label_targets=node_label_targets,
            )
            self.conditional_node_generator.fit(
                node_encodings_list=node_encodings_list,
                conditional_graph_encodings=conditional_graph_encodings,
                edge_label_pairs=edge_label_pairs,
                edge_label_targets=edge_label_targets,
                auxiliary_edge_pairs=auxiliary_edge_pairs,
                auxiliary_edge_targets=auxiliary_edge_targets,
                node_mask=node_mask,
                node_label_targets=node_label_targets,
            )

        return self

    @timeit
    def predict(
        self,
        conditional_graph_encodings: Any,
        desired_class: Optional[Union[int, Sequence[int]]] = None,
    ) -> List[np.ndarray]:
        if int(self.verbose) >= 2:
            print(f"Predicting node matrices for {len(conditional_graph_encodings)} graphs...")
        predicted_node_encodings_list = self.conditional_node_generator.predict(
            conditional_graph_encodings,
            desired_class=desired_class,
        )
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
        self.node_label_classes_ = None
        self.node_label_to_index_ = None
        self.node_label_histogram_dimension = 0
        self.supervision_plan_: Optional[SupervisionPlan] = None
        if not 0.0 < locality_sample_fraction <= 1.0:
            raise ValueError("locality_sample_fraction must be between 0.0 (exclusive) and 1.0 (inclusive)")
        self.locality_sample_fraction = locality_sample_fraction
        if locality_horizon < 1:
            raise ValueError("locality_horizon must be >= 1")
        self.locality_horizon = locality_horizon

    def _build_supervision_plan(
        self,
        graphs: List[nx.Graph],
        node_label_targets: List[np.ndarray],
        edge_label_targets: Optional[np.ndarray],
    ) -> SupervisionPlan:
        """Build a single explicit supervision plan for the whole fit() call."""
        del graphs

        flat_node_labels = [
            label
            for labels in node_label_targets
            for label in np.asarray(labels, dtype=object).tolist()
        ]
        if len(flat_node_labels) == 0:
            node_label_mode = "disabled"
            node_label_reason = "No node labels were provided."
            node_label_constant = None
        else:
            unique_node_labels = np.unique(np.asarray(flat_node_labels, dtype=object))
            if len(unique_node_labels) == 1:
                node_label_mode = "constant"
                node_label_reason = "All training nodes share one label."
                node_label_constant = unique_node_labels[0]
            else:
                node_label_mode = "learned"
                node_label_reason = f"{len(unique_node_labels)} node labels detected."
                node_label_constant = None

        if edge_label_targets is None:
            edge_label_mode = "disabled"
            edge_label_reason = "No usable edge labels were provided."
            edge_label_constant = None
        else:
            unique_edge_labels = np.unique(np.asarray(edge_label_targets, dtype=object))
            if len(unique_edge_labels) == 1:
                edge_label_mode = "constant"
                edge_label_reason = "All labelled edges share one label."
                edge_label_constant = unique_edge_labels[0]
            else:
                edge_label_mode = "learned"
                edge_label_reason = f"{len(unique_edge_labels)} edge labels detected."
                edge_label_constant = None

        direct_edges_enabled = bool(self.use_locality_supervision)
        if direct_edges_enabled:
            direct_edges_mode = "learned"
            direct_edges_reason = "Generator should learn horizon-1 edge presence for the decoder."
        else:
            direct_edges_mode = "disabled"
            direct_edges_reason = "Direct edge supervision was not requested."

        aux_enabled = bool(self.use_locality_supervision and self.locality_horizon > 1)
        if aux_enabled:
            auxiliary_mode = "learned"
            auxiliary_reason = f"Use horizon-{self.locality_horizon} locality as auxiliary regularization."
        elif self.use_locality_supervision:
            auxiliary_mode = "disabled"
            auxiliary_reason = "No auxiliary locality is needed when locality_horizon=1."
        else:
            auxiliary_mode = "disabled"
            auxiliary_reason = "Auxiliary locality depends on direct edge supervision being enabled."

        return SupervisionPlan(
            node_labels=SupervisionChannelPlan(
                name="node_labels",
                mode=node_label_mode,
                reason=node_label_reason,
                constant_value=node_label_constant,
                enabled=node_label_mode != "disabled",
            ),
            edge_labels=SupervisionChannelPlan(
                name="edge_labels",
                mode=edge_label_mode,
                reason=edge_label_reason,
                constant_value=edge_label_constant,
                enabled=edge_label_mode != "disabled",
            ),
            direct_edges=SupervisionChannelPlan(
                name="direct_edges",
                mode=direct_edges_mode,
                reason=direct_edges_reason,
                horizon=1,
                enabled=direct_edges_enabled,
            ),
            auxiliary_locality=SupervisionChannelPlan(
                name="auxiliary_locality",
                mode=auxiliary_mode,
                reason=auxiliary_reason,
                horizon=self.locality_horizon if aux_enabled else None,
                enabled=aux_enabled,
            ),
        )

    def _log_supervision_plan(self, supervision_plan: SupervisionPlan) -> None:
        """Print the current supervision plan when verbose logging is enabled."""
        if not self.verbose:
            return
        print("Supervision plan:")
        for channel in supervision_plan.as_dict().values():
            enabled_text = "enabled" if channel.enabled else "disabled"
            horizon_text = f", horizon={channel.horizon}" if channel.horizon is not None else ""
            print(f"  {channel.name}: mode={channel.mode}, {enabled_text}{horizon_text}. {channel.reason}")

    def _plan_channel(self, channel_name: str) -> Optional[SupervisionChannelPlan]:
        """Return the named supervision channel when a plan is available."""
        plan = getattr(self, "supervision_plan_", None)
        if plan is None:
            return None
        return getattr(plan, channel_name, None)

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
        node_label_targets = self.graphs_to_node_label_targets(graphs)
        edge_label_targets, edge_label_pairs = self.graphs_to_edge_label_targets(graphs)
        self._fit_node_label_vocab(node_label_targets)
        supervision_plan = self._build_supervision_plan(
            graphs,
            node_label_targets=node_label_targets,
            edge_label_targets=edge_label_targets,
        )
        self.supervision_plan_ = supervision_plan
        if self.conditioning_to_node_embeddings_generator is not None:
            setattr(self.conditioning_to_node_embeddings_generator, "supervision_plan_", supervision_plan)
        if self.node_embeddings_to_graph_generator is not None:
            setattr(self.node_embeddings_to_graph_generator, "supervision_plan_", supervision_plan)

        # Generate encodings
        node_encodings_list, conditional_graph_encodings = self.encode(graphs)

        if train_conditioning_to_node_embeddings_generator:
            edge_pairs_for_cond_gen = None
            edge_targets_for_cond_gen = None
            auxiliary_edge_pairs_for_cond_gen = None
            auxiliary_edge_targets_for_cond_gen = None
            node_mask_for_cond_gen = None # Assuming node_mask might be needed by ConditionalNodeGeneratorModel

            if supervision_plan.direct_edges.enabled:
                if self.node_embeddings_to_graph_generator is None:
                    raise RuntimeError("Locality supervision requested but node_embeddings_to_graph_generator is None.")
                self._log_supervision_plan(supervision_plan)

                edge_targets_for_cond_gen, edge_pairs_for_cond_gen = self.node_embeddings_to_graph_generator.compute_edge_supervision(
                    graphs,
                    node_encodings_list,
                    locality_sample_fraction=self.locality_sample_fraction,
                    horizon=1,
                    supervision_name="direct_edge",
                )
                if supervision_plan.auxiliary_locality.enabled:
                    auxiliary_edge_targets_for_cond_gen, auxiliary_edge_pairs_for_cond_gen = (
                        self.node_embeddings_to_graph_generator.compute_edge_supervision(
                            graphs,
                            node_encodings_list,
                            locality_sample_fraction=self.locality_sample_fraction,
                            horizon=supervision_plan.auxiliary_locality.horizon,
                            supervision_name="aux_locality",
                        )
                    )
            else:
                self._log_supervision_plan(supervision_plan)
            
            self.conditioning_to_node_embeddings_generator.fit(
                node_encodings_list=node_encodings_list,
                conditional_graph_encodings=conditional_graph_encodings,
                edge_pairs=edge_pairs_for_cond_gen,
                edge_targets=edge_targets_for_cond_gen,
                edge_label_pairs=edge_label_pairs if supervision_plan.edge_labels.enabled else None,
                edge_label_targets=edge_label_targets if supervision_plan.edge_labels.enabled else None,
                auxiliary_edge_pairs=auxiliary_edge_pairs_for_cond_gen,
                auxiliary_edge_targets=auxiliary_edge_targets_for_cond_gen,
                node_mask=node_mask_for_cond_gen, # Pass if available/needed
                node_label_targets=node_label_targets if supervision_plan.node_labels.enabled else None,
            )

        if train_node_embeddings_to_graph_generator:
            self.node_embeddings_to_graph_generator.fit(
                graphs,
                node_encodings_list,
                locality_sample_fraction_for_adj_mtx=self.locality_sample_fraction,
                train_adjacency_matrix_classifier=not self._generator_predicts_edge_presence(),
            )

        return self

    @timeit
    def node_encode(self, graphs: List[nx.Graph]) -> List[np.ndarray]:
        """Transform graphs into per-node embedding matrices."""
        if int(self.verbose) >= 2:
            print(f"Node encoding {len(graphs)} graphs")
        return self.node_graph_vectorizer.transform(graphs)

    @timeit
    def graph_encode(self, graphs: List[nx.Graph]) -> List[np.ndarray]:
        """Transform graphs into global conditioning vectors."""
        if int(self.verbose) >= 2:
            print(f"Encoding {len(graphs)} graphs")
        cond_encodings = np.asarray(self.graph_vectorizer.transform(graphs))
        if not self._should_use_node_label_histograms():
            return cond_encodings
        histograms = self.graphs_to_node_label_histograms(graphs)
        if histograms is None:
            return cond_encodings
        return self._augment_condition_encodings(cond_encodings, histograms)

    def encode(self, graphs: List[nx.Graph]) -> Tuple[List[np.ndarray], Any]:
        """Produce both node-level encodings and their matching conditioning vectors."""
        node_encs = self.node_encode(graphs)
        cond_encs = self.graph_encode(graphs)
        return node_encs, cond_encs

    def graphs_to_node_label_targets(self, graphs: List[nx.Graph]) -> List[np.ndarray]:
        """Extract per-node categorical labels in the node ordering used elsewhere."""
        node_label_targets = []
        for graph in graphs:
            node_label_targets.append(np.asarray([graph.nodes[u]["label"] for u in graph.nodes()], dtype=object))
        return node_label_targets

    def _graphs_have_usable_edge_labels(self, graphs: List[nx.Graph]) -> bool:
        """Return True only when every observed edge carries a label and at least one labelled edge exists."""
        saw_any_edge = False
        for graph in graphs:
            for u, v, attrs in graph.edges(data=True):
                saw_any_edge = True
                if "label" not in attrs:
                    return False
        return saw_any_edge

    def graphs_to_edge_label_targets(
        self,
        graphs: List[nx.Graph],
    ) -> Tuple[Optional[np.ndarray], Optional[List[Tuple[int, int, int]]]]:
        """Extract per-edge categorical labels in the ordered node-pair convention used elsewhere."""
        if not self._graphs_have_usable_edge_labels(graphs):
            if self.verbose:
                print("Edge-label channel disabled at graph inspection time: no usable edge labels were found.")
            return None, None
        edge_label_targets = []
        edge_label_pairs = []
        for graph_idx, graph in enumerate(graphs):
            nodes = list(graph.nodes())
            for i, u in enumerate(nodes):
                for j, v in enumerate(nodes):
                    if graph.has_edge(u, v):
                        edge_label_pairs.append((graph_idx, i, j))
                        edge_label_targets.append(graph.edges[u, v]["label"])
        return np.asarray(edge_label_targets, dtype=object), edge_label_pairs

    def _should_use_node_label_histograms(self) -> bool:
        generator = getattr(self.conditioning_to_node_embeddings_generator, "conditional_node_generator", None)
        if generator is None:
            return False
        return bool(getattr(generator, "require_embedded_node_label_histogram", False))

    def _generator_predicts_edge_presence(self) -> bool:
        generator_model = self.conditioning_to_node_embeddings_generator
        if generator_model is None:
            return False
        conditional_generator = getattr(generator_model, "conditional_node_generator", None)
        if conditional_generator is None:
            return False
        return bool(getattr(conditional_generator, "use_locality_supervision", False))

    def _fit_node_label_vocab(self, node_label_targets: List[np.ndarray]) -> None:
        if not self._should_use_node_label_histograms():
            self.node_label_classes_ = None
            self.node_label_to_index_ = None
            self.node_label_histogram_dimension = 0
            return
        flat_labels = [label for labels in node_label_targets for label in np.asarray(labels, dtype=object).tolist()]
        if len(flat_labels) == 0:
            self.node_label_classes_ = np.asarray([], dtype=object)
            self.node_label_to_index_ = {}
            self.node_label_histogram_dimension = 0
            return
        self.node_label_classes_ = np.unique(np.asarray(flat_labels, dtype=object))
        self.node_label_to_index_ = {label: idx for idx, label in enumerate(self.node_label_classes_)}
        self.node_label_histogram_dimension = int(len(self.node_label_classes_))

    def _augment_condition_encodings(self, cond_encodings: np.ndarray, histograms: np.ndarray) -> np.ndarray:
        if histograms is None or self.node_label_histogram_dimension == 0:
            return cond_encodings
        if cond_encodings.ndim == 2:
            return np.concatenate([cond_encodings, histograms], axis=1)
        if cond_encodings.ndim == 3:
            hist_tokens = np.repeat(histograms[:, None, :], cond_encodings.shape[1], axis=1)
            return np.concatenate([cond_encodings, hist_tokens], axis=2)
        raise ValueError(
            "graph conditioning vectors must have shape (B, C) or (B, M, C); "
            f"received {cond_encodings.shape}"
        )

    def graphs_to_node_label_histograms(self, graphs: List[nx.Graph]) -> Optional[np.ndarray]:
        """Convert graphs into label histograms using the fitted graph-level label vocabulary."""
        if not self._should_use_node_label_histograms():
            return None
        if self.node_label_to_index_ is None:
            return None
        histograms = []
        num_classes = self.node_label_histogram_dimension
        for labels in self.graphs_to_node_label_targets(graphs):
            hist = np.zeros(num_classes, dtype=float)
            if labels.size > 0:
                for label in labels:
                    class_idx = self.node_label_to_index_.get(label)
                    if class_idx is not None:
                        hist[class_idx] += 1.0
                total = hist.sum()
                if total > 0:
                    hist /= total
            histograms.append(hist)
        return np.asarray(histograms, dtype=float)

    def _get_predicted_node_labels_from_generator(self) -> Optional[List[np.ndarray]]:
        generator_model = self.conditioning_to_node_embeddings_generator
        if generator_model is None:
            return None
        conditional_generator = getattr(generator_model, "conditional_node_generator", None)
        if conditional_generator is None:
            return None
        predicted = getattr(conditional_generator, "last_predicted_node_label_classes_", None)
        if predicted is None:
            return None
        return [np.asarray(labels, dtype=object) for labels in predicted]

    def _get_predicted_edge_probabilities_from_generator(self) -> Optional[List[np.ndarray]]:
        generator_model = self.conditioning_to_node_embeddings_generator
        if generator_model is None:
            return None
        conditional_generator = getattr(generator_model, "conditional_node_generator", None)
        if conditional_generator is None:
            return None
        predicted = getattr(conditional_generator, "last_predicted_edge_probability_matrices_", None)
        if predicted is None:
            return None
        return [np.asarray(prob_matrix, dtype=float) for prob_matrix in predicted]

    def _get_predicted_edge_labels_from_generator(self) -> Optional[List[np.ndarray]]:
        generator_model = self.conditioning_to_node_embeddings_generator
        if generator_model is None:
            return None
        conditional_generator = getattr(generator_model, "conditional_node_generator", None)
        if conditional_generator is None:
            return None
        predicted = getattr(conditional_generator, "last_predicted_edge_label_matrices_", None)
        if predicted is None:
            return None
        return [np.asarray(label_matrix, dtype=object) for label_matrix in predicted]

    def _resolve_generated_node_labels(
        self,
        node_feats: List[np.ndarray],
    ) -> Optional[List[np.ndarray]]:
        channel_plan = self._plan_channel("node_labels")
        if channel_plan is not None and channel_plan.mode != "learned":
            return None
        predicted_node_labels_list = self._get_predicted_node_labels_from_generator()
        if predicted_node_labels_list is None:
            return None
        if len(predicted_node_labels_list) != len(node_feats):
            return None
        for labels, feats in zip(predicted_node_labels_list, node_feats):
            if len(labels) != feats.shape[0]:
                return None
        return predicted_node_labels_list

    def _resolve_generated_edge_probabilities(
        self,
        node_feats: List[np.ndarray],
    ) -> Optional[List[np.ndarray]]:
        channel_plan = self._plan_channel("direct_edges")
        if channel_plan is not None and channel_plan.mode != "learned":
            return None
        predicted_edge_probability_matrices = self._get_predicted_edge_probabilities_from_generator()
        if predicted_edge_probability_matrices is None:
            return None
        if len(predicted_edge_probability_matrices) != len(node_feats):
            return None
        for prob_matrix, feats in zip(predicted_edge_probability_matrices, node_feats):
            if prob_matrix.shape != (feats.shape[0], feats.shape[0]):
                return None
        return predicted_edge_probability_matrices

    def _resolve_generated_edge_labels(
        self,
        node_feats: List[np.ndarray],
    ) -> Optional[List[np.ndarray]]:
        channel_plan = self._plan_channel("edge_labels")
        if channel_plan is not None and channel_plan.mode != "learned":
            return None
        predicted_edge_label_matrices = self._get_predicted_edge_labels_from_generator()
        if predicted_edge_label_matrices is None:
            return None
        if len(predicted_edge_label_matrices) != len(node_feats):
            return None
        for edge_label_matrix, feats in zip(predicted_edge_label_matrices, node_feats):
            if edge_label_matrix.shape != (feats.shape[0], feats.shape[0]):
                return None
        return predicted_edge_label_matrices

    def decode(
        self,
        conditioning_vectors: Any,
        desired_class: Optional[Union[int, Sequence[int]]] = None,
    ) -> List[nx.Graph]:
        """Decode conditioning vectors into reconstructed graphs."""
        if self.verbose:
            print(f"Decoding {len(conditioning_vectors)} conditioning vectors")
            if desired_class is not None:
                print(f"Using classifier guidance toward class(es): {desired_class}")
        
        node_feats = self.conditioning_to_node_embeddings_generator.predict(
            conditioning_vectors,
            desired_class=desired_class,
        )
        predicted_node_labels_list = self._resolve_generated_node_labels(node_feats)
        predicted_edge_probability_matrices = self._resolve_generated_edge_probabilities(node_feats)
        predicted_edge_label_matrices = self._resolve_generated_edge_labels(node_feats)
        return self.node_embeddings_to_graph_generator.decode(
            node_feats,
            predicted_node_labels_list=predicted_node_labels_list,
            predicted_edge_probability_matrices=predicted_edge_probability_matrices,
            predicted_edge_label_matrices=predicted_edge_label_matrices,
        )

    @timeit
    def sample(self, n_samples: int = 1, desired_class: Optional[Union[int, Sequence[int]]] = None) -> List[nx.Graph]:
        """Generate random graphs by sampling conditioning vectors from the prior."""
        if self.verbose:
            print(f"Sampling {n_samples} graphs")
            if desired_class is not None:
                print(f"Using classifier guidance toward class(es): {desired_class}")
        
        node_feats = self.conditioning_to_node_embeddings_generator.predict(
            self._sample_conditions(n_samples),
            desired_class=desired_class,
        )
        predicted_node_labels_list = self._resolve_generated_node_labels(node_feats)
        predicted_edge_probability_matrices = self._resolve_generated_edge_probabilities(node_feats)
        predicted_edge_label_matrices = self._resolve_generated_edge_labels(node_feats)
        return self.node_embeddings_to_graph_generator.decode(
            node_feats,
            predicted_node_labels_list=predicted_node_labels_list,
            predicted_edge_probability_matrices=predicted_edge_probability_matrices,
            predicted_edge_label_matrices=predicted_edge_label_matrices,
        )

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
                y_i,
                desired_class=desired_class,
            )
            predicted_node_labels_list = self._resolve_generated_node_labels(node_feats)
            predicted_edge_probability_matrices = self._resolve_generated_edge_probabilities(node_feats)
            predicted_edge_label_matrices = self._resolve_generated_edge_labels(node_feats)
            decoded = self.node_embeddings_to_graph_generator.decode(
                node_feats,
                predicted_node_labels_list=predicted_node_labels_list,
                predicted_edge_probability_matrices=predicted_edge_probability_matrices,
                predicted_edge_label_matrices=predicted_edge_label_matrices,
            )
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
        emb1 = self.graph_encode([G1])[0]
        emb2 = self.graph_encode([G2])[0]
        for t in ts:
            emb_t = scaled_slerp(emb1, emb2, t)
            results.append(self.decode([emb_t])[0])
        
        return results

    def mean(
        self,
        graphs: List[nx.Graph]
    ) -> nx.Graph:
        """Compute a geometric mean graph via the SLERP barycentre of encodings."""
        Y = np.vstack(self.graph_encode(graphs))
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
