#!/usr/bin/env python
"""Graph encoder/decoder helpers used by the maintained conditional graph-generation pipeline."""

from dataclasses import dataclass
import numpy as np
import networkx as nx
import random
import pulp
import dill as pickle
from .timeit import timeit
from typing import List, Tuple, Optional, Any, Sequence, Dict, Union
from .conditional_node_generator_base import (
    ConditionalNodeGeneratorBase,
    GeneratedNodeBatch,
    GraphConditioningBatch,
    NodeGenerationBatch,
)


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
# EqMDecompositionalGraphDecoder Class
# =============================================================================

class EqMDecompositionalGraphDecoder(object):
    """Graph decoder that turns generator outputs into final NetworkX graphs."""
    
    def __init__(
        self,
        verbose: bool = True,
        existence_threshold: float = 0.5,
        enforce_connectivity: bool = True,
        degree_slack_penalty: float = 1e6,
        warm_start_mst: bool = True,
    ) -> None:
        """Store graph decoding hyper-parameters."""
        self.verbose                    = verbose
        self.existence_threshold        = existence_threshold
        self.enforce_connectivity       = enforce_connectivity
        self.degree_slack_penalty       = degree_slack_penalty
        self.warm_start_mst             = warm_start_mst
        self.supervision_plan_          = None

    def _plan_channel(self, channel_name: str) -> Optional[SupervisionChannelPlan]:
        """Return the named supervision channel when a plan is available."""
        plan = getattr(self, "supervision_plan_", None)
        if plan is None:
            return None
        return getattr(plan, channel_name, None)

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

    def _target_stats(self, targets: List[int]) -> Tuple[int, int]:
        """Count positive and negative supervision pairs."""
        positive = int(sum(1 for target in targets if int(target) == 1))
        negative = int(len(targets) - positive)
        return positive, negative

    def _sample_pair_indices(
        self,
        targets: List[int],
        sample_count: int,
        locality_sampling_strategy: str,
        locality_target_positive_ratio: Optional[float],
    ) -> np.ndarray:
        """Sample supervision indices using the configured class-balancing strategy."""
        num_pairs = len(targets)
        if sample_count <= 0:
            return np.asarray([], dtype=int)
        if sample_count >= num_pairs:
            return np.arange(num_pairs, dtype=int)

        targets_array = np.asarray(targets, dtype=int)
        if locality_sampling_strategy == "uniform":
            return np.random.choice(num_pairs, sample_count, replace=False)

        pos_indices = np.flatnonzero(targets_array == 1)
        neg_indices = np.flatnonzero(targets_array == 0)
        if len(pos_indices) == 0 or len(neg_indices) == 0:
            return np.random.choice(num_pairs, sample_count, replace=False)

        if locality_sampling_strategy == "stratified_target":
            target_positive_ratio = locality_target_positive_ratio
            if target_positive_ratio is None:
                raise ValueError(
                    "locality_sampling_strategy='stratified_target' requires locality_target_positive_ratio."
                )
        else:
            target_positive_ratio = len(pos_indices) / float(num_pairs)

        num_pos = int(round(sample_count * target_positive_ratio))
        num_pos = max(0, min(num_pos, sample_count))
        num_neg = sample_count - num_pos

        num_pos = min(num_pos, len(pos_indices))
        num_neg = min(num_neg, len(neg_indices))

        remaining = sample_count - (num_pos + num_neg)
        if remaining > 0:
            extra_pos = min(remaining, len(pos_indices) - num_pos)
            num_pos += extra_pos
            remaining -= extra_pos
        if remaining > 0:
            extra_neg = min(remaining, len(neg_indices) - num_neg)
            num_neg += extra_neg

        sampled_pos = np.random.choice(pos_indices, num_pos, replace=False) if num_pos > 0 else np.asarray([], dtype=int)
        sampled_neg = np.random.choice(neg_indices, num_neg, replace=False) if num_neg > 0 else np.asarray([], dtype=int)
        sampled = np.concatenate([sampled_pos, sampled_neg])
        np.random.shuffle(sampled)
        return sampled

    def adj_mtx_to_targets(
        self,
        adj_mtx_list: List[np.ndarray],
        node_encodings_list: List[np.ndarray],
        locality_sample_fraction: float,
        negative_sample_factor: int = 1,
        locality_sampling_strategy: str = "stratified_preserve",
        locality_target_positive_ratio: Optional[float] = None,
        force_bi_directional_edges: bool = True,
        is_training: bool = False,
        horizon: int = 1,
        supervision_name: str = "locality",
    ) -> Tuple[np.ndarray, List[Tuple[int, int, int]]]:
        """Label node pairs as local or non-local using shortest-path distance within each graph."""
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        valid_sampling_strategies = {"uniform", "stratified_preserve", "stratified_target"}
        if locality_sampling_strategy not in valid_sampling_strategies:
            raise ValueError(
                f"locality_sampling_strategy must be one of {sorted(valid_sampling_strategies)} "
                f"(got {locality_sampling_strategy!r})."
            )
        if locality_target_positive_ratio is not None and not 0.0 < locality_target_positive_ratio < 1.0:
            raise ValueError("locality_target_positive_ratio must be between 0 and 1 when provided.")

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
                num_neg_samples = int(round(negative_sample_factor * num_pos))
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
        pos_before, neg_before = self._target_stats(all_targets)
        if is_training and locality_sample_fraction < 1.0:
            num_pairs = len(all_pairs)
            num_pairs_to_use = int(round(num_pairs * locality_sample_fraction))

            if self.verbose and num_pairs > 0:
                print(
                    f"adj_mtx_to_targets[{supervision_name}, horizon={horizon}]: "
                    f"sampling {num_pairs_to_use} pairs ({locality_sample_fraction:.2%}) "
                    f"from {num_pairs} total pairs "
                    f"(pos={pos_before}, neg={neg_before}, "
                    f"negative_sample_factor={negative_sample_factor}, "
                    f"sampling_strategy={locality_sampling_strategy}"
                    f"{'' if locality_target_positive_ratio is None else f', target_positive_ratio={locality_target_positive_ratio:.3f}'})."
                )

            if 0 < num_pairs_to_use < num_pairs:
                indices = self._sample_pair_indices(
                    all_targets,
                    num_pairs_to_use,
                    locality_sampling_strategy=locality_sampling_strategy,
                    locality_target_positive_ratio=locality_target_positive_ratio,
                )
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

        if self.verbose and len(all_targets) > 0:
            pos_after, neg_after = self._target_stats(all_targets)
            ratio_after = pos_after / float(pos_after + neg_after)
            print(
                f"adj_mtx_to_targets[{supervision_name}, horizon={horizon}]: "
                f"using pos={pos_after}, neg={neg_after}, positive_ratio={ratio_after:.3f}."
            )

        return np.array(all_targets), all_pairs

    def compute_edge_supervision(
        self, 
        graphs: List[nx.Graph], 
        node_encodings_list: List[np.ndarray],
        locality_sample_fraction: float,
        negative_sample_factor: int = 1,
        locality_sampling_strategy: str = "stratified_preserve",
        locality_target_positive_ratio: Optional[float] = None,
        horizon: int = 1,
        supervision_name: str = "locality",
    ) -> Tuple[np.ndarray, List[Tuple[int, int, int]]]:
        """Compute locality supervision pairs for training."""
        adj = self.graphs_to_adjacency_matrices(graphs)
        return self.adj_mtx_to_targets(
            adj,
            node_encodings_list,
            locality_sample_fraction=locality_sample_fraction,
            negative_sample_factor=negative_sample_factor,
            locality_sampling_strategy=locality_sampling_strategy,
            locality_target_positive_ratio=locality_target_positive_ratio,
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
        degree_predictions: np.ndarray,
        existence_mask: np.ndarray,
    ) -> List[int]:
        """Derive integer degree targets from explicit degree and existence predictions."""
        degs = np.rint(np.asarray(degree_predictions, dtype=float))
        existent = np.asarray(existence_mask, dtype=bool)
        degs = np.where(existent, np.maximum(degs, 1), 0)
        return degs.astype(int).tolist()

    def decode_adjacency_matrix(
        self,
        generated_nodes: GeneratedNodeBatch,
        predicted_edge_probability_matrices: Optional[List[np.ndarray]] = None,
    ) -> List[np.ndarray]:
        """Project predicted adjacency probabilities onto valid binary graphs."""
        node_embeddings_list = self.constrained_node_encodings_list(generated_nodes.node_embeddings_list)
        channel_plan = self._plan_channel("direct_edges")
        existence_masks = generated_nodes.node_presence_mask
        degree_predictions = generated_nodes.node_degree_predictions

        if existence_masks is None:
            raise RuntimeError("decode_adjacency_matrix requires explicit node_presence_mask predictions.")
        if degree_predictions is None:
            raise RuntimeError("decode_adjacency_matrix requires explicit node_degree_predictions.")

        predicted_probs_list: List[np.ndarray]
        if predicted_edge_probability_matrices is None:
            if channel_plan is not None and channel_plan.mode == "disabled":
                raise RuntimeError(
                    "decode_adjacency_matrix cannot reconstruct graph structure because direct_edges are disabled "
                    "in the supervision plan."
                )
                raise RuntimeError(
                    "decode_adjacency_matrix requires generator-provided edge probabilities."
                )
        else:
            if len(predicted_edge_probability_matrices) != len(node_embeddings_list):
                raise ValueError(
                    "predicted_edge_probability_matrices must align with generated node batches "
                    f"(got {len(predicted_edge_probability_matrices)} matrices for {len(node_embeddings_list)} graphs)."
                )
            predicted_probs_list = []
            for node_embeddings, prob_matrix in zip(node_embeddings_list, predicted_edge_probability_matrices):
                n_nodes = node_embeddings.shape[0]
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
        for graph_idx, (prob_list, node_embeddings) in enumerate(zip(predicted_probs_list, node_embeddings_list)):
            n_nodes = node_embeddings.shape[0]
            idx = 0
            prob_matrix = np.zeros((n_nodes, n_nodes))
            # Reconstruct the probability matrix from the flat list.
            for i in range(n_nodes):
                for j in range(n_nodes):
                    if i != j:
                        prob_matrix[i, j] = prob_list[idx]
                        idx += 1
            # Zero out probabilities for edges where either node is non-existent.
            existent = np.asarray(existence_masks[graph_idx][:n_nodes], dtype=bool)
            for i in range(n_nodes):
                for j in range(n_nodes):
                    if not (existent[i] and existent[j]):
                        prob_matrix[i, j] = 0
            # Ensure the matrix is symmetric.
            prob_matrix = (prob_matrix + prob_matrix.T) / 2
            target_degrees = self.get_degrees(
                np.asarray(degree_predictions[graph_idx][:n_nodes], dtype=float),
                existent,
            )
            # Optimize the probability matrix into a binary adjacency matrix.
            adj = self.optimize_adjacency_matrix(prob_matrix, target_degrees)
            adj_mtx_list.append(adj)
        return adj_mtx_list

    def decode_node_labels(
        self,
        generated_nodes: GeneratedNodeBatch,
    ) -> List[np.ndarray]:
        """Predict node-level labels for each encoding matrix."""
        channel_plan = self._plan_channel("node_labels")
        if channel_plan is not None and channel_plan.mode == "constant":
            predicted_node_labels_list = []
            for node_embeddings in generated_nodes.node_embeddings_list:
                n = node_embeddings.shape[0]
                predicted_node_labels_list.append(np.array([channel_plan.constant_value] * n, dtype=object))
            return predicted_node_labels_list

        if channel_plan is not None and channel_plan.mode == "disabled":
            predicted_node_labels_list = []
            for node_embeddings in generated_nodes.node_embeddings_list:
                n = node_embeddings.shape[0]
                predicted_node_labels_list.append(np.array([None] * n, dtype=object))
            return predicted_node_labels_list

        raise RuntimeError(
            "decode_node_labels requires generator-provided node labels."
        )

    def decode_edge_labels(
        self,
        generated_nodes: GeneratedNodeBatch,
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

        if channel_plan is not None and channel_plan.mode == "disabled":
            return [np.asarray([], dtype=object) for _ in generated_nodes.node_embeddings_list]

        raise RuntimeError(
            "decode_edge_labels requires generator-provided edge labels."
        )

    @timeit
    def decode(
        self,
        generated_nodes: GeneratedNodeBatch,
        predicted_node_labels_list: Optional[List[np.ndarray]] = None,
        predicted_edge_probability_matrices: Optional[List[np.ndarray]] = None,
        predicted_edge_label_matrices: Optional[List[np.ndarray]] = None,
    ) -> List[nx.Graph]:
        """Reconstruct labelled graphs from node encodings, respecting existence masks."""
        adj_mtx_list = self.decode_adjacency_matrix(
            generated_nodes,
            predicted_edge_probability_matrices=predicted_edge_probability_matrices,
        )
        
        if predicted_node_labels_list is None:
            predicted_node_labels_list = self.decode_node_labels(generated_nodes)
        
        predicted_edge_labels_list = self.decode_edge_labels(
            generated_nodes,
            adj_mtx_list,
            predicted_edge_label_matrices=predicted_edge_label_matrices,
        )
        
        graphs = []
        for node_embeddings, node_presence_mask, node_labels, edge_labels, adj_mtx in zip(
                generated_nodes.node_embeddings_list,
                generated_nodes.node_presence_mask,
                predicted_node_labels_list,
                predicted_edge_labels_list,
                adj_mtx_list):
            graph = nx.from_numpy_array(adj_mtx)
            
            if len(node_labels) > 0 and not all(label is None for label in node_labels):
                node_label_map = {i: label for i, label in enumerate(node_labels)}
                nx.set_node_attributes(graph, node_label_map, 'label')
            
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
            
            existent_indices = np.where(np.asarray(node_presence_mask[:node_embeddings.shape[0]], dtype=bool))[0]
            filtered_graph = graph.subgraph(existent_indices).copy()
            graphs.append(filtered_graph)
        
        return graphs
    
    def save(self, filename: str = 'generative_model.obj') -> None:
        """Serialise the current object to `filename` using pickle."""
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
    
    def load(self, filename: str = 'generative_model.obj') -> 'EqMDecompositionalGraphDecoder':
        """Load a previously saved instance from disk and return it."""
        with open(filename, 'rb') as f:
            self = pickle.load(f)
        return self

# =============================================================================
# EqMDecompositionalGraphGenerator Class 
# =============================================================================

class EqMDecompositionalGraphGenerator(object):
    """End-to-end manager that vectorises graphs, trains generators, and rebuilds structures."""
    def __init__(
            self,
            graph_vectorizer: Any = None,
            node_graph_vectorizer: Any = None,
            conditional_node_generator_model: Optional[ConditionalNodeGeneratorBase] = None,
            graph_decoder: Optional[EqMDecompositionalGraphDecoder] = None,
            verbose: bool = True,
            locality_sample_fraction: float = 1.0,
            locality_horizon: int = 1,
            negative_sample_factor: int = 1,
            locality_sampling_strategy: str = "stratified_preserve",
            locality_target_positive_ratio: Optional[float] = None,
            feasibility_estimator: Any = None,
            use_feasibility_filtering: bool = True,
            max_feasibility_attempts: int = 10,
            feasibility_candidates_per_attempt: int = 4,
            feasibility_failure_mode: str = "return_partial",
            ) -> None:
        """Store the collaborating components and configuration used for the pipeline."""
        self.graph_vectorizer = graph_vectorizer
        self.node_graph_vectorizer = node_graph_vectorizer
        self.conditional_node_generator_model = conditional_node_generator_model
        self.graph_decoder = graph_decoder
        self.verbose = verbose
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
        self.negative_sample_factor = negative_sample_factor
        self.locality_sampling_strategy = locality_sampling_strategy
        self.locality_target_positive_ratio = locality_target_positive_ratio
        self.feasibility_estimator = feasibility_estimator
        self.use_feasibility_filtering = bool(use_feasibility_filtering)
        self.max_feasibility_attempts = int(max_feasibility_attempts)
        self.feasibility_candidates_per_attempt = int(feasibility_candidates_per_attempt)
        self.feasibility_failure_mode = str(feasibility_failure_mode)
        valid_sampling_strategies = {"uniform", "stratified_preserve", "stratified_target"}
        if self.locality_sampling_strategy not in valid_sampling_strategies:
            raise ValueError(
                f"locality_sampling_strategy must be one of {sorted(valid_sampling_strategies)} "
                f"(got {self.locality_sampling_strategy!r})."
            )
        if self.locality_target_positive_ratio is not None and not 0.0 < self.locality_target_positive_ratio < 1.0:
            raise ValueError("locality_target_positive_ratio must be between 0 and 1 when provided.")
        if self.max_feasibility_attempts < 1:
            raise ValueError("max_feasibility_attempts must be >= 1")
        if self.feasibility_candidates_per_attempt < 1:
            raise ValueError("feasibility_candidates_per_attempt must be >= 1")
        valid_feasibility_failure_modes = {"raise", "return_partial"}
        if self.feasibility_failure_mode not in valid_feasibility_failure_modes:
            raise ValueError(
                f"feasibility_failure_mode must be one of {sorted(valid_feasibility_failure_modes)} "
                f"(got {self.feasibility_failure_mode!r})."
            )

    def set_feasibility_filtering(self, enabled: bool) -> None:
        """Enable or disable feasibility filtering during generation without discarding the fitted estimator."""
        self.use_feasibility_filtering = bool(enabled)

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

        direct_edges_enabled = True
        direct_edges_mode = "learned"
        direct_edges_reason = "Generator should learn horizon-1 edge presence for the decoder."

        aux_enabled = bool(self.locality_horizon > 1)
        if aux_enabled:
            auxiliary_mode = "learned"
            auxiliary_reason = f"Use horizon-{self.locality_horizon} locality as auxiliary regularization."
        else:
            auxiliary_mode = "disabled"
            auxiliary_reason = "No auxiliary locality is needed when locality_horizon=1."

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
        if self.conditional_node_generator_model is not None:
            self.conditional_node_generator_model.verbose = self.verbose
        if self.graph_decoder is not None:
            self.graph_decoder.verbose = self.verbose

    @timeit
    def fit(
        self,
        graphs: List[nx.Graph],
        train_node_generator: bool = True,
        targets: Optional[Sequence[Any]] = None,
    ) -> 'EqMDecompositionalGraphGenerator':
        if self.verbose:
            print(f"Fitting model on {len(graphs)} graphs")
        if targets is not None and len(targets) != len(graphs):
            raise ValueError(
                "targets length must match the number of graphs "
                f"(got {len(targets)} targets for {len(graphs)} graphs)."
            )

        # Fit vectorizers
        self.graph_vectorizer.fit(graphs)
        self.node_graph_vectorizer.fit(graphs)
        if self.feasibility_estimator is not None:
            if self.verbose:
                print(f"Fitting feasibility estimator on {len(graphs)} graphs")
            self.feasibility_estimator.fit(graphs)
        node_label_targets = self.graphs_to_node_label_targets(graphs)
        edge_label_targets, edge_label_pairs = self.graphs_to_edge_label_targets(graphs)
        self._fit_node_label_vocab(node_label_targets)
        supervision_plan = self._build_supervision_plan(
            graphs,
            node_label_targets=node_label_targets,
            edge_label_targets=edge_label_targets,
        )
        self.supervision_plan_ = supervision_plan
        if self.conditional_node_generator_model is not None:
            setattr(self.conditional_node_generator_model, "supervision_plan_", supervision_plan)
        if self.graph_decoder is not None:
            setattr(self.graph_decoder, "supervision_plan_", supervision_plan)

        node_embeddings_list, graph_conditioning = self.encode(graphs)

        if train_node_generator:
            edge_pairs_for_cond_gen = None
            edge_targets_for_cond_gen = None
            auxiliary_edge_pairs_for_cond_gen = None
            auxiliary_edge_targets_for_cond_gen = None
            if supervision_plan.direct_edges.enabled:
                if self.graph_decoder is None:
                    raise RuntimeError("Locality supervision requested but graph_decoder is None.")
                self._log_supervision_plan(supervision_plan)

                edge_targets_for_cond_gen, edge_pairs_for_cond_gen = self.graph_decoder.compute_edge_supervision(
                    graphs,
                    node_embeddings_list,
                    locality_sample_fraction=self.locality_sample_fraction,
                    negative_sample_factor=self.negative_sample_factor,
                    locality_sampling_strategy=self.locality_sampling_strategy,
                    locality_target_positive_ratio=self.locality_target_positive_ratio,
                    horizon=1,
                    supervision_name="direct_edge",
                )
                if supervision_plan.auxiliary_locality.enabled:
                    auxiliary_edge_targets_for_cond_gen, auxiliary_edge_pairs_for_cond_gen = (
                        self.graph_decoder.compute_edge_supervision(
                            graphs,
                            node_embeddings_list,
                            locality_sample_fraction=self.locality_sample_fraction,
                            negative_sample_factor=self.negative_sample_factor,
                            locality_sampling_strategy=self.locality_sampling_strategy,
                            locality_target_positive_ratio=self.locality_target_positive_ratio,
                            horizon=supervision_plan.auxiliary_locality.horizon,
                            supervision_name="aux_locality",
                        )
                    )
            else:
                self._log_supervision_plan(supervision_plan)

            node_batch = self._build_node_batch(
                graphs,
                node_embeddings_list,
                node_label_targets=node_label_targets if supervision_plan.node_labels.enabled else None,
                edge_pairs=edge_pairs_for_cond_gen,
                edge_targets=edge_targets_for_cond_gen,
                edge_label_pairs=edge_label_pairs if supervision_plan.edge_labels.enabled else None,
                edge_label_targets=edge_label_targets if supervision_plan.edge_labels.enabled else None,
                auxiliary_edge_pairs=auxiliary_edge_pairs_for_cond_gen,
                auxiliary_edge_targets=auxiliary_edge_targets_for_cond_gen,
            )
            if self.verbose:
                print(
                    f"Training conditional model on {len(node_batch)} graphs "
                    f"with up to {node_batch.node_presence_mask.shape[1]} nodes each."
                )
            if node_batch.edge_pairs is not None and node_batch.edge_targets is not None and self.verbose:
                print(f"Using direct-edge supervision with {len(node_batch.edge_pairs)} labelled pairs.")
            self.conditional_node_generator_model.setup(
                node_batch=node_batch,
                graph_conditioning=graph_conditioning,
                targets=targets,
            )
            self.conditional_node_generator_model.fit(
                node_batch=node_batch,
                graph_conditioning=graph_conditioning,
                targets=targets,
            )

        return self

    @timeit
    def node_encode(self, graphs: List[nx.Graph]) -> List[np.ndarray]:
        """Transform graphs into per-node embedding matrices."""
        if int(self.verbose) >= 3:
            print(f"Node encoding {len(graphs)} graphs")
        return self.node_graph_vectorizer.transform(graphs)

    @timeit
    def graph_encode(self, graphs: List[nx.Graph]) -> GraphConditioningBatch:
        """Transform graphs into explicit graph-level conditioning signals."""
        if int(self.verbose) >= 3:
            print(f"Encoding {len(graphs)} graphs")
        graph_embeddings = np.asarray(self.graph_vectorizer.transform(graphs))
        node_counts = np.asarray([graph.number_of_nodes() for graph in graphs], dtype=np.int64)
        edge_counts = np.asarray([graph.number_of_edges() for graph in graphs], dtype=np.int64)
        return GraphConditioningBatch(
            graph_embeddings=graph_embeddings,
            node_counts=node_counts,
            edge_counts=edge_counts,
            node_label_histograms=None,
        )

    def encode(self, graphs: List[nx.Graph]) -> Tuple[List[np.ndarray], GraphConditioningBatch]:
        """Produce both node-level embeddings and explicit graph-level conditioning."""
        return self.node_encode(graphs), self.graph_encode(graphs)

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
        return False

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

    def _build_node_batch(
        self,
        graphs: List[nx.Graph],
        node_embeddings_list: List[np.ndarray],
        node_label_targets: Optional[List[np.ndarray]] = None,
        edge_pairs: Optional[List[Tuple[int, int, int]]] = None,
        edge_targets: Optional[np.ndarray] = None,
        edge_label_pairs: Optional[List[Tuple[int, int, int]]] = None,
        edge_label_targets: Optional[np.ndarray] = None,
        auxiliary_edge_pairs: Optional[List[Tuple[int, int, int]]] = None,
        auxiliary_edge_targets: Optional[np.ndarray] = None,
    ) -> NodeGenerationBatch:
        """Assemble explicit node-level supervision tensors from graphs."""
        max_num_rows = max(emb.shape[0] for emb in node_embeddings_list)
        node_presence_mask = np.zeros((len(graphs), max_num_rows), dtype=bool)
        node_degree_targets = np.zeros((len(graphs), max_num_rows), dtype=np.int64)
        for graph_idx, graph in enumerate(graphs):
            nodes = list(graph.nodes())
            node_presence_mask[graph_idx, :len(nodes)] = True
            node_degree_targets[graph_idx, :len(nodes)] = np.asarray(
                [graph.degree(node) for node in nodes],
                dtype=np.int64,
            )
        return NodeGenerationBatch(
            node_embeddings_list=node_embeddings_list,
            node_presence_mask=node_presence_mask,
            node_degree_targets=node_degree_targets,
            node_label_targets=node_label_targets,
            edge_pairs=edge_pairs,
            edge_targets=edge_targets,
            edge_label_pairs=edge_label_pairs,
            edge_label_targets=edge_label_targets,
            auxiliary_edge_pairs=auxiliary_edge_pairs,
            auxiliary_edge_targets=auxiliary_edge_targets,
        )

    def _log_generated_batch_info(
        self,
        graph_conditioning: GraphConditioningBatch,
        generated_nodes: GeneratedNodeBatch,
    ) -> None:
        """Print per-graph generation summaries at the highest verbosity level."""
        if int(self.verbose) < 3:
            return
        total_graphs = len(generated_nodes.node_embeddings_list)
        for graph_idx in range(total_graphs):
            node_embeddings = generated_nodes.node_embeddings_list[graph_idx]
            predicted_node_count = (
                int(np.sum(generated_nodes.node_presence_mask[graph_idx][: node_embeddings.shape[0]]))
                if generated_nodes.node_presence_mask is not None
                else node_embeddings.shape[0]
            )
            conditioning_node_count = int(graph_conditioning.node_counts[graph_idx])
            conditioning_edge_count = int(graph_conditioning.edge_counts[graph_idx])
            message = (
                f"Generated graph {graph_idx + 1}/{total_graphs}: "
                f"conditioning_nodes={conditioning_node_count}, "
                f"conditioning_edges={conditioning_edge_count}, "
                f"predicted_nodes={predicted_node_count}"
            )
            if generated_nodes.node_degree_predictions is not None:
                valid_deg = np.asarray(
                    generated_nodes.node_degree_predictions[graph_idx][: node_embeddings.shape[0]],
                    dtype=float,
                )
                if generated_nodes.node_presence_mask is not None:
                    valid_mask = np.asarray(
                        generated_nodes.node_presence_mask[graph_idx][: node_embeddings.shape[0]],
                        dtype=bool,
                    )
                    valid_deg = valid_deg[valid_mask]
                if valid_deg.size > 0:
                    message += (
                        f", mean_degree={float(np.mean(valid_deg)):.2f}, "
                        f"max_degree={int(np.max(valid_deg))}"
                    )
            if generated_nodes.edge_probability_matrices is not None:
                edge_probs = np.asarray(generated_nodes.edge_probability_matrices[graph_idx], dtype=float)
                off_diag = edge_probs[~np.eye(edge_probs.shape[0], dtype=bool)]
                if off_diag.size > 0:
                    message += (
                        f", mean_edge_prob={float(np.mean(off_diag)):.3f}, "
                        f"max_edge_prob={float(np.max(off_diag)):.3f}"
                    )
            print(message)
            if generated_nodes.node_labels is not None:
                labels = np.asarray(generated_nodes.node_labels[graph_idx], dtype=object)
                if generated_nodes.node_presence_mask is not None:
                    valid_mask = np.asarray(
                        generated_nodes.node_presence_mask[graph_idx][: labels.shape[0]],
                        dtype=bool,
                    )
                    labels = labels[valid_mask]
                if labels.size > 0:
                    unique_labels, counts = np.unique(labels, return_counts=True)
                    label_summary = {label: int(count) for label, count in zip(unique_labels.tolist(), counts.tolist())}
                    print(f"  node_labels={label_summary}")

    @staticmethod
    def _slice_graph_conditioning(
        graph_conditioning: GraphConditioningBatch,
        indices: Sequence[int],
    ) -> GraphConditioningBatch:
        """Select a subset of conditioning rows by integer indices."""
        idx = np.asarray(indices, dtype=np.int64)
        node_label_histograms = None
        if graph_conditioning.node_label_histograms is not None:
            node_label_histograms = np.asarray(graph_conditioning.node_label_histograms)[idx]
        return GraphConditioningBatch(
            graph_embeddings=np.asarray(graph_conditioning.graph_embeddings)[idx],
            node_counts=np.asarray(graph_conditioning.node_counts)[idx],
            edge_counts=np.asarray(graph_conditioning.edge_counts)[idx],
            node_label_histograms=node_label_histograms,
        )

    @staticmethod
    def _repeat_graph_conditioning(
        graph_conditioning: GraphConditioningBatch,
        repeats: int,
    ) -> GraphConditioningBatch:
        """Repeat each conditioning row a fixed number of times."""
        if repeats < 1:
            raise ValueError("repeats must be >= 1")
        node_label_histograms = None
        if graph_conditioning.node_label_histograms is not None:
            node_label_histograms = np.repeat(
                np.asarray(graph_conditioning.node_label_histograms),
                repeats,
                axis=0,
            )
        return GraphConditioningBatch(
            graph_embeddings=np.repeat(
                np.asarray(graph_conditioning.graph_embeddings),
                repeats,
                axis=0,
            ),
            node_counts=np.repeat(np.asarray(graph_conditioning.node_counts), repeats, axis=0),
            edge_counts=np.repeat(np.asarray(graph_conditioning.edge_counts), repeats, axis=0),
            node_label_histograms=node_label_histograms,
        )

    def _decode_conditioning_batch(
        self,
        graph_conditioning: GraphConditioningBatch,
        desired_target: Optional[Union[int, float, Sequence[Any]]] = None,
        guidance_scale: float = 1.0,
        desired_class: Optional[Union[int, Sequence[int]]] = None,
    ) -> List[nx.Graph]:
        """Run a single generator pass and decode graphs without feasibility retries."""
        if int(self.verbose) >= 3:
            print(f"Predicting node matrices for {len(graph_conditioning)} graphs...")
        generated_nodes = self.conditional_node_generator_model.predict(
            graph_conditioning,
            desired_target=desired_target,
            guidance_scale=guidance_scale,
            desired_class=desired_class,
        )
        self._log_generated_batch_info(graph_conditioning, generated_nodes)
        return self.graph_decoder.decode(
            generated_nodes,
            predicted_node_labels_list=generated_nodes.node_labels,
            predicted_edge_probability_matrices=generated_nodes.edge_probability_matrices,
            predicted_edge_label_matrices=generated_nodes.edge_label_matrices,
        )

    def _decode_with_feasibility_slots(
        self,
        graph_conditioning: GraphConditioningBatch,
        desired_target: Optional[Union[int, float, Sequence[Any]]] = None,
        guidance_scale: float = 1.0,
        desired_class: Optional[Union[int, Sequence[int]]] = None,
        apply_feasibility_filtering: Optional[bool] = None,
    ) -> List[Optional[nx.Graph]]:
        """Decode graphs and optionally reject infeasible outputs until the batch is filled."""
        use_filtering = (
            self.use_feasibility_filtering
            if apply_feasibility_filtering is None
            else bool(apply_feasibility_filtering)
        )
        if self.feasibility_estimator is None or not use_filtering:
            return list(
                self._decode_conditioning_batch(
                    graph_conditioning,
                    desired_target=desired_target,
                    guidance_scale=guidance_scale,
                    desired_class=desired_class,
                )
            )

        accepted_graphs_by_slot: List[Optional[nx.Graph]] = [None] * len(graph_conditioning)
        pending_conditioning = graph_conditioning
        pending_slot_indices = list(range(len(graph_conditioning)))
        attempt = 0
        total_generated = 0
        while len(pending_conditioning) > 0 and attempt < self.max_feasibility_attempts:
            attempt += 1
            candidate_conditioning = self._repeat_graph_conditioning(
                pending_conditioning,
                repeats=self.feasibility_candidates_per_attempt,
            )
            candidate_slot_indices = [
                slot_idx
                for slot_idx in pending_slot_indices
                for _ in range(self.feasibility_candidates_per_attempt)
            ]
            decoded_graphs = self._decode_conditioning_batch(
                candidate_conditioning,
                desired_target=desired_target,
                guidance_scale=guidance_scale,
                desired_class=desired_class,
            )
            total_generated += len(decoded_graphs)
            feasibility_mask = np.asarray(
                self.feasibility_estimator.predict(decoded_graphs),
                dtype=bool,
            )
            if feasibility_mask.shape[0] != len(decoded_graphs):
                raise RuntimeError(
                    "Feasibility estimator returned a mask of unexpected length "
                    f"({feasibility_mask.shape[0]} for {len(decoded_graphs)} graphs)."
                )
            accepted_this_round = set()
            for local_idx, (graph, is_feasible) in enumerate(zip(decoded_graphs, feasibility_mask.tolist())):
                slot_idx = candidate_slot_indices[local_idx]
                if is_feasible and accepted_graphs_by_slot[slot_idx] is None:
                    accepted_graphs_by_slot[slot_idx] = graph
                    accepted_this_round.add(slot_idx)
            rejected_slot_indices = [
                slot_idx for slot_idx in pending_slot_indices if accepted_graphs_by_slot[slot_idx] is None
            ]
            if int(self.verbose) >= 1:
                accepted_now = len(accepted_this_round)
                rejected_now = len(rejected_slot_indices)
                accepted_total = sum(graph is not None for graph in accepted_graphs_by_slot)
                missing_total = len(graph_conditioning) - accepted_total
                attempted_total = len(decoded_graphs)
                acceptance_rate = (accepted_now / attempted_total) if attempted_total > 0 else 0.0
                print(
                    f"Feasibility attempt {attempt:>2}/{self.max_feasibility_attempts:<2} | "
                    f"generated={attempted_total:>4} | "
                    f"accepted={accepted_now:>2} | "
                    f"rejected={rejected_now:>2} | "
                    f"rate={acceptance_rate:>6.1%} | "
                    f"accepted_total={accepted_total:>2} | "
                    f"missing_total={missing_total:>2}"
                )
            if not rejected_slot_indices:
                break
            pending_slot_indices = rejected_slot_indices
            pending_conditioning = self._slice_graph_conditioning(
                graph_conditioning,
                pending_slot_indices,
            )

        accepted_count = sum(graph is not None for graph in accepted_graphs_by_slot)
        if int(self.verbose) >= 1:
            overall_rate = (accepted_count / total_generated) if total_generated > 0 else 0.0
            print(
                "Feasibility filtering summary: "
                f"generated={total_generated}, accepted={accepted_count}, "
                f"acceptance_rate={overall_rate:.1%}."
            )
        return accepted_graphs_by_slot

    def _decode_with_feasibility(
        self,
        graph_conditioning: GraphConditioningBatch,
        desired_target: Optional[Union[int, float, Sequence[Any]]] = None,
        guidance_scale: float = 1.0,
        desired_class: Optional[Union[int, Sequence[int]]] = None,
        apply_feasibility_filtering: Optional[bool] = None,
    ) -> List[nx.Graph]:
        """Decode graphs and optionally reject infeasible outputs until the batch is filled."""
        accepted_graphs_by_slot = self._decode_with_feasibility_slots(
            graph_conditioning,
            desired_target=desired_target,
            guidance_scale=guidance_scale,
            desired_class=desired_class,
            apply_feasibility_filtering=apply_feasibility_filtering,
        )
        accepted_count = sum(graph is not None for graph in accepted_graphs_by_slot)
        if accepted_count != len(graph_conditioning):
            if self.feasibility_failure_mode == "raise":
                raise RuntimeError(
                    "Feasibility filtering did not recover enough graphs: "
                    f"accepted {accepted_count} of {len(graph_conditioning)} after "
                    f"{self.max_feasibility_attempts} attempts."
                )
            if int(self.verbose) >= 1:
                print(
                    "Feasibility filtering exhausted retries; returning only feasible graphs: "
                    f"accepted {accepted_count} of {len(graph_conditioning)}."
                )
        return [graph for graph in accepted_graphs_by_slot if graph is not None]

    def decode(
        self,
        graph_conditioning: GraphConditioningBatch,
        desired_target: Optional[Union[int, float, Sequence[Any]]] = None,
        guidance_scale: float = 1.0,
        desired_class: Optional[Union[int, Sequence[int]]] = None,
        apply_feasibility_filtering: Optional[bool] = None,
    ) -> List[nx.Graph]:
        """Decode conditioning vectors into reconstructed graphs."""
        if self.verbose:
            print(f"Decoding {len(graph_conditioning)} conditioning vectors")
            if desired_target is not None:
                print(f"Using CFG target guidance: {desired_target} (scale={guidance_scale})")
            if desired_class is not None:
                print(f"Using classifier guidance toward class(es): {desired_class}")
        
        return self._decode_with_feasibility(
            graph_conditioning,
            desired_target=desired_target,
            guidance_scale=guidance_scale,
            desired_class=desired_class,
            apply_feasibility_filtering=apply_feasibility_filtering,
        )

    @timeit
    def sample(
        self,
        n_samples: int = 1,
        desired_target: Optional[Union[int, float, Sequence[Any]]] = None,
        guidance_scale: float = 1.0,
        desired_class: Optional[Union[int, Sequence[int]]] = None,
        apply_feasibility_filtering: Optional[bool] = None,
    ) -> List[nx.Graph]:
        """Generate random graphs by sampling conditioning vectors from the prior."""
        if self.verbose:
            print(f"Sampling {n_samples} graphs")
            if desired_target is not None:
                print(f"Using CFG target guidance: {desired_target} (scale={guidance_scale})")
            if desired_class is not None:
                print(f"Using classifier guidance toward class(es): {desired_class}")
        
        sampled_conditioning = self._sample_conditions(n_samples)
        return self._decode_with_feasibility(
            sampled_conditioning,
            desired_target=desired_target,
            guidance_scale=guidance_scale,
            desired_class=desired_class,
            apply_feasibility_filtering=apply_feasibility_filtering,
        )

    @timeit
    def conditional_sample(
        self,
        graphs: List[nx.Graph],
        n_samples: int = 1,
        desired_target: Optional[Union[int, float, Sequence[Any]]] = None,
        guidance_scale: float = 1.0,
        desired_class: Optional[Union[int, Sequence[int]]] = None,
        apply_feasibility_filtering: Optional[bool] = None,
    ) -> List[List[nx.Graph]]:
        """Sample multiple graphs per input by conditioning on each graph's encoding."""
        _, graph_conditioning = self.encode(graphs)
        repeated_conditioning = self._repeat_graph_conditioning(
            graph_conditioning,
            repeats=n_samples,
        )
        decoded_slots = self._decode_with_feasibility_slots(
            repeated_conditioning,
            desired_target=desired_target,
            guidance_scale=guidance_scale,
            desired_class=desired_class,
            apply_feasibility_filtering=apply_feasibility_filtering,
        )
        return [
            [
                graph
                for graph in decoded_slots[i * n_samples:(i + 1) * n_samples]
                if graph is not None
            ]
            for i in range(len(graphs))
        ]

    def sample_conditioned_on_random(
        self,
        graphs,
        n_samples=1,
        desired_target: Optional[Union[int, float, Sequence[Any]]] = None,
        guidance_scale: float = 1.0,
        apply_feasibility_filtering: Optional[bool] = None,
    ):
        sampled_seed_graphs = random.choices(graphs, k=n_samples)
        reconstructed_graphs_list = self.conditional_sample(
            sampled_seed_graphs,
            n_samples=1,
            desired_target=desired_target,
            guidance_scale=guidance_scale,
            apply_feasibility_filtering=apply_feasibility_filtering,
        )
        sampled_graphs = [reconstructed_graphs[0] for reconstructed_graphs in reconstructed_graphs_list if reconstructed_graphs]
        return sampled_graphs

    def interpolate(
        self,
        G1: nx.Graph,
        G2: nx.Graph,
        k: int = 7,
        apply_feasibility_filtering: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Interpolate between two graph condition vectors and decode intermediate graphs."""

        def _interpolate_integer_series(start, end, ts, minimum):
            values = np.rint([(1.0 - t) * start + t * end for t in ts]).astype(np.int64)
            return np.maximum(values, np.int64(minimum))

        cond_a = self.graph_encode([G1])
        cond_b = self.graph_encode([G2])
        ts = np.linspace(0.0, 1.0, k + 2)[1:-1]

        interpolated_graph_embeddings = np.stack(
            [(1.0 - t) * cond_a.graph_embeddings[0] + t * cond_b.graph_embeddings[0] for t in ts],
            axis=0,
        )
        interpolated_node_counts = _interpolate_integer_series(
            cond_a.node_counts[0],
            cond_b.node_counts[0],
            ts,
            minimum=1,
        )
        interpolated_edge_counts = _interpolate_integer_series(
            cond_a.edge_counts[0],
            cond_b.edge_counts[0],
            ts,
            minimum=0,
        )

        interpolated_histograms = None
        if cond_a.node_label_histograms is not None and cond_b.node_label_histograms is not None:
            interpolated_histograms = np.stack(
                [
                    (1.0 - t) * cond_a.node_label_histograms[0] + t * cond_b.node_label_histograms[0]
                    for t in ts
                ],
                axis=0,
            )

        interpolated_conditioning = GraphConditioningBatch(
            graph_embeddings=interpolated_graph_embeddings,
            node_counts=interpolated_node_counts,
            edge_counts=interpolated_edge_counts,
            node_label_histograms=interpolated_histograms,
        )
        decoded_slots = self._decode_with_feasibility_slots(
            interpolated_conditioning,
            apply_feasibility_filtering=apply_feasibility_filtering,
        )
        step_summary = pd.DataFrame(
            {
                "step": np.arange(1, len(ts) + 1),
                "t": np.round(ts, 3),
                "target_nodes": interpolated_node_counts,
                "target_edges": interpolated_edge_counts,
                "decoded": [graph is not None for graph in decoded_slots],
            }
        )
        return {
            "ts": ts,
            "conditioning": interpolated_conditioning,
            "decoded_slots": decoded_slots,
            "generated_graphs": [graph for graph in decoded_slots if graph is not None],
            "summary": step_summary,
        }

    def mean(
        self,
        graphs: List[nx.Graph]
    ) -> nx.Graph:
        """Compute a geometric mean graph via the SLERP barycentre of encodings."""
        graph_conditioning = self.graph_encode(graphs)
        Y = np.vstack(graph_conditioning.graph_embeddings)
        centroid = e(Y)
        mean_node_count = int(round(np.mean(graph_conditioning.node_counts)))
        mean_edge_count = int(round(np.mean(graph_conditioning.edge_counts)))
        mean_hist = None
        if graph_conditioning.node_label_histograms is not None:
            mean_hist = np.asarray([np.mean(graph_conditioning.node_label_histograms, axis=0)])
        return self.decode(
            GraphConditioningBatch(
                graph_embeddings=np.asarray([centroid]),
                node_counts=np.asarray([mean_node_count], dtype=np.int64),
                edge_counts=np.asarray([mean_edge_count], dtype=np.int64),
                node_label_histograms=mean_hist,
            )
        )[0]
    
    @timeit
    def fit_classifier(self, graphs, targets, epochs=20, lr=1e-3):
        del graphs, targets, epochs, lr
        raise NotImplementedError(
            "fit_classifier() is deprecated in strict CFG mode. "
            "Pass targets directly to fit(..., targets=...) and use desired_target/guidance_scale at decode/sample time."
        )
