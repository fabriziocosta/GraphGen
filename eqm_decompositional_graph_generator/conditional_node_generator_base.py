from dataclasses import dataclass
from typing import List, Any, Optional, Tuple, Sequence, Union
import numpy as np

@dataclass
class GraphConditioningBatch:
    """Explicit graph-level conditioning signals passed into the generator."""

    graph_embeddings: np.ndarray
    node_counts: np.ndarray
    edge_counts: np.ndarray
    node_label_histograms: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return int(len(self.graph_embeddings))


@dataclass
class NodeGenerationBatch:
    """Explicit node-level training inputs and semantic supervision signals."""

    node_embeddings_list: List[np.ndarray]
    node_presence_mask: np.ndarray
    node_degree_targets: np.ndarray
    node_label_targets: Optional[List[np.ndarray]] = None
    edge_pairs: Optional[List[Tuple[int, int, int]]] = None
    edge_targets: Optional[np.ndarray] = None
    edge_label_pairs: Optional[List[Tuple[int, int, int]]] = None
    edge_label_targets: Optional[np.ndarray] = None
    auxiliary_edge_pairs: Optional[List[Tuple[int, int, int]]] = None
    auxiliary_edge_targets: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return int(len(self.node_embeddings_list))


@dataclass
class GeneratedNodeBatch:
    """Generator outputs separated into embeddings and explicit semantic predictions."""

    node_embeddings_list: List[np.ndarray]
    node_presence_mask: Optional[np.ndarray] = None
    node_degree_predictions: Optional[np.ndarray] = None
    node_labels: Optional[List[np.ndarray]] = None
    edge_probability_matrices: Optional[List[np.ndarray]] = None
    edge_label_matrices: Optional[List[np.ndarray]] = None

    def __len__(self) -> int:
        return int(len(self.node_embeddings_list))


class ConditionalNodeGeneratorBase:
    """
    Abstract base class for conditional node generators.

    All conditional node generators should inherit from this class and implement
    the fit and predict methods.

    Methods
    -------
    fit(
        node_batch: NodeGenerationBatch,
        graph_conditioning: GraphConditioningBatch,
    )
        Fit the model to training data.

    predict(graph_conditioning)
        Generate samples conditioned on explicit graph-level signals.
    """

    def fit(
        self,
        node_batch: NodeGenerationBatch,
        graph_conditioning: GraphConditioningBatch,
        targets: Optional[Sequence[Any]] = None,
    ):
        raise NotImplementedError("fit() must be implemented by subclasses.")

    def predict(
        self,
        graph_conditioning: GraphConditioningBatch,
        desired_target: Optional[Union[int, float, Sequence[Any]]] = None,
        guidance_scale: float = 1.0,
        desired_class: Optional[Union[int, Sequence[int]]] = None,
    ):
        raise NotImplementedError("predict() must be implemented by subclasses.")
