import contextlib
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split

from .generator_shared import (
    CrossTransformerEncoderLayer,
    EdgeMLP,
    MetricsLogger,
    plot_metrics,
    suppress_output,
)
from .lightning_utils import run_trainer_fit
from .conditional_node_generator_base import ConditionalNodeGeneratorBase


class EqMGraphDataset(Dataset):
    """Dataset carrying graph node features, conditioning, masks, and label targets."""

    def __init__(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        node_mask: np.ndarray,
        node_label_targets: Optional[np.ndarray] = None,
    ):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
        self.node_mask = torch.tensor(node_mask, dtype=torch.bool)
        self.node_label_targets = None
        if node_label_targets is not None:
            self.node_label_targets = torch.tensor(node_label_targets, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        if self.node_label_targets is None:
            return self.X[idx], self.Y[idx], self.node_mask[idx]
        return self.X[idx], self.Y[idx], self.node_mask[idx], self.node_label_targets[idx]


class EqMGraphWithEdgesDataset(Dataset):
    """Dataset carrying edge supervision, auxiliary locality, and optional node labels."""

    def __init__(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        edge_pairs: List[Tuple[int, int, int]],
        edge_targets: np.ndarray,
        edge_label_pairs: Optional[List[Tuple[int, int, int]]],
        edge_label_targets: Optional[np.ndarray],
        auxiliary_edge_pairs: Optional[List[Tuple[int, int, int]]],
        auxiliary_edge_targets: Optional[np.ndarray],
        node_mask: np.ndarray,
        node_label_targets: Optional[np.ndarray] = None,
    ):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
        self.node_mask = torch.tensor(node_mask, dtype=torch.bool)
        self.node_label_targets = None
        if node_label_targets is not None:
            self.node_label_targets = torch.tensor(node_label_targets, dtype=torch.long)
        self.edge_idx_by_graph: Dict[int, List[Tuple[int, int]]] = {b: [] for b in range(len(X))}
        self.edge_lbl_by_graph: Dict[int, List[float]] = {b: [] for b in range(len(X))}
        if edge_pairs is not None and edge_targets is not None:
            for (b, i, j), lbl in zip(edge_pairs, edge_targets):
                self.edge_idx_by_graph[b].append((i, j))
                self.edge_lbl_by_graph[b].append(lbl)
        self.aux_edge_idx_by_graph: Dict[int, List[Tuple[int, int]]] = {b: [] for b in range(len(X))}
        self.aux_edge_lbl_by_graph: Dict[int, List[float]] = {b: [] for b in range(len(X))}
        if auxiliary_edge_pairs is not None and auxiliary_edge_targets is not None:
            for (b, i, j), lbl in zip(auxiliary_edge_pairs, auxiliary_edge_targets):
                self.aux_edge_idx_by_graph[b].append((i, j))
                self.aux_edge_lbl_by_graph[b].append(lbl)
        self.edge_label_idx_by_graph: Dict[int, List[Tuple[int, int]]] = {b: [] for b in range(len(X))}
        self.edge_label_tgt_by_graph: Dict[int, List[int]] = {b: [] for b in range(len(X))}
        if edge_label_pairs is not None and edge_label_targets is not None:
            for (b, i, j), lbl in zip(edge_label_pairs, edge_label_targets):
                self.edge_label_idx_by_graph[b].append((i, j))
                self.edge_label_tgt_by_graph[b].append(lbl)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        edge_idxs = (
            torch.tensor(self.edge_idx_by_graph[idx], dtype=torch.long)
            if self.edge_idx_by_graph[idx]
            else torch.empty((0, 2), dtype=torch.long)
        )
        edge_lbls = (
            torch.tensor(self.edge_lbl_by_graph[idx], dtype=torch.float32)
            if self.edge_lbl_by_graph[idx]
            else torch.empty((0,), dtype=torch.float32)
        )
        aux_edge_idxs = (
            torch.tensor(self.aux_edge_idx_by_graph[idx], dtype=torch.long)
            if self.aux_edge_idx_by_graph[idx]
            else torch.empty((0, 2), dtype=torch.long)
        )
        aux_edge_lbls = (
            torch.tensor(self.aux_edge_lbl_by_graph[idx], dtype=torch.float32)
            if self.aux_edge_lbl_by_graph[idx]
            else torch.empty((0,), dtype=torch.float32)
        )
        edge_label_idxs = (
            torch.tensor(self.edge_label_idx_by_graph[idx], dtype=torch.long)
            if self.edge_label_idx_by_graph[idx]
            else torch.empty((0, 2), dtype=torch.long)
        )
        edge_label_tgts = (
            torch.tensor(self.edge_label_tgt_by_graph[idx], dtype=torch.long)
            if self.edge_label_tgt_by_graph[idx]
            else torch.empty((0,), dtype=torch.long)
        )
        if self.node_label_targets is None:
            return self.X[idx], self.Y[idx], edge_idxs, edge_lbls, edge_label_idxs, edge_label_tgts, aux_edge_idxs, aux_edge_lbls, self.node_mask[idx]
        return self.X[idx], self.Y[idx], edge_idxs, edge_lbls, edge_label_idxs, edge_label_tgts, aux_edge_idxs, aux_edge_lbls, self.node_mask[idx], self.node_label_targets[idx]


def collate_eqm_graph_with_edges(batch):
    """Batch EqMGraphWithEdgesDataset items into tensors with optional label targets."""
    xs, ys, masks = [], [], []
    local_edge_idxs, local_edge_lbls = [], []
    edge_label_idxs_list, edge_label_targets_list = [], []
    aux_local_edge_idxs, aux_local_edge_lbls = [], []
    label_targets = []
    has_labels = len(batch[0]) == 10
    for sample in batch:
        if has_labels:
            x, y, ei, el, edge_label_ei, edge_label_el, aux_ei, aux_el, mask, node_labels = sample
            label_targets.append(node_labels)
        else:
            x, y, ei, el, edge_label_ei, edge_label_el, aux_ei, aux_el, mask = sample
        xs.append(x)
        ys.append(y)
        masks.append(mask)
        local_edge_idxs.append(ei)
        local_edge_lbls.append(el)
        edge_label_idxs_list.append(edge_label_ei)
        edge_label_targets_list.append(edge_label_el)
        aux_local_edge_idxs.append(aux_ei)
        aux_local_edge_lbls.append(aux_el)
    X = torch.stack(xs)
    Y = torch.stack(ys)
    M = torch.stack(masks)
    def _pack_pairs(per_graph_idxs, per_graph_lbls):
        all_edge_idxs = []
        all_edge_lbls = []
        for b, (ei, el) in enumerate(zip(per_graph_idxs, per_graph_lbls)):
            if ei.numel() == 0:
                continue
            b_col = torch.full((ei.size(0), 1), b, dtype=torch.long)
            all_edge_idxs.append(torch.cat([b_col, ei], dim=1))
            all_edge_lbls.append(el)
        if all_edge_idxs:
            return torch.cat(all_edge_idxs, dim=0), torch.cat(all_edge_lbls, dim=0)
        return torch.empty((0, 3), dtype=torch.long), torch.empty((0,), dtype=torch.float32)

    edge_idx, edge_lbl = _pack_pairs(local_edge_idxs, local_edge_lbls)
    edge_label_idx, edge_label_tgt = _pack_pairs(edge_label_idxs_list, edge_label_targets_list)
    aux_edge_idx, aux_edge_lbl = _pack_pairs(aux_local_edge_idxs, aux_local_edge_lbls)
    if has_labels:
        label_tensor = torch.stack(label_targets)
        return X, Y, edge_idx, edge_lbl, edge_label_idx, edge_label_tgt, aux_edge_idx, aux_edge_lbl, M, label_tensor
    return X, Y, edge_idx, edge_lbl, edge_label_idx, edge_label_tgt, aux_edge_idx, aux_edge_lbl, M


class EqMConditionalNodeGeneratorModule(pl.LightningModule):
    """Conditional EqM model with an explicit scalar energy and score via autograd."""

    def __init__(
        self,
        number_of_rows_per_example: int,
        input_feature_dimension: int,
        condition_feature_dimension: int,
        latent_embedding_dimension: int,
        number_of_transformer_layers: int,
        transformer_attention_head_count: int,
        transformer_dropout: float = 0.1,
        learning_rate: float = 1e-3,
        verbose: bool = False,
        important_feature_index: int = 1,
        max_degree: Optional[int] = None,
        lambda_degree_importance: float = 1.0,
        degree_temperature: Optional[float] = None,
        degree_min_val: float = 0.0,
        degree_range_val: float = 1.0,
        lambda_node_exist_importance: float = 1.0,
        lambda_node_label_importance: float = 1.0,
        lambda_edge_label_importance: float = 1.0,
        use_locality_supervision: bool = False,
        lambda_locality_importance: float = 1.0,
        use_auxiliary_locality_supervision: bool = False,
        lambda_auxiliary_locality_importance: float = 1.0,
        exist_pos_weight: Union[torch.Tensor, float] = 1.0,
        num_node_label_classes: int = 0,
        use_node_label_head: bool = False,
        num_edge_label_classes: int = 0,
        use_edge_label_head: bool = False,
        eqm_sigma: float = 0.2,
        sampling_step_size: float = 0.05,
        sampling_steps: int = 100,
        langevin_noise_scale: float = 0.0,
        pool_condition_tokens: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["verbose"])

        if max_degree is None:
            raise ValueError("max_degree must be provided when initializing the EqM model.")
        if eqm_sigma <= 0:
            raise ValueError(f"eqm_sigma must be positive (got {eqm_sigma}).")
        if sampling_step_size <= 0:
            raise ValueError(f"sampling_step_size must be positive (got {sampling_step_size}).")
        if sampling_steps <= 0:
            raise ValueError(f"sampling_steps must be positive (got {sampling_steps}).")

        self.number_of_rows_per_example = number_of_rows_per_example
        self.input_feature_dimension = input_feature_dimension
        self.condition_feature_dimension = condition_feature_dimension
        self.latent_embedding_dimension = latent_embedding_dimension
        self.number_of_transformer_layers = number_of_transformer_layers
        self.transformer_attention_head_count = transformer_attention_head_count
        self.transformer_dropout = transformer_dropout
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.important_feature_index = important_feature_index
        self.max_degree = int(max_degree)
        self.lambda_degree_importance = lambda_degree_importance
        self.degree_temperature = degree_temperature
        self.lambda_node_exist_importance = lambda_node_exist_importance
        self.lambda_node_label_importance = lambda_node_label_importance
        self.lambda_edge_label_importance = lambda_edge_label_importance
        self.use_locality_supervision = bool(use_locality_supervision)
        self.lambda_locality_importance = lambda_locality_importance
        self.use_auxiliary_locality_supervision = bool(use_auxiliary_locality_supervision)
        self.lambda_auxiliary_locality_importance = lambda_auxiliary_locality_importance
        self.num_node_label_classes = int(num_node_label_classes)
        self.use_node_label_head = bool(use_node_label_head and num_node_label_classes > 0)
        self.num_edge_label_classes = int(num_edge_label_classes)
        self.use_edge_label_head = bool(use_edge_label_head and num_edge_label_classes > 0)
        self.eqm_sigma = float(eqm_sigma)
        self.sampling_step_size = float(sampling_step_size)
        self.sampling_steps = int(sampling_steps)
        self.langevin_noise_scale = float(langevin_noise_scale)
        self.pool_condition_tokens = bool(pool_condition_tokens)
        self.use_guidance = False
        self.noise_degree_factor = 1.0
        self.use_existence_head = True
        self.constant_existence_value = 1.0

        self.register_buffer(
            "exist_pos_weight",
            torch.as_tensor(exist_pos_weight, dtype=torch.float32),
        )
        self.register_buffer("deg_min_val", torch.tensor(degree_min_val, dtype=torch.float32))
        self.register_buffer("deg_range_val", torch.tensor(max(degree_range_val, 1e-8), dtype=torch.float32))

        self.train_losses = []
        self.val_losses = []
        self.train_deg_ce = []
        self.val_deg_ce = []
        self.train_loss_all = []
        self.val_loss_all = []
        self.train_exist = []
        self.val_exist = []
        self.train_node_label_ce = []
        self.val_node_label_ce = []
        self.train_edge_label_ce = []
        self.val_edge_label_ce = []
        self.train_recon = []
        self.val_recon = []
        if self.use_locality_supervision:
            self.train_edge_loss = []
            self.val_edge_loss = []
            self.train_edge_acc = []
            self.val_edge_acc = []
        if self.use_auxiliary_locality_supervision:
            self.train_aux_edge_loss = []
            self.val_aux_edge_loss = []
            self.train_aux_edge_acc = []
            self.val_aux_edge_acc = []

        self.layernorm_in = nn.LayerNorm(input_feature_dimension, elementwise_affine=True)
        self.linear_encoder_input_to_latent = nn.Linear(input_feature_dimension, latent_embedding_dimension)
        self.linear_encoder_condition_to_latent = nn.Linear(condition_feature_dimension, latent_embedding_dimension)
        self.shared_transformer = nn.ModuleList(
            [
                CrossTransformerEncoderLayer(
                    embed_dim=latent_embedding_dimension,
                    num_heads=transformer_attention_head_count,
                    dropout=transformer_dropout,
                )
                for _ in range(number_of_transformer_layers)
            ]
        )
        self.potential_head = nn.Sequential(
            nn.LayerNorm(latent_embedding_dimension),
            nn.Linear(latent_embedding_dimension, latent_embedding_dimension),
            nn.GELU(),
            nn.Linear(latent_embedding_dimension, 1),
        )
        self.degree_head = nn.Linear(latent_embedding_dimension, self.max_degree + 1)
        self.exist_head = nn.Linear(latent_embedding_dimension, 1)
        self.node_label_head = (
            nn.Linear(latent_embedding_dimension, self.num_node_label_classes)
            if self.use_node_label_head
            else None
        )
        self.edge_label_head = None
        if self.use_edge_label_head:
            self.edge_label_head = EdgeMLP(
                latent_dim=latent_embedding_dimension,
                hidden_dim=2 * latent_embedding_dimension,
                dropout=transformer_dropout,
                output_dim=self.num_edge_label_classes,
            )
        if self.use_locality_supervision:
            self.edge_head = EdgeMLP(
                latent_dim=latent_embedding_dimension,
                hidden_dim=2 * latent_embedding_dimension,
                dropout=transformer_dropout,
            )
        if self.use_auxiliary_locality_supervision:
            self.auxiliary_edge_head = EdgeMLP(
                latent_dim=latent_embedding_dimension,
                hidden_dim=2 * latent_embedding_dimension,
                dropout=transformer_dropout,
            )

    def _compute_edge_probability_matrices(self, latent_tokens: torch.Tensor) -> torch.Tensor:
        """Evaluate the edge head on every ordered node pair."""
        batch_size, node_count, latent_dim = latent_tokens.shape
        src = latent_tokens.unsqueeze(2).expand(batch_size, node_count, node_count, latent_dim)
        dst = latent_tokens.unsqueeze(1).expand(batch_size, node_count, node_count, latent_dim)
        edge_logits = self.edge_head(
            src.reshape(-1, latent_dim),
            dst.reshape(-1, latent_dim),
        ).reshape(batch_size, node_count, node_count)
        edge_probs = torch.sigmoid(edge_logits)
        diagonal_mask = torch.eye(node_count, dtype=torch.bool, device=latent_tokens.device).unsqueeze(0)
        edge_probs = edge_probs.masked_fill(diagonal_mask, 0.0)
        return edge_probs

    def _compute_edge_label_logits(self, latent_tokens: torch.Tensor) -> torch.Tensor:
        """Evaluate the edge-label head on every ordered node pair."""
        batch_size, node_count, latent_dim = latent_tokens.shape
        src = latent_tokens.unsqueeze(2).expand(batch_size, node_count, node_count, latent_dim)
        dst = latent_tokens.unsqueeze(1).expand(batch_size, node_count, node_count, latent_dim)
        edge_label_logits = self.edge_label_head(
            src.reshape(-1, latent_dim),
            dst.reshape(-1, latent_dim),
        ).reshape(batch_size, node_count, node_count, self.num_edge_label_classes)
        return edge_label_logits

    def _encode_with_condition(
        self,
        input_rows: torch.Tensor,
        global_condition_vector: torch.Tensor,
        node_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if node_mask is not None:
            input_rows = input_rows * node_mask.unsqueeze(-1).to(dtype=input_rows.dtype)

        x_norm = self.layernorm_in(input_rows)
        latent_tokens = self.linear_encoder_input_to_latent(x_norm)

        if global_condition_vector.dim() == 2:
            cond_tokens = global_condition_vector.unsqueeze(1)
        elif global_condition_vector.dim() == 3:
            cond_tokens = global_condition_vector
        else:
            raise ValueError(
                "global_condition_vector must have shape (B, C) or (B, M, C); "
                f"received {tuple(global_condition_vector.shape)}"
            )

        if self.pool_condition_tokens and cond_tokens.size(1) > 1:
            cond_tokens = cond_tokens.mean(dim=1, keepdim=True)

        weight = self.linear_encoder_condition_to_latent.weight.transpose(0, 1)
        cond_proj = torch.matmul(cond_tokens, weight)
        bias = self.linear_encoder_condition_to_latent.bias
        if bias is not None:
            cond_proj = cond_proj + bias

        if node_mask is not None:
            latent_tokens = latent_tokens * node_mask.unsqueeze(-1).to(dtype=latent_tokens.dtype)

        self_padding_mask = None if node_mask is None else ~node_mask
        for layer in self.shared_transformer:
            latent_tokens = layer(
                latent_tokens,
                k=cond_proj,
                v=cond_proj,
                self_key_padding_mask=self_padding_mask,
                query_mask=node_mask,
            )
            if node_mask is not None:
                latent_tokens = latent_tokens * node_mask.unsqueeze(-1).to(dtype=latent_tokens.dtype)
        return latent_tokens

    def _build_noise_scale(self, x: torch.Tensor) -> torch.Tensor:
        noise_scale = torch.full_like(x, self.eqm_sigma)
        noise_scale[..., self.important_feature_index] /= self.noise_degree_factor
        return noise_scale

    def _compute_score_field(
        self,
        noisy_input: torch.Tensor,
        global_condition_vector: torch.Tensor,
        node_mask: Optional[torch.Tensor] = None,
        *,
        create_graph: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent_tokens = self._encode_with_condition(noisy_input, global_condition_vector, node_mask=node_mask)
        phi_per_node = self.potential_head(latent_tokens).squeeze(-1)
        if node_mask is not None:
            phi_per_node = phi_per_node * node_mask.to(dtype=phi_per_node.dtype)
        phi = phi_per_node.sum(dim=1)
        score = -torch.autograd.grad(
            phi.sum(),
            noisy_input,
            create_graph=create_graph,
            retain_graph=True,
        )[0]
        return score, phi, latent_tokens

    def _eqm_loss(
        self,
        input_examples: torch.Tensor,
        global_condition: torch.Tensor,
        node_mask: Optional[torch.Tensor] = None,
        node_label_targets: Optional[torch.Tensor] = None,
        *,
        create_graph: bool,
    ) -> Tuple[dict, torch.Tensor]:
        eps = torch.randn_like(input_examples)
        noise_scale = self._build_noise_scale(input_examples)
        noisy_input = (input_examples + noise_scale * eps).detach().requires_grad_(True)
        score_mask = (
            node_mask.unsqueeze(-1).to(dtype=input_examples.dtype)
            if node_mask is not None
            else torch.ones_like(input_examples[..., :1])
        )

        score, _, latent_noisy = self._compute_score_field(
            noisy_input,
            global_condition,
            node_mask=node_mask,
            create_graph=create_graph,
        )
        target_score = -eps / noise_scale
        eqm_error = (score - target_score).pow(2) * score_mask
        loss_eqm = eqm_error.sum() / score_mask.sum().clamp_min(1.0)

        denoised = noisy_input + noise_scale.pow(2) * score
        latent_clean = self._encode_with_condition(denoised, global_condition, node_mask=node_mask)
        logits_deg = self.degree_head(latent_clean)
        logits_exist = self.exist_head(latent_clean).squeeze(-1) if self.use_existence_head else None
        logits_label = self.node_label_head(latent_clean) if self.use_node_label_head else None

        if self.use_existence_head:
            target_exist = (input_examples[..., 0] >= 0.5).float()
            exist_loss_map = F.binary_cross_entropy_with_logits(
                logits_exist,
                target_exist,
                pos_weight=self.exist_pos_weight,
                reduction="none",
            )
            exist_mask = node_mask.to(dtype=exist_loss_map.dtype) if node_mask is not None else torch.ones_like(exist_loss_map)
            loss_exist = (exist_loss_map * exist_mask).sum() / exist_mask.sum().clamp_min(1.0)
        else:
            exist_mask = node_mask.to(dtype=input_examples.dtype) if node_mask is not None else torch.ones_like(input_examples[..., 0])
            loss_exist = input_examples.new_zeros(())

        target_degree_scaled = input_examples[..., self.important_feature_index]
        deg_unscaled = target_degree_scaled * self.deg_range_val + self.deg_min_val
        true_deg_class = torch.clamp(torch.round(deg_unscaled), 0, self.max_degree).long()

        if self.degree_temperature is not None and self.degree_temperature > 0:
            logits_for_loss = logits_deg / self.degree_temperature
        else:
            logits_for_loss = logits_deg

        deg_loss_map = F.cross_entropy(
            logits_for_loss.reshape(-1, self.max_degree + 1),
            true_deg_class.reshape(-1),
            reduction="none",
        )
        deg_mask = exist_mask.reshape(-1)
        loss_deg_ce = (deg_loss_map * deg_mask).sum() / deg_mask.sum().clamp_min(1.0)
        if self.use_node_label_head and node_label_targets is not None:
            label_valid_mask = node_mask if node_mask is not None else torch.ones_like(node_label_targets, dtype=torch.bool)
            safe_targets = torch.where(label_valid_mask, node_label_targets, torch.zeros_like(node_label_targets))
            label_loss_map = F.cross_entropy(
                logits_label.reshape(-1, self.num_node_label_classes),
                safe_targets.reshape(-1),
                reduction="none",
            ).reshape_as(safe_targets)
            label_mask = label_valid_mask.to(dtype=label_loss_map.dtype)
            loss_label_ce = (label_loss_map * label_mask).sum() / label_mask.sum().clamp_min(1.0)
        else:
            loss_label_ce = input_examples.new_zeros(())

        total_loss = (
            loss_eqm
            + self.lambda_degree_importance * loss_deg_ce
        )
        if self.use_existence_head:
            total_loss = total_loss + self.lambda_node_exist_importance * loss_exist
        if self.use_node_label_head:
            total_loss = total_loss + self.lambda_node_label_importance * loss_label_ce

        return (
            {
                "total": total_loss,
                "eqm": loss_eqm,
                "exist": loss_exist,
                "deg_ce": loss_deg_ce,
                "label_ce": loss_label_ce,
            },
            latent_clean if self.use_locality_supervision else latent_noisy,
        )

    def training_step(self, batch, batch_idx):
        uses_pairwise_supervision = (
            self.use_locality_supervision
            or self.use_edge_label_head
            or self.use_auxiliary_locality_supervision
        )
        if uses_pairwise_supervision:
            if self.use_node_label_head:
                input_examples, global_condition, edge_idx, edge_labels, edge_label_idx, edge_label_targets, aux_edge_idx, aux_edge_labels, node_mask, node_label_targets = batch
            else:
                input_examples, global_condition, edge_idx, edge_labels, edge_label_idx, edge_label_targets, aux_edge_idx, aux_edge_labels, node_mask = batch
                node_label_targets = None
        else:
            if self.use_node_label_head:
                input_examples, global_condition, node_mask, node_label_targets = batch
            else:
                input_examples, global_condition, node_mask = batch
                node_label_targets = None
            edge_label_idx = torch.empty((0, 3), dtype=torch.long, device=input_examples.device)
            edge_label_targets = torch.empty((0,), dtype=torch.long, device=input_examples.device)
            aux_edge_idx = torch.empty((0, 3), dtype=torch.long, device=input_examples.device)
            aux_edge_labels = torch.empty((0,), dtype=torch.float32, device=input_examples.device)

        losses, latent_tokens = self._eqm_loss(
            input_examples,
            global_condition,
            node_mask=node_mask,
            node_label_targets=node_label_targets,
            create_graph=True,
        )
        total_loss = losses["total"]

        if self.use_locality_supervision and edge_idx.numel() > 0:
            b, i, j = edge_idx.unbind(1)
            h_i = latent_tokens[b, i]
            h_j = latent_tokens[b, j]
            edge_logits = self.edge_head(h_i, h_j)
            edge_loss = F.binary_cross_entropy_with_logits(edge_logits, edge_labels)
            total_loss = total_loss + self.lambda_locality_importance * edge_loss

            with torch.no_grad():
                edge_pred = (torch.sigmoid(edge_logits) > 0.5).float()
                edge_acc = (edge_pred == edge_labels).float().mean()

            self.log("train_edge_ce", edge_loss, on_step=False, on_epoch=True)
            self.log("train_edge_acc", edge_acc, on_step=False, on_epoch=True)

        if self.use_edge_label_head and edge_label_idx.numel() > 0:
            b, i, j = edge_label_idx.unbind(1)
            h_i = latent_tokens[b, i]
            h_j = latent_tokens[b, j]
            edge_label_logits = self.edge_label_head(h_i, h_j)
            edge_label_loss = F.cross_entropy(edge_label_logits, edge_label_targets)
            total_loss = total_loss + self.lambda_edge_label_importance * edge_label_loss
            self.log("train_edge_label_ce", edge_label_loss, on_step=False, on_epoch=True)

        if self.use_auxiliary_locality_supervision and aux_edge_idx.numel() > 0:
            b, i, j = aux_edge_idx.unbind(1)
            h_i = latent_tokens[b, i]
            h_j = latent_tokens[b, j]
            aux_edge_logits = self.auxiliary_edge_head(h_i, h_j)
            aux_edge_loss = F.binary_cross_entropy_with_logits(aux_edge_logits, aux_edge_labels)
            total_loss = total_loss + self.lambda_auxiliary_locality_importance * aux_edge_loss

            with torch.no_grad():
                aux_edge_pred = (torch.sigmoid(aux_edge_logits) > 0.5).float()
                aux_edge_acc = (aux_edge_pred == aux_edge_labels).float().mean()

            self.log("train_aux_locality_ce", aux_edge_loss, on_step=False, on_epoch=True)
            self.log("train_aux_edge_acc", aux_edge_acc, on_step=False, on_epoch=True)

        self.log("train_total", total_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_recon", losses["eqm"], on_step=False, on_epoch=True)
        self.log("train_deg_ce", losses["deg_ce"], on_step=False, on_epoch=True)
        if self.use_existence_head:
            self.log("train_exist", losses["exist"], on_step=False, on_epoch=True)
        if self.use_node_label_head:
            self.log("train_node_label_ce", losses["label_ce"], on_step=False, on_epoch=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        with torch.enable_grad():
            uses_pairwise_supervision = (
                self.use_locality_supervision
                or self.use_edge_label_head
                or self.use_auxiliary_locality_supervision
            )
            if uses_pairwise_supervision:
                if self.use_node_label_head:
                    input_examples, global_condition, edge_idx, edge_labels, edge_label_idx, edge_label_targets, aux_edge_idx, aux_edge_labels, node_mask, node_label_targets = batch
                else:
                    input_examples, global_condition, edge_idx, edge_labels, edge_label_idx, edge_label_targets, aux_edge_idx, aux_edge_labels, node_mask = batch
                    node_label_targets = None
            else:
                if self.use_node_label_head:
                    input_examples, global_condition, node_mask, node_label_targets = batch
                else:
                    input_examples, global_condition, node_mask = batch
                    node_label_targets = None
                edge_label_idx = torch.empty((0, 3), dtype=torch.long, device=input_examples.device)
                edge_label_targets = torch.empty((0,), dtype=torch.long, device=input_examples.device)
                aux_edge_idx = torch.empty((0, 3), dtype=torch.long, device=input_examples.device)
                aux_edge_labels = torch.empty((0,), dtype=torch.float32, device=input_examples.device)

            losses, latent_tokens = self._eqm_loss(
                input_examples,
                global_condition,
                node_mask=node_mask,
                node_label_targets=node_label_targets,
                create_graph=False,
            )
            total_loss = losses["total"]

            if self.use_locality_supervision and edge_idx.numel() > 0:
                b, i, j = edge_idx.unbind(1)
                h_i = latent_tokens[b, i]
                h_j = latent_tokens[b, j]
                edge_logits = self.edge_head(h_i, h_j)
                edge_loss = F.binary_cross_entropy_with_logits(edge_logits, edge_labels)
                total_loss = total_loss + self.lambda_locality_importance * edge_loss

                edge_pred = (torch.sigmoid(edge_logits) > 0.5).float()
                edge_acc = (edge_pred == edge_labels).float().mean()

                self.log("val_edge_ce", edge_loss, on_step=False, on_epoch=True)
                self.log("val_edge_acc", edge_acc, on_step=False, on_epoch=True)

            if self.use_edge_label_head and edge_label_idx.numel() > 0:
                b, i, j = edge_label_idx.unbind(1)
                h_i = latent_tokens[b, i]
                h_j = latent_tokens[b, j]
                edge_label_logits = self.edge_label_head(h_i, h_j)
                edge_label_loss = F.cross_entropy(edge_label_logits, edge_label_targets)
                total_loss = total_loss + self.lambda_edge_label_importance * edge_label_loss
                self.log("val_edge_label_ce", edge_label_loss, on_step=False, on_epoch=True)

            if self.use_auxiliary_locality_supervision and aux_edge_idx.numel() > 0:
                b, i, j = aux_edge_idx.unbind(1)
                h_i = latent_tokens[b, i]
                h_j = latent_tokens[b, j]
                aux_edge_logits = self.auxiliary_edge_head(h_i, h_j)
                aux_edge_loss = F.binary_cross_entropy_with_logits(aux_edge_logits, aux_edge_labels)
                total_loss = total_loss + self.lambda_auxiliary_locality_importance * aux_edge_loss

                aux_edge_pred = (torch.sigmoid(aux_edge_logits) > 0.5).float()
                aux_edge_acc = (aux_edge_pred == aux_edge_labels).float().mean()

                self.log("val_aux_locality_ce", aux_edge_loss, on_step=False, on_epoch=True)
                self.log("val_aux_edge_acc", aux_edge_acc, on_step=False, on_epoch=True)

            self.log("val_total", total_loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log("val_recon", losses["eqm"], on_step=False, on_epoch=True)
            self.log("val_deg_ce", losses["deg_ce"], on_step=False, on_epoch=True)
            if self.use_existence_head:
                self.log("val_exist", losses["exist"], on_step=False, on_epoch=True)
            if self.use_node_label_head:
                self.log("val_node_label_ce", losses["label_ce"], on_step=False, on_epoch=True)
        return total_loss.detach()

    def on_train_end(self):
        if not self.verbose:
            return
        plot_metrics(
            train_metrics={
                "total": self.train_losses,
                "deg_ce": self.train_deg_ce,
                "eqm": self.train_recon,
                **({"exist": self.train_exist} if self.use_existence_head else {}),
                **({"node_label_ce": self.train_node_label_ce} if self.use_node_label_head else {}),
                **({"edge_label_ce": self.train_edge_label_ce} if self.use_edge_label_head else {}),
                **({"edge_ce": self.train_edge_loss} if self.use_locality_supervision else {}),
                **({"aux_locality": self.train_aux_edge_loss} if self.use_auxiliary_locality_supervision else {}),
            },
            val_metrics={
                "total": self.val_losses,
                "deg_ce": self.val_deg_ce,
                "eqm": self.val_recon,
                **({"exist": self.val_exist} if self.use_existence_head else {}),
                **({"node_label_ce": self.val_node_label_ce} if self.use_node_label_head else {}),
                **({"edge_label_ce": self.val_edge_label_ce} if self.use_edge_label_head else {}),
                **({"edge_ce": self.val_edge_loss} if self.use_locality_supervision else {}),
                **({"aux_locality": self.val_aux_edge_loss} if self.use_auxiliary_locality_supervision else {}),
            },
            window=10,
            alpha=0.1,
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def set_guidance_classifier(self, num_classes: int) -> None:
        raise NotImplementedError("Classifier guidance is not implemented for EqMConditionalNodeGenerator phase 1.")

    def train_guidance_classifier(self, *args, **kwargs):
        raise NotImplementedError("Classifier guidance is not implemented for EqMConditionalNodeGenerator phase 1.")

    def generate(
        self,
        global_condition: torch.Tensor,
        total_steps: Optional[int] = None,
        desired_class: Optional[Union[int, Sequence[int]]] = None,
        use_heads_projection: bool = False,
        exist_threshold: float = 0.5,
    ) -> torch.Tensor:
        del desired_class
        self.eval()
        steps = int(total_steps if total_steps is not None else self.sampling_steps)
        eta = self.sampling_step_size
        batch_size = global_condition.size(0)
        device = global_condition.device
        self._last_edge_probability_matrices = None
        self._last_edge_label_matrices = None

        node_mask = None
        x = torch.randn(
            batch_size,
            self.number_of_rows_per_example,
            self.input_feature_dimension,
            device=device,
        )

        for _ in range(steps):
            x = x.detach().requires_grad_(True)
            with torch.enable_grad():
                score, _, latent_tokens = self._compute_score_field(
                    x,
                    global_condition,
                    node_mask=node_mask,
                    create_graph=False,
                )
            x = (x + eta * score).detach()
            if self.langevin_noise_scale > 0:
                x = x + np.sqrt(2.0 * eta) * self.langevin_noise_scale * torch.randn_like(x)

        if use_heads_projection:
            with torch.enable_grad():
                x_for_heads = x.detach().requires_grad_(True)
                _, _, latent_tokens = self._compute_score_field(
                    x_for_heads,
                    global_condition,
                    node_mask=node_mask,
                    create_graph=False,
                )
            with torch.no_grad():
                logits_deg = self.degree_head(latent_tokens)
                logits_exist = self.exist_head(latent_tokens).squeeze(-1) if self.use_existence_head else None
                logits_label = self.node_label_head(latent_tokens) if self.use_node_label_head else None

            if self.use_existence_head:
                exist_probs = torch.sigmoid(logits_exist)
                x[..., 0] = (exist_probs >= exist_threshold).float()
            else:
                x[..., 0] = self.constant_existence_value
            self._last_deg_classes = torch.argmax(logits_deg, dim=-1).detach().cpu()
            if logits_label is not None:
                self._last_node_label_classes = torch.argmax(logits_label, dim=-1).detach().cpu()
            if self.use_locality_supervision:
                edge_probs = self._compute_edge_probability_matrices(latent_tokens)
                self._last_edge_probability_matrices = edge_probs.detach().cpu()
            if self.use_edge_label_head:
                edge_label_logits = self._compute_edge_label_logits(latent_tokens)
                edge_label_classes = torch.argmax(edge_label_logits, dim=-1)
                self._last_edge_label_matrices = edge_label_classes.detach().cpu()

        return x.detach()


class EqMConditionalNodeGenerator(ConditionalNodeGeneratorBase):
    """Scikit-learn friendly facade for a conditional EqM node generator."""

    def __init__(
        self,
        latent_embedding_dimension: int = 128,
        number_of_transformer_layers: int = 4,
        transformer_attention_head_count: int = 4,
        transformer_dropout: float = 0.1,
        learning_rate: float = 1e-3,
        maximum_epochs: int = 10,
        batch_size: int = 32,
        total_steps: int = 100,
        verbose: bool = False,
        important_feature_index: int = 1,
        lambda_degree_importance: float = 1.0,
        noise_degree_factor: float = 2.0,
        degree_temperature: Optional[float] = None,
        lambda_node_exist_importance: float = 1.0,
        lambda_node_label_importance: float = 1.0,
        default_exist_pos_weight: float = 1.0,
        lambda_locality_importance: float = 1.0,
        lambda_auxiliary_locality_importance: float = 1.0,
        lambda_edge_label_importance: float = 1.0,
        use_guidance: bool = False,
        pool_condition_tokens: bool = False,
        use_locality_supervision: bool = False,
        eqm_sigma: float = 0.2,
        sampling_step_size: float = 0.05,
        sampling_steps: Optional[int] = None,
        langevin_noise_scale: float = 0.0,
        require_embedded_node_label_histogram: bool = True,
    ):
        self.latent_embedding_dimension = latent_embedding_dimension
        self.number_of_transformer_layers = number_of_transformer_layers
        self.transformer_attention_head_count = transformer_attention_head_count
        self.transformer_dropout = transformer_dropout
        self.learning_rate = learning_rate
        self.maximum_epochs = maximum_epochs
        self.batch_size = batch_size
        self.total_steps = int(total_steps)
        self.verbose = verbose
        self.important_feature_index = important_feature_index
        self.lambda_degree_importance = lambda_degree_importance
        self.noise_degree_factor = float(noise_degree_factor)
        self.degree_temperature = degree_temperature
        self.lambda_node_exist_importance = lambda_node_exist_importance
        self.lambda_node_label_importance = lambda_node_label_importance
        self.lambda_edge_label_importance = lambda_edge_label_importance
        self.default_exist_pos_weight = default_exist_pos_weight
        self.lambda_locality_importance = lambda_locality_importance
        self.lambda_auxiliary_locality_importance = lambda_auxiliary_locality_importance
        self.use_guidance = bool(use_guidance)
        self.pool_condition_tokens = bool(pool_condition_tokens)
        self.use_locality_supervision = bool(use_locality_supervision)
        self.eqm_sigma = float(eqm_sigma)
        self.sampling_step_size = float(sampling_step_size)
        self.sampling_steps = int(sampling_steps if sampling_steps is not None else total_steps)
        self.langevin_noise_scale = float(langevin_noise_scale)
        self.require_embedded_node_label_histogram = bool(require_embedded_node_label_histogram)
        self.use_existence_head = True
        self.constant_existence_value = 1.0
        self.use_node_label_head = False
        self.node_label_classes_ = None
        self.node_label_to_index_ = None
        self.edge_label_classes_ = None
        self.edge_label_to_index_ = None
        self.base_condition_feature_dimension = None
        self.node_label_histogram_dimension = 0
        self.last_predicted_node_label_classes_ = None
        self.last_predicted_edge_probability_matrices_ = None
        self.last_predicted_edge_label_matrices_ = None

        self.number_of_rows_per_example = None
        self.input_feature_dimension = None
        self.model = None
        self.x_scaler = None
        self.y_scaler = None
        self.D_max = None
        self.condition_token_count = 1
        self.condition_feature_dimension = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.use_guidance:
            raise ValueError("EqMConditionalNodeGenerator phase 1 does not implement classifier guidance.")

    def _plan_channel(self, channel_name: str):
        """Return the named channel from the orchestration supervision plan when available."""
        plan = getattr(self, "supervision_plan_", None)
        if plan is None:
            return None
        return getattr(plan, channel_name, None)

    def _planned_enabled(self, channel_name: str, fallback: bool) -> bool:
        """Resolve whether a supervision channel is intended to be active."""
        channel = self._plan_channel(channel_name)
        if channel is None:
            return fallback
        return bool(getattr(channel, "enabled", fallback))

    def _effective_supervision_flags(
        self,
        edge_pairs: Optional[List[Tuple[int, int, int]]],
        edge_targets: Optional[np.ndarray],
        edge_label_pairs: Optional[List[Tuple[int, int, int]]],
        edge_label_targets: Optional[np.ndarray],
        auxiliary_edge_pairs: Optional[List[Tuple[int, int, int]]],
        auxiliary_edge_targets: Optional[np.ndarray],
    ) -> Tuple[bool, bool, bool]:
        """Combine the supervision plan with the actually supplied arrays."""
        planned_direct_edges = self._planned_enabled("direct_edges", self.use_locality_supervision)
        planned_aux_locality = self._planned_enabled("auxiliary_locality", False)
        planned_edge_labels = self._planned_enabled(
            "edge_labels",
            edge_label_pairs is not None and edge_label_targets is not None,
        )

        effective_locality = planned_direct_edges and edge_pairs is not None and edge_targets is not None
        effective_auxiliary_locality = (
            planned_aux_locality and auxiliary_edge_pairs is not None and auxiliary_edge_targets is not None
        )
        effective_edge_labels = planned_edge_labels and edge_label_pairs is not None and edge_label_targets is not None
        return effective_locality, effective_auxiliary_locality, effective_edge_labels

    def _fit_scalers(self, X_array: np.ndarray, y_array: np.ndarray) -> None:
        self.x_scaler = MinMaxScaler()
        self.y_scaler = MinMaxScaler()
        self.x_scaler.fit(X_array.reshape(-1, X_array.shape[2]))
        if y_array.ndim == 3:
            self.y_scaler.fit(y_array.reshape(-1, y_array.shape[-1]))
        else:
            self.y_scaler.fit(y_array)

    def _transform_data(self, X_array: np.ndarray, y_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        batch_size, row_count, feature_count = X_array.shape
        X_scaled = self.x_scaler.transform(X_array.reshape(-1, feature_count)).reshape(batch_size, row_count, feature_count)
        if y_array.ndim == 3:
            cond_batch, token_count, cond_dim = y_array.shape
            y_scaled = self.y_scaler.transform(y_array.reshape(-1, cond_dim)).reshape(cond_batch, token_count, cond_dim)
        else:
            y_scaled = self.y_scaler.transform(y_array)
        return X_scaled, y_scaled

    def _inverse_transform_input(self, X_array: np.ndarray) -> np.ndarray:
        batch_size, row_count, feature_count = X_array.shape
        X_orig = self.x_scaler.inverse_transform(X_array.reshape(-1, feature_count)).reshape(batch_size, row_count, feature_count)
        X_orig[..., self.important_feature_index] = np.clip(
            X_orig[..., self.important_feature_index],
            0,
            self.D_max,
        )
        return X_orig

    def _fit_node_label_vocab(self, node_label_targets: List[np.ndarray]) -> None:
        flat_labels = [label for labels in node_label_targets for label in np.asarray(labels, dtype=object).tolist()]
        if len(flat_labels) == 0:
            self.node_label_classes_ = np.asarray([], dtype=object)
            self.node_label_to_index_ = {}
            self.use_node_label_head = False
            self.node_label_histogram_dimension = 0
            return
        self.node_label_classes_ = np.unique(np.asarray(flat_labels, dtype=object))
        self.node_label_to_index_ = {label: idx for idx, label in enumerate(self.node_label_classes_)}
        self.node_label_histogram_dimension = int(len(self.node_label_classes_))
        self.use_node_label_head = self.node_label_histogram_dimension > 1

    def _fit_edge_label_vocab(self, edge_label_targets: np.ndarray) -> None:
        flat_labels = np.asarray(edge_label_targets, dtype=object).tolist()
        if len(flat_labels) == 0:
            self.edge_label_classes_ = np.asarray([], dtype=object)
            self.edge_label_to_index_ = {}
            self.use_edge_label_head = False
            return
        self.edge_label_classes_ = np.unique(np.asarray(flat_labels, dtype=object))
        self.edge_label_to_index_ = {label: idx for idx, label in enumerate(self.edge_label_classes_)}
        self.use_edge_label_head = len(self.edge_label_classes_) > 1

    def _encode_edge_label_targets(
        self,
        edge_label_targets: np.ndarray,
    ) -> np.ndarray:
        return np.asarray([self.edge_label_to_index_[label] for label in np.asarray(edge_label_targets, dtype=object)], dtype=np.int64)

    def _encode_node_label_targets(
        self,
        node_label_targets: List[np.ndarray],
        max_num_rows: int,
    ) -> np.ndarray:
        encoded = np.zeros((len(node_label_targets), max_num_rows), dtype=np.int64)
        for graph_idx, labels in enumerate(node_label_targets):
            labels = np.asarray(labels, dtype=object)
            for node_idx, label in enumerate(labels[:max_num_rows]):
                encoded[graph_idx, node_idx] = self.node_label_to_index_[label]
        return encoded

    def _compute_node_label_histograms(self, encoded_targets: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
        histograms = np.zeros((encoded_targets.shape[0], self.node_label_histogram_dimension), dtype=float)
        if self.node_label_histogram_dimension == 0:
            return histograms
        for graph_idx in range(encoded_targets.shape[0]):
            valid_indices = encoded_targets[graph_idx][valid_mask[graph_idx]]
            if valid_indices.size == 0:
                continue
            counts = np.bincount(valid_indices, minlength=self.node_label_histogram_dimension).astype(float)
            total = counts.sum()
            if total > 0:
                histograms[graph_idx] = counts / total
        return histograms

    def _condition_encodings_include_histogram(
        self,
        y_array: np.ndarray,
        node_label_histograms: np.ndarray,
    ) -> bool:
        if (
            not self.require_embedded_node_label_histogram
            or self.node_label_histogram_dimension == 0
        ):
            return False
        if y_array.ndim == 2:
            if y_array.shape[1] < self.node_label_histogram_dimension:
                return False
            return np.allclose(
                y_array[:, -self.node_label_histogram_dimension :],
                node_label_histograms,
                atol=1e-8,
            )
        if y_array.ndim == 3:
            if y_array.shape[2] < self.node_label_histogram_dimension:
                return False
            expected_hist = np.repeat(node_label_histograms[:, None, :], y_array.shape[1], axis=1)
            return np.allclose(
                y_array[:, :, -self.node_label_histogram_dimension :],
                expected_hist,
                atol=1e-8,
            )
        return False

    def setup(
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
    ):
        edge_label_plan = self._plan_channel("edge_labels")
        planned_direct_edges = self._planned_enabled("direct_edges", self.use_locality_supervision)
        planned_aux_locality = self._planned_enabled("auxiliary_locality", False)
        planned_learned_edge_labels = (
            getattr(edge_label_plan, "mode", None) == "learned"
            if edge_label_plan is not None
            else edge_label_pairs is not None and edge_label_targets is not None
        )
        effective_locality, effective_auxiliary_locality, effective_edge_labels = self._effective_supervision_flags(
            edge_pairs,
            edge_targets,
            edge_label_pairs,
            edge_label_targets,
            auxiliary_edge_pairs,
            auxiliary_edge_targets,
        )
        if planned_direct_edges and not effective_locality and self.verbose:
            print("Direct-edge channel planned as learned, but usable edge_pairs/edge_targets were not supplied.")
        if planned_aux_locality and not effective_auxiliary_locality and self.verbose:
            print("Auxiliary-locality channel planned as learned, but usable auxiliary_edge_pairs/auxiliary_edge_targets were not supplied.")
        if planned_learned_edge_labels and not effective_edge_labels and self.verbose:
            print("Edge-label channel planned as learned, but usable edge_label_pairs/edge_label_targets were not supplied.")
        if not effective_locality:
            edge_pairs = None
            edge_targets = None
        if not effective_auxiliary_locality:
            auxiliary_edge_pairs = None
            auxiliary_edge_targets = None
        if not effective_edge_labels:
            edge_label_pairs = None
            edge_label_targets = None

        max_num_rows = max(x.shape[0] for x in node_encodings_list)
        self.number_of_rows_per_example = max_num_rows

        padded_examples = []
        for x in node_encodings_list:
            if x.shape[0] < max_num_rows:
                x = np.pad(x, ((0, max_num_rows - x.shape[0]), (0, 0)), mode="constant", constant_values=0)
            padded_examples.append(x)

        X_array = np.stack(padded_examples, axis=0)
        y_array = np.array(conditional_graph_encodings)
        if y_array.ndim == 1:
            y_array = y_array[:, None]
        if y_array.ndim not in (2, 3):
            raise ValueError(
                "conditional_graph_encodings must be array-like of shape (B, C) or (B, M, C); "
                f"received shape {y_array.shape}"
            )
        self.condition_token_count = 1 if y_array.ndim == 2 else y_array.shape[1]

        if node_mask is None:
            valid_mask = np.zeros((len(node_encodings_list), max_num_rows), dtype=bool)
            for idx, x in enumerate(node_encodings_list):
                valid_mask[idx, :x.shape[0]] = True
        else:
            valid_mask = np.asarray(node_mask, dtype=bool)

        encoded_node_label_targets = None
        if node_label_targets is not None:
            self._fit_node_label_vocab(node_label_targets)
            encoded_node_label_targets = self._encode_node_label_targets(node_label_targets, max_num_rows)
            if self.node_label_histogram_dimension > 0:
                histogram_array = self._compute_node_label_histograms(encoded_node_label_targets, valid_mask)
                already_augmented = self._condition_encodings_include_histogram(y_array, histogram_array)
                if not already_augmented:
                    raise ValueError(
                        "EqMConditionalNodeGenerator expects conditioning vectors that already include "
                        "the node label histogram extracted from the real graph. Use "
                        "DecompositionalEncoderDecoder.graph_encode(graphs) or encode(graphs)."
                    )
                self.base_condition_feature_dimension = y_array.shape[-1] - self.node_label_histogram_dimension
            else:
                self.base_condition_feature_dimension = y_array.shape[-1]
        else:
            self.base_condition_feature_dimension = y_array.shape[-1]
            self.use_node_label_head = False
            self.node_label_histogram_dimension = 0
            self.node_label_classes_ = None
            self.node_label_to_index_ = None

        if edge_label_targets is not None:
            self._fit_edge_label_vocab(edge_label_targets)
        else:
            self.use_edge_label_head = False
            self.edge_label_classes_ = None
            self.edge_label_to_index_ = None

        if y_array.ndim == 2:
            self.condition_feature_dimension = y_array.shape[1]
        else:
            self.condition_feature_dimension = y_array.shape[2]

        self._fit_scalers(X_array, y_array)

        exist_mask = X_array[..., 0] >= 0.5
        ones = int(exist_mask.sum())
        zeros = int(exist_mask.size) - ones
        if ones == 0:
            exist_pos_weight = float(self.default_exist_pos_weight)
        elif zeros > ones:
            exist_pos_weight = float(zeros) / float(ones)
        else:
            exist_pos_weight = 1.0

        deg_column = X_array[..., self.important_feature_index].reshape(-1, 1)
        degree_scaler = MinMaxScaler().fit(deg_column)
        deg_min_val = degree_scaler.data_min_[0]
        deg_range_val = degree_scaler.data_range_[0]
        if deg_range_val == 0:
            deg_range_val = 1e-8

        X_scaled, y_scaled = self._transform_data(X_array, y_array)
        self.input_feature_dimension = X_scaled.shape[2]
        self.D_max = int(X_array[..., self.important_feature_index].max())
        valid_exist_targets = (X_array[..., 0] >= 0.5)[valid_mask]
        all_same_node_size = len(set(x.shape[0] for x in node_encodings_list)) == 1
        disable_existence = (
            all_same_node_size
            and valid_exist_targets.size > 0
            and np.all(valid_exist_targets == valid_exist_targets.flat[0])
        )
        self.use_existence_head = not disable_existence
        if disable_existence:
            self.constant_existence_value = float(valid_exist_targets.flat[0])
            if self.verbose:
                print(
                    "Existence supervision disabled: all training graphs have the same node count "
                    "and the valid existence target is constant."
                )
        else:
            self.constant_existence_value = 1.0

        if self.verbose:
            if effective_locality:
                print("Direct edge supervision enabled: horizon-1 edge presence will be learned and used by the decoder.")
            elif planned_direct_edges:
                print("Direct edge supervision disabled at setup time: the plan requested it but usable training pairs were not supplied.")
            else:
                print("Direct edge supervision disabled: the supervision plan does not request horizon-1 edge prediction.")

            if effective_auxiliary_locality:
                print("Auxiliary locality supervision enabled: higher-horizon locality pairs will be used only as an encoding regularizer.")
            elif effective_locality:
                print("Auxiliary locality supervision disabled: only direct edge supervision will be used; no extra locality head is trained.")
            elif planned_aux_locality:
                print("Auxiliary locality supervision disabled at setup time: the plan requested it but usable higher-horizon pairs were not supplied.")
            else:
                print("Auxiliary locality supervision disabled: the supervision plan does not request higher-horizon locality regularization.")

            if self.use_edge_label_head and effective_edge_labels:
                print(
                    "Edge-label supervision enabled: discrete edge labels will be predicted by the generator."
                )
            elif planned_learned_edge_labels:
                print("Edge-label supervision disabled at setup time: the plan requested it but usable labelled edges were not supplied.")
            elif edge_label_plan is not None and edge_label_plan.mode == "constant":
                print(
                    f"Edge-label supervision collapsed to a constant label: {edge_label_plan.constant_value}."
                )
            elif edge_label_targets is not None:
                print(
                    "Edge-label supervision collapsed to a constant label: no edge-label head will be trained."
                )
            else:
                print("Edge-label supervision disabled: the supervision plan does not request learned edge-label prediction.")

        self.model = EqMConditionalNodeGeneratorModule(
            number_of_rows_per_example=self.number_of_rows_per_example,
            input_feature_dimension=self.input_feature_dimension,
            condition_feature_dimension=self.condition_feature_dimension,
            latent_embedding_dimension=self.latent_embedding_dimension,
            number_of_transformer_layers=self.number_of_transformer_layers,
            transformer_attention_head_count=self.transformer_attention_head_count,
            transformer_dropout=self.transformer_dropout,
            learning_rate=self.learning_rate,
            verbose=self.verbose,
            important_feature_index=self.important_feature_index,
            max_degree=self.D_max,
            lambda_degree_importance=self.lambda_degree_importance,
            degree_temperature=self.degree_temperature,
            degree_min_val=deg_min_val,
            degree_range_val=deg_range_val,
            lambda_node_exist_importance=self.lambda_node_exist_importance,
            lambda_node_label_importance=self.lambda_node_label_importance,
            lambda_edge_label_importance=self.lambda_edge_label_importance,
            use_locality_supervision=effective_locality,
            lambda_locality_importance=self.lambda_locality_importance,
            use_auxiliary_locality_supervision=effective_auxiliary_locality,
            lambda_auxiliary_locality_importance=self.lambda_auxiliary_locality_importance,
            exist_pos_weight=exist_pos_weight,
            num_node_label_classes=self.node_label_histogram_dimension,
            use_node_label_head=self.use_node_label_head,
            num_edge_label_classes=0 if self.edge_label_classes_ is None else len(self.edge_label_classes_),
            use_edge_label_head=self.use_edge_label_head,
            eqm_sigma=self.eqm_sigma,
            sampling_step_size=self.sampling_step_size,
            sampling_steps=self.sampling_steps,
            langevin_noise_scale=self.langevin_noise_scale,
            pool_condition_tokens=self.pool_condition_tokens,
        )
        self.model.noise_degree_factor = self.noise_degree_factor
        self.model.use_existence_head = self.use_existence_head
        self.model.constant_existence_value = self.constant_existence_value

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
    ):
        X_array = np.stack(
            [
                np.pad(x, ((0, self.number_of_rows_per_example - x.shape[0]), (0, 0)), mode="constant", constant_values=0)
                for x in node_encodings_list
            ],
            axis=0,
        )
        y_array = np.array(conditional_graph_encodings)
        if y_array.ndim == 1:
            y_array = y_array[:, None]
        mask_array = np.zeros((len(node_encodings_list), self.number_of_rows_per_example), dtype=bool)
        for index, x in enumerate(node_encodings_list):
            mask_array[index, :x.shape[0]] = True
        if node_mask is not None:
            mask_array = np.asarray(node_mask, dtype=bool)
        encoded_node_label_targets = None
        encoded_edge_label_targets = None
        if node_label_targets is not None and self.node_label_histogram_dimension > 0:
            encoded_node_label_targets = self._encode_node_label_targets(node_label_targets, self.number_of_rows_per_example)
            histogram_array = self._compute_node_label_histograms(encoded_node_label_targets, mask_array)
            if not self._condition_encodings_include_histogram(y_array, histogram_array):
                raise ValueError(
                    "EqMConditionalNodeGenerator.fit() expects conditioning vectors that already include "
                    "the node label histogram extracted from the real graph."
                )
        if edge_label_targets is not None and self.edge_label_to_index_ is not None:
            encoded_edge_label_targets = self._encode_edge_label_targets(edge_label_targets)
        X_scaled, y_scaled = self._transform_data(X_array, y_array)

        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y_scaled, dtype=torch.float32)
        mask_tensor = torch.tensor(mask_array, dtype=torch.bool)

        effective_locality, effective_auxiliary_locality, effective_edge_labels = self._effective_supervision_flags(
            edge_pairs,
            edge_targets,
            edge_label_pairs,
            encoded_edge_label_targets,
            auxiliary_edge_pairs,
            auxiliary_edge_targets,
        )
        if effective_locality or effective_auxiliary_locality or effective_edge_labels:
            dataset = EqMGraphWithEdgesDataset(
                X_scaled,
                y_scaled,
                edge_pairs,
                edge_targets,
                edge_label_pairs,
                encoded_edge_label_targets,
                auxiliary_edge_pairs,
                auxiliary_edge_targets,
                mask_array,
                encoded_node_label_targets,
            )
            dataset_size = len(node_encodings_list)
            train_size = int(0.9 * dataset_size)
            val_size = dataset_size - train_size
            indices = torch.randperm(dataset_size).tolist()
            train_indices = indices[:train_size]
            val_indices = indices[train_size:]
            train_dataset = torch.utils.data.Subset(dataset, train_indices)
            val_dataset = torch.utils.data.Subset(dataset, val_indices)
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                collate_fn=collate_eqm_graph_with_edges,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=collate_eqm_graph_with_edges,
            )
        else:
            if encoded_node_label_targets is None:
                dataset = TensorDataset(X_tensor, y_tensor, mask_tensor)
            else:
                label_tensor = torch.tensor(encoded_node_label_targets, dtype=torch.long)
                dataset = TensorDataset(X_tensor, y_tensor, mask_tensor, label_tensor)
            train_size = int(0.9 * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        trainer = pl.Trainer(
            max_epochs=self.maximum_epochs,
            callbacks=[MetricsLogger()],
            logger=True,
            enable_checkpointing=False,
            enable_progress_bar=False,
        )
        if not self.verbose:
            with suppress_output():
                run_trainer_fit(
                    trainer,
                    self.model,
                    train_loader,
                    val_loader,
                    context=f"{self.__class__.__name__}.fit",
                )
        else:
            run_trainer_fit(
                trainer,
                self.model,
                train_loader,
                val_loader,
                context=f"{self.__class__.__name__}.fit",
            )

    def predict(
        self,
        conditional_graph_encodings: Any,
        desired_class: Optional[Union[int, Sequence[int]]] = None,
    ) -> List[np.ndarray]:
        if desired_class is not None and int(self.verbose) >= 2:
            print("EqMConditionalNodeGenerator phase 1 ignores desired_class because classifier guidance is not implemented.")

        self.device = next(self.model.parameters()).device
        cond_array = np.asarray(conditional_graph_encodings)
        if cond_array.ndim == 1:
            cond_array = cond_array[:, None]
        expected_aug_dim = None
        if cond_array.ndim == 2:
            expected_aug_dim = self.base_condition_feature_dimension + self.node_label_histogram_dimension
        elif cond_array.ndim == 3:
            expected_aug_dim = self.base_condition_feature_dimension + self.node_label_histogram_dimension
        if (
            self.require_embedded_node_label_histogram
            and self.node_label_histogram_dimension > 0
            and cond_array.shape[-1] != expected_aug_dim
        ):
            raise ValueError(
                "EqMConditionalNodeGenerator.predict() expects conditioning vectors that already include "
                "the node label histogram extracted from a real graph; "
                f"expected feature dimension {expected_aug_dim}, "
                f"received {cond_array.shape[-1]}"
            )
        if cond_array.ndim == 3:
            cond_scaled = self.y_scaler.transform(
                cond_array.reshape(-1, cond_array.shape[-1])
            ).reshape(cond_array.shape)
        else:
            cond_scaled = self.y_scaler.transform(cond_array)
        cond_tensor = torch.tensor(cond_scaled, dtype=torch.float32, device=self.device)

        generated = self.model.generate(
            cond_tensor,
            total_steps=self.sampling_steps,
            desired_class=desired_class,
            use_heads_projection=True,
        )

        gen_np = generated.detach().cpu().numpy()
        gen_orig = self._inverse_transform_input(gen_np)
        if not self.model.use_existence_head:
            for index in range(len(gen_orig)):
                gen_orig[index][..., 0] = self.constant_existence_value

        deg_classes = getattr(self.model, "_last_deg_classes", None)
        if deg_classes is not None:
            deg_classes = deg_classes.cpu().numpy()
            for index in range(len(gen_orig)):
                gen_orig[index][..., self.important_feature_index] = np.clip(deg_classes[index], 0, None)
        label_classes = getattr(self.model, "_last_node_label_classes", None)
        self.last_predicted_node_label_classes_ = None
        if label_classes is not None and self.node_label_classes_ is not None:
            label_classes = label_classes.cpu().numpy()
            self.last_predicted_node_label_classes_ = [
                self.node_label_classes_[label_classes[index]]
                for index in range(label_classes.shape[0])
            ]
        edge_probability_matrices = getattr(self.model, "_last_edge_probability_matrices", None)
        self.last_predicted_edge_probability_matrices_ = None
        if edge_probability_matrices is not None:
            edge_probability_matrices = edge_probability_matrices.cpu().numpy()
            self.last_predicted_edge_probability_matrices_ = [
                edge_probability_matrices[index]
                for index in range(edge_probability_matrices.shape[0])
            ]
        edge_label_matrices = getattr(self.model, "_last_edge_label_matrices", None)
        self.last_predicted_edge_label_matrices_ = None
        if edge_label_matrices is not None and self.edge_label_classes_ is not None:
            edge_label_matrices = edge_label_matrices.cpu().numpy()
            self.last_predicted_edge_label_matrices_ = [
                self.edge_label_classes_[edge_label_matrices[index]]
                for index in range(edge_label_matrices.shape[0])
            ]

        return gen_orig

    def plot_metrics(self, window: int = 10, alpha: float = 0.3):
        if self.model is None:
            print("Model is not fitted yet.")
            return
        plot_metrics(
            train_metrics={
                "total": self.model.train_losses,
                "deg_ce": self.model.train_deg_ce,
                "eqm": self.model.train_recon,
                **({"exist": self.model.train_exist} if self.model.use_existence_head else {}),
                **({"node_label_ce": self.model.train_node_label_ce} if self.model.use_node_label_head else {}),
                **({"edge_label_ce": self.model.train_edge_label_ce} if self.model.use_edge_label_head else {}),
                **({"edge_ce": self.model.train_edge_loss} if self.model.use_locality_supervision else {}),
                **({"aux_locality": self.model.train_aux_edge_loss} if self.model.use_auxiliary_locality_supervision else {}),
            },
            val_metrics={
                "total": self.model.val_losses,
                "deg_ce": self.model.val_deg_ce,
                "eqm": self.model.val_recon,
                **({"exist": self.model.val_exist} if self.model.use_existence_head else {}),
                **({"node_label_ce": self.model.val_node_label_ce} if self.model.use_node_label_head else {}),
                **({"edge_label_ce": self.model.val_edge_label_ce} if self.model.use_edge_label_head else {}),
                **({"edge_ce": self.model.val_edge_loss} if self.model.use_locality_supervision else {}),
                **({"aux_locality": self.model.val_aux_edge_loss} if self.model.use_auxiliary_locality_supervision else {}),
            },
            window=window,
            alpha=alpha,
        )
