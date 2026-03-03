import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt
import contextlib, os, sys
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import random_split, DataLoader, TensorDataset, Dataset
from typing import Dict, Sequence, Optional, Union, Tuple, List, Any
import math
from sklearn.model_selection import train_test_split

from .lightning_utils import run_trainer_fit

# --- Utility Context Manager ---
@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

# --- Sinusoidal Time Embedding ---
def get_sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Encode diffusion time steps `t` into sinusoidal position embeddings.

    Args:
        t (torch.Tensor): Batch of diffusion step indices with shape `(B,)` or `(B, 1)`.
        dim (int): Dimensionality of the resulting positional embedding; must be even.

    Returns:
        torch.Tensor: Sinusoidal embeddings with shape `(B, dim)`.
    """
    half_dim = dim // 2
    inv_freq = torch.exp(
        torch.arange(0, half_dim, device=t.device).float() * (-math.log(10000) / (half_dim - 1))
    )
    # Shape: (B,1) * (D/2,) -> (B,D/2)
    angles = t * inv_freq.view(1, -1)
    # Shape: (B,D)
    return torch.cat([angles.sin(), angles.cos()], dim=-1)

# --- Cross-Attention Transformer Layer ---
class CrossTransformerEncoderLayer(nn.Module):
    """Pre-norm transformer encoder layer with shared self- and cross-attention.

    Args:
        embed_dim (int): Dimensionality of both the node and condition token
            embeddings processed by the layer.
        num_heads (int): Number of attention heads used for both self- and
            cross-attention blocks.
        dropout (float): Dropout probability applied to attention outputs and the
            feed-forward network.
    """
    def __init__(self, 
                 embed_dim: int,
                 num_heads: int,
                 dropout: float = 0.1):
        super().__init__()
        # Self attention block
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-5)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads, 
            dropout=dropout,
            batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout)
        
        # Cross attention block with memory tokens
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-5)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.dropout2 = nn.Dropout(dropout)
        
        # Feed-forward block with increased width
        self.norm3 = nn.LayerNorm(embed_dim, eps=1e-5)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.dropout3 = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        self_key_padding_mask: Optional[torch.Tensor] = None,
        cross_key_padding_mask: Optional[torch.Tensor] = None,
        query_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Mix node tokens with conditioning tokens via self- and cross-attention."""
        # Pre-norm architecture (more stable training)
        # Self attention
        x1 = self.norm1(x)
        x = x + self.dropout1(
            self.self_attn(
                x1,
                x1,
                x1,
                key_padding_mask=self_key_padding_mask,
            )[0]
        )
        if query_mask is not None:
            x = x * query_mask.unsqueeze(-1).to(dtype=x.dtype)
        
        # Cross attention with memory tokens
        x2 = self.norm2(x) 
        x = x + self.dropout2(
            self.cross_attn(
                x2,
                k,
                v,
                key_padding_mask=cross_key_padding_mask,
            )[0]
        )
        if query_mask is not None:
            x = x * query_mask.unsqueeze(-1).to(dtype=x.dtype)
        
        # Feed-forward
        x3 = self.norm3(x)
        x = x + self.dropout3(self.ff(x3))
        if query_mask is not None:
            x = x * query_mask.unsqueeze(-1).to(dtype=x.dtype)
        return x

# --- Plotting Metrics ---
def plot_metrics(
    train_metrics: Dict[str, Sequence[float]],
    val_metrics: Dict[str, Sequence[float]],
    window: int = 10,
    alpha: float = 0.3
) -> None:
    """Visualise training and validation metrics with a geometric moving average.

    Args:
        train_metrics (Dict[str, Sequence[float]]): History of metric values recorded
            on the training split keyed by metric name.
        val_metrics (Dict[str, Sequence[float]]): History of metric values recorded on
            the validation split keyed by metric name.
        window (int): Window size for the geometric moving average applied to the
            curves.
        alpha (float): Transparency used when plotting the raw metric traces.
    """
    def _moving_average(data: Sequence[float], window_size: int) -> np.ndarray:
        arr = np.asarray(data, dtype=float)
        if len(arr) < window_size:
            return np.array([])
        arr_clamped = np.where(arr <= 0, np.finfo(float).tiny, arr)
        log_arr = np.log(arr_clamped)
        kernel = np.ones(window_size, dtype=float) / window_size
        smoothed_log = np.convolve(log_arr, kernel, mode='valid')
        return np.exp(smoothed_log)

    fig, ax0 = plt.subplots(figsize=(15, 8))
    metrics = list(train_metrics.keys())
    axes = [ax0] + [ax0.twinx() for _ in range(len(metrics) - 1)]
    for i, ax in enumerate(axes[1:], start=1):
        ax.spines['right'].set_position(('outward', 60 * i))
    colors = ["blue", "red", "green", "purple", "orange"]
    lines, labels = [], []
    for name, ax, color in zip(metrics, axes, colors):
        train_vals = train_metrics[name]
        val_vals = val_metrics[name]
        if len(train_vals) < 1 or len(val_vals) < 1:
            continue
        N = min(len(train_vals), len(val_vals))
        train = train_vals[:N]
        val = val_vals[:N]
        epochs = np.arange(1, N+1)
        ax.plot(epochs, train, color=color, alpha=alpha)
        ax.plot(epochs, val, color=color, linestyle='--', alpha=alpha)
        sm_train = _moving_average(train, window)
        sm_val = _moving_average(val, window)
        if sm_train.size:
            sm_epochs = np.arange(window, window + len(sm_train))
            l1, = ax.plot(sm_epochs, sm_train, color=color, linewidth=2,
                       label=f"Train {name} (MA{window})")
            l2, = ax.plot(sm_epochs, sm_val, color=color, linewidth=2, linestyle='--',
                       label=f"Val {name} (MA{window})")
            lines += [l1, l2]
            labels += [f"Train {name} (MA{window})", f"Val {name} (MA{window})"]
        ax.set_ylabel(name, color=color)
        ax.tick_params(axis='y', labelcolor=color)
        ax.set_yscale('log')
    
    fig.legend(lines, labels, loc='upper center', ncol=len(lines)//2, fontsize='small')
    ax0.set_xlabel("Epoch")
    ax0.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    plt.show()

# =============================================================================
# Revised IterativeDenoisingAutoencoderTransformerModel with Cross-Attention
# =============================================================================
class GuidanceMLP(nn.Module):
    """Compact classifier used for guidance during sampling.

    Args:
        input_dim (int): Size of the pooled latent vectors fed into the MLP.
        hidden_dim (int): Width of the hidden layer that projects latent features
            before the output layer.
        output_dim (int): Number of guidance classes to predict during sampling.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.2
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim,  hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class EdgeMLP(nn.Module):
    """Single-hidden-layer MLP that scores locality between two node embeddings.

    Args:
        latent_dim (int): Width of the node latent representations concatenated and
            compared by the network.
        hidden_dim (Optional[int]): Size of the hidden layer; defaults to twice the
            latent dimension when not provided.
        dropout (float): Dropout probability applied to the hidden activations.
    """
    def __init__(self, latent_dim: int, hidden_dim: Optional[int] = None, dropout: float = 0.1, output_dim: int = 1):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 2 * latent_dim
        self.output_dim = int(output_dim)
        self.mlp = nn.Sequential(
            nn.Linear(4 * latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.output_dim)
        )

    def forward(self, h_i: torch.Tensor, h_j: torch.Tensor) -> torch.Tensor:
        """Return a scalar logit describing how strongly two nodes should connect."""
        diff = torch.abs(h_i - h_j)
        prod = h_i * h_j
        x = torch.cat([h_i, h_j, diff, prod], dim=-1)
        out = self.mlp(x)
        return out.squeeze(-1) if self.output_dim == 1 else out


class IterativeDenoisingAutoencoderTransformerModel(pl.LightningModule):
    """Cross-attentional diffusion model for conditional node generation.

    Args:
        number_of_rows_per_example (int): Number of node tokens each training sample
            contains; effectively the maximum node count per graph.
        input_feature_dimension (int): Width of the node feature vectors supplied to
            the diffusion model.
        condition_feature_dimension (int): Width of the conditioning vector appended
            to every graph instance.
        latent_embedding_dimension (int): Size of the latent space used by the
            transformer blocks and decoder head.
        number_of_transformer_layers (int): How many shared cross-attention encoder
            layers to stack in the transformer backbone.
        transformer_attention_head_count (int): Number of attention heads inside each
            transformer layer.
        transformer_dropout (float): Dropout probability applied within attention and
            feed-forward sublayers.
        learning_rate (float): Optimiser step size used by Lightning during training.
        verbose (bool): Whether to emit additional logging such as sample previews.
        important_feature_index (int): Column index of the feature that tracks node
            degree and receives specialised scaling and losses.
        max_degree (Optional[int]): Maximum discrete degree label expected by the
            auxiliary classifier head.
        lambda_degree_importance (float): Weight assigned to the degree prediction
            loss when combining training objectives.
        noise_degree_factor (float): Factor used to reduce noise applied to the degree
            feature during diffusion.
        degree_temperature (Optional[float]): Optional softmax temperature used when
            calibrating the degree logits.
        degree_min_val (float): Minimum degree value observed during scaling; stored
            for inverse transformations.
        degree_range_val (float): Range of the scaled degree feature used for
            unnormalising predictions.
        lambda_node_exist_importance (float): Weight assigned to the node existence
            loss component.
        use_locality_supervision (bool): Whether to train the optional locality edge
            scoring head.
        lambda_locality_importance (float): Scaling factor for the locality loss if
            supervision is enabled.
        exist_pos_weight (Union[torch.Tensor, float]): Positive class weight for the
            node existence classifier head.
        use_guidance (bool): Enable classifier guidance during sampling by training an
            auxiliary MLP on latent representations.
        guidance_weight (float): Multiplier applied to the guidance gradient injected
            during sampling.
        sigma_min (float): Lowest noise level used in the diffusion schedule.
        sigma_max (float): Highest noise level used in the diffusion schedule.
        sampling_final_sigma (float): Noise level to terminate the sampling loop at.
        pool_condition_tokens (bool): Whether to average condition tokens before they
            interact with node tokens in cross-attention layers.
    """
    def __init__(self,
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
                 noise_degree_factor: float = 2.0,
                 degree_temperature: Optional[float] = None,
                 degree_min_val: float = 0.0, # Changed from degree_median
                 degree_range_val: float = 1.0, # Changed from degree_iqr
                 lambda_node_exist_importance: float = 1.0,
                 use_locality_supervision: bool = False,
                 lambda_locality_importance: float = 1.0,
                 use_auxiliary_locality_supervision: bool = False,
                 lambda_auxiliary_locality_importance: float = 1.0,
                 exist_pos_weight: Union[torch.Tensor, float] = 1.0,
                 use_guidance: bool = False,
                 guidance_weight: float = 1.0,
                 sigma_min: float = 0.1,
                 sigma_max: float = 1.0,
                 sampling_final_sigma: float = 0.0,
                 pool_condition_tokens: bool = False):
        super().__init__()
        self.save_hyperparameters(ignore=['verbose'])
        # Must set use_locality_supervision _before_ we refer to it below:
        self.use_locality_supervision = use_locality_supervision
        self.use_auxiliary_locality_supervision = use_auxiliary_locality_supervision

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
        self.pool_condition_tokens = bool(pool_condition_tokens)

        if max_degree is None:
            raise ValueError("max_degree must be provided when initializing the diffusion model.")
        if int(max_degree) < 0:
            raise ValueError(f"max_degree must be non-negative (got {max_degree}).")
        self.max_degree = int(max_degree)

        # Centralise diffusion schedule metadata (backward-compatible defaults handled later)
        self._set_schedule_metadata(
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            sampling_final_sigma=sampling_final_sigma,
        )

        self.lambda_degree_importance = lambda_degree_importance
        self.noise_degree_factor = noise_degree_factor
        self.degree_temperature = degree_temperature
        self.lambda_node_exist_importance = lambda_node_exist_importance
        self.register_buffer(
            "exist_pos_weight",
            torch.as_tensor(exist_pos_weight, dtype=torch.float32)
        )

        # ----------  guidance flags ----------
        self.use_guidance = use_guidance
        self.guidance_weight = guidance_weight
        self.guidance_classifier: Optional[GuidanceMLP] = None

        # Store degree scaling parameters (MinMaxScaler based)
        self.register_buffer('deg_min_val', torch.tensor(degree_min_val, dtype=torch.float32))
        # Ensure range is not zero to avoid division by zero in scaling
        self.register_buffer('deg_range_val', torch.tensor(max(degree_range_val, 1e-8), dtype=torch.float32))

        # Initialize metric lists
        self.train_losses = []
        self.val_losses   = []
        self.train_deg_ce = []
        self.val_deg_ce   = []
        self.train_loss_all = []
        self.val_loss_all   = []
        self.train_exist    = []
        self.val_exist      = []
        self.train_recon = []
        self.val_recon = []
        if self.use_locality_supervision:
            self.train_edge_loss = []
            self.val_edge_loss   = []
            self.train_edge_acc = []
            self.val_edge_acc = []
        if self.use_auxiliary_locality_supervision:
            self.train_aux_edge_loss = []
            self.val_aux_edge_loss = []
            self.train_aux_edge_acc = []
            self.val_aux_edge_acc = []

        # Model layers
        self.layernorm_in = nn.LayerNorm(input_feature_dimension, elementwise_affine=True)
        self.linear_encoder_input_to_latent = nn.Linear(input_feature_dimension, latent_embedding_dimension)
        self.linear_encoder_condition_to_latent = nn.Linear(condition_feature_dimension, latent_embedding_dimension)
        
        # Replace transformer with cross-attention version
        self.shared_transformer = nn.ModuleList([
            CrossTransformerEncoderLayer(
                embed_dim=latent_embedding_dimension,
                num_heads=transformer_attention_head_count,
                dropout=transformer_dropout,
            ) for _ in range(number_of_transformer_layers)
        ])
        
        self.linear_decoder_latent_to_output = nn.Linear(latent_embedding_dimension, input_feature_dimension)
        self.degree_head = nn.Linear(latent_embedding_dimension, max_degree + 1)
        self.exist_head = nn.Linear(latent_embedding_dimension, 1)
        self.lambda_locality_importance = lambda_locality_importance
        self.lambda_auxiliary_locality_importance = lambda_auxiliary_locality_importance
        if self.use_locality_supervision:
            self.edge_head = EdgeMLP(
                latent_dim=latent_embedding_dimension,
                hidden_dim=2 * latent_embedding_dimension,
                dropout=transformer_dropout
            )
        if self.use_auxiliary_locality_supervision:
            self.auxiliary_edge_head = EdgeMLP(
                latent_dim=latent_embedding_dimension,
                hidden_dim=2 * latent_embedding_dimension,
                dropout=transformer_dropout
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

    # -----------------------------------------------------------------------
    # Diffusion schedule helpers
    # -----------------------------------------------------------------------
    def _set_schedule_metadata(
        self,
        *,
        sigma_min: float,
        sigma_max: float,
        sampling_final_sigma: float,
    ) -> None:
        """Persist the diffusion schedule hyper-parameters and mirror legacy attributes."""
        sigma_min = float(sigma_min)
        sigma_max = float(sigma_max)
        sampling_final_sigma = float(sampling_final_sigma)

        if not sigma_min < sigma_max:
            raise ValueError(f"sigma_min must be < sigma_max (got {sigma_min} >= {sigma_max})")
        if sampling_final_sigma < 0:
            raise ValueError(f"sampling_final_sigma must be non-negative (got {sampling_final_sigma})")
        if sampling_final_sigma > sigma_max:
            raise ValueError(
                f"sampling_final_sigma must be <= sigma_max "
                f"(got {sampling_final_sigma} > {sigma_max})"
            )

        self._schedule_metadata = {
            "sigma_min": sigma_min,
            "sigma_max": sigma_max,
            "sampling_final_sigma": sampling_final_sigma,
        }
        # Maintain legacy attributes for backwards compatibility with training loops / checkpoints.
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sampling_final_sigma = sampling_final_sigma

    def _ensure_schedule_metadata(self) -> None:
        """Backfill missing schedule metadata so checkpoints without it remain usable."""
        meta = getattr(self, "_schedule_metadata", None)
        if not isinstance(meta, dict):
            meta = {}

        sigma_min = float(meta.get("sigma_min", getattr(self, "sigma_min", 0.1)))
        sigma_max = float(meta.get("sigma_max", getattr(self, "sigma_max", 1.0)))
        sampling_final_sigma = float(
            meta.get("sampling_final_sigma", getattr(self, "sampling_final_sigma", 0.0))
        )

        # Clamp gracefully instead of failing hard when loading legacy weights.
        if sigma_max <= sigma_min:
            sigma_max = max(sigma_min + 1e-6, 1.0)
        sampling_final_sigma = max(0.0, min(sampling_final_sigma, sigma_max))

        self._set_schedule_metadata(
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            sampling_final_sigma=sampling_final_sigma,
        )

    def _build_sigma_schedule(self, total_steps: int, device: torch.device) -> torch.Tensor:
        """Assemble the linear noise ladder used by both training and sampling loops."""
        self._ensure_schedule_metadata()
        if total_steps <= 1:
            return torch.tensor([self.sigma_max], device=device)
        return torch.linspace(self.sigma_max, self.sampling_final_sigma, total_steps, device=device)

    def forward(
        self,
        input_rows: torch.Tensor,
        global_condition_vector: torch.Tensor,
        diffusion_time_step: torch.Tensor,
        return_latents: bool = False,
        add_noise: bool = True,
    ):
        """Predict the denoising target for a batch of node features at time step `t`."""
        self._ensure_schedule_metadata()
        if add_noise:
            noisy_input, eps, sigma_t = self.apply_noise_schedule(input_rows, diffusion_time_step)
        else:
            noisy_input = input_rows
            eps = None
            sigma_t = None

        latent_tokens = self._encode_with_condition(
            noisy_input,
            global_condition_vector,
            diffusion_time_step,
        )

        # Predict ε for continuous features
        pred_eps = self.linear_decoder_latent_to_output(latent_tokens)

        # Other prediction heads remain the same
        logits_deg = self.degree_head(latent_tokens)
        logits_exist = self.exist_head(latent_tokens).squeeze(-1)  # (B,N)

        if return_latents:
            return pred_eps, logits_deg, logits_exist, latent_tokens, eps, sigma_t
        return pred_eps, logits_deg, logits_exist, eps, sigma_t


    def _sigma_from_t(self, t: torch.Tensor) -> torch.Tensor:
        """Compute the noise amplitude associated with a normalized diffusion time."""
        self._ensure_schedule_metadata()
        return self.sigma_min + t * (self.sigma_max - self.sigma_min)

    def _t_from_sigma(self, sigma: torch.Tensor) -> torch.Tensor:
        """Invert the schedule by turning a noise level back into its normalized time."""
        self._ensure_schedule_metadata()
        return ((sigma - self.sigma_min) / (self.sigma_max - self.sigma_min)).clamp(0.0, 1.0)

    def _build_noise_scale(self, x: torch.Tensor, sigma_t: torch.Tensor) -> torch.Tensor:
        """Construct feature-wise noise scales, reducing diffusion on the degree channel."""
        noise_scale = torch.ones_like(x) * sigma_t
        noise_scale[..., self.important_feature_index] /= self.noise_degree_factor
        return noise_scale

    def apply_noise_schedule(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perturb inputs with Gaussian noise sampled from the current schedule."""
        self._ensure_schedule_metadata()
        # base Gaussian noise
        eps = torch.randn_like(x)

        sigma_t = self._sigma_from_t(t).unsqueeze(-1)  # (B,1,1)
        noise_scale = self._build_noise_scale(x, sigma_t)

        x_t = x + eps * noise_scale
        return x_t, eps, sigma_t

    def _encode_with_condition(
        self,
        input_rows: torch.Tensor,
        global_condition_vector: torch.Tensor,
        diffusion_time_step: torch.Tensor,
    ) -> torch.Tensor:
        """Encode node tokens with time and condition context via shared transformer."""
        x_norm = self.layernorm_in(input_rows)
        latent_tokens = self.linear_encoder_input_to_latent(x_norm)

        time_token = get_sinusoidal_time_embedding(
            diffusion_time_step, self.latent_embedding_dimension
        ).unsqueeze(1)  # (B,1,D)

        if global_condition_vector.dim() == 2:
            cond_tokens = global_condition_vector.unsqueeze(1)  # (B,1,C)
        elif global_condition_vector.dim() == 3:
            cond_tokens = global_condition_vector
        else:
            raise ValueError(
                "global_condition_vector must have shape (B, C) or (B, M, C); "
                f"received {tuple(global_condition_vector.shape)}"
            )

        if self.pool_condition_tokens and cond_tokens.size(1) > 1:
            cond_tokens = cond_tokens.mean(dim=1, keepdim=True)

        weight = self.linear_encoder_condition_to_latent.weight.transpose(0, 1)  # (C,D)
        cond_proj = torch.matmul(cond_tokens, weight)
        bias = self.linear_encoder_condition_to_latent.bias
        if bias is not None:
            cond_proj = cond_proj + bias

        mem = torch.cat([time_token, cond_proj], dim=1)  # (B, 1+M, D)
        for layer in self.shared_transformer:
            latent_tokens = layer(latent_tokens, k=mem, v=mem)
        return latent_tokens

    # ---------------------------------------------------------------------------
    # single-source loss computation – returns all partials
    # ---------------------------------------------------------------------------
    def compute_weighted_loss(
        self,
        prediction: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        target: torch.Tensor,
        global_condition: torch.Tensor,
        diffusion_time_step: torch.Tensor,
    ) -> dict:
        """Return the composite objective for diffusion, existence, and degree heads."""
        pred_eps, eps, sigma_t = prediction

        # --- ε-prediction loss for continuous channels ---
        loss_eps = F.mse_loss(pred_eps, eps, reduction='mean')

        # --- Clean reconstruction used for auxiliary heads ---
        noise_scale = self._build_noise_scale(target, sigma_t)
        x_t = target + noise_scale * eps
        x_hat0 = x_t - noise_scale * pred_eps

        clean_time = torch.zeros_like(diffusion_time_step)
        clean_latent = self._encode_with_condition(
            x_hat0,
            global_condition,
            clean_time,
        )
        logits_deg_clean = self.degree_head(clean_latent)
        logits_exist_clean = self.exist_head(clean_latent).squeeze(-1)

        # --- Node existence head (evaluated on the clean reconstruction) ---
        target_exist = (target[..., 0] >= 0.5).float()
        loss_exist = F.binary_cross_entropy_with_logits(
            logits_exist_clean,
            target_exist,
            pos_weight=self.exist_pos_weight
        )

        # --- Degree classification head (evaluated on the clean reconstruction) ---
        target_degree_scaled = target[..., self.important_feature_index]
        deg_unscaled = target_degree_scaled * self.deg_range_val + self.deg_min_val
        true_deg_class = torch.clamp(torch.round(deg_unscaled), 0, self.max_degree).long()
        loss_deg_ce = F.cross_entropy(
            logits_deg_clean.reshape(-1, self.max_degree + 1),
            true_deg_class.reshape(-1)
        )

        # total combined loss
        total_loss = (
            loss_eps
            + self.lambda_node_exist_importance * loss_exist
            + self.lambda_degree_importance * loss_deg_ce
        )

        return {
            "total": total_loss,
            "eps": loss_eps,
            "exist": loss_exist,
            "deg_ce": loss_deg_ce
        }

    # ---------------------------------------------------------------------------


    # ---------------------------------------------------------------------------
    # TRAINING STEP – uses the dict
    # ---------------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        """Run one optimisation step, optionally including locality supervision."""
        if self.use_locality_supervision:
            input_examples, global_condition, edge_idx, edge_labels, aux_edge_idx, aux_edge_labels, node_mask = batch
        else:
            input_examples, global_condition = batch
            aux_edge_idx = torch.empty((0, 3), dtype=torch.long, device=input_examples.device)
            aux_edge_labels = torch.empty((0,), dtype=torch.float32, device=input_examples.device)

        diffusion_time_step = torch.rand(input_examples.size(0), 1, device=self.device)

        if self.use_locality_supervision:
            pred_eps, _, _, latent_tokens, eps, sigma_t = \
                self.forward(input_examples, global_condition, diffusion_time_step, return_latents=True)
        else:
            pred_eps, _, _, eps, sigma_t = \
                self.forward(input_examples, global_condition, diffusion_time_step)

        # Core diffusion + degree + existence losses
        losses = self.compute_weighted_loss(
            (pred_eps, eps, sigma_t),
            input_examples,
            global_condition,
            diffusion_time_step,
        )
        total_loss = losses["total"]

        # ───────────────────────────────
        # Locality supervision (auxiliary MLP head)
        # ───────────────────────────────
        if self.use_locality_supervision and edge_idx.numel() > 0:
            b, i, j = edge_idx.unbind(1)  # each (E,)
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

        # ───────────────────────────────
        # Log and return
        # ───────────────────────────────
        self.log("train_total", total_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_recon", losses["eps"], on_step=False, on_epoch=True)  # reconstruction-only loss
        self.log("train_deg_ce", losses["deg_ce"], on_step=False, on_epoch=True)
        self.log("train_exist", losses["exist"], on_step=False, on_epoch=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        """Evaluate a batch without gradient updates, mirroring the training step logic."""
        if self.use_locality_supervision:
            input_examples, global_condition, edge_idx, edge_labels, aux_edge_idx, aux_edge_labels, node_mask = batch
        else:
            input_examples, global_condition = batch
            aux_edge_idx = torch.empty((0, 3), dtype=torch.long, device=input_examples.device)
            aux_edge_labels = torch.empty((0,), dtype=torch.float32, device=input_examples.device)

        diffusion_time_step = torch.rand(input_examples.size(0), 1, device=self.device)

        if self.use_locality_supervision:
            pred_eps, _, _, latent_tokens, eps, sigma_t = \
                self.forward(input_examples, global_condition, diffusion_time_step, return_latents=True)
        else:
            pred_eps, _, _, eps, sigma_t = \
                self.forward(input_examples, global_condition, diffusion_time_step)

        losses = self.compute_weighted_loss(
            (pred_eps, eps, sigma_t),
            input_examples,
            global_condition,
            diffusion_time_step,
        )
        total_loss = losses["total"]

        # ───────────────────────────────
        # Locality supervision (auxiliary MLP head)
        # ───────────────────────────────
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

            self.log("val_edge_ce", edge_loss, on_step=False, on_epoch=True)
            self.log("val_edge_acc", edge_acc, on_step=False, on_epoch=True)

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

            self.log("val_aux_locality_ce", aux_edge_loss, on_step=False, on_epoch=True)
            self.log("val_aux_edge_acc", aux_edge_acc, on_step=False, on_epoch=True)

        # ───────────────────────────────
        # Log and return
        # ───────────────────────────────
        self.log("val_total", total_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_recon", losses["eps"], on_step=False, on_epoch=True)  # reconstruction-only loss
        self.log("val_deg_ce", losses["deg_ce"], on_step=False, on_epoch=True)
        self.log("val_exist", losses["exist"], on_step=False, on_epoch=True)
        return total_loss


    def on_train_end(self):
        if not self.verbose:
            return
        plot_metrics(
            train_metrics={
                "total": self.train_losses,
                "deg_ce": self.train_deg_ce,
                "recon": self.train_recon,   # updated
                "exist": self.train_exist,
                **({"edge_ce": self.train_edge_loss} if self.use_locality_supervision else {}),
                **({"aux_locality": self.train_aux_edge_loss} if self.use_auxiliary_locality_supervision else {}),
            },
            val_metrics={
                "total": self.val_losses,
                "deg_ce": self.val_deg_ce,
                "recon": self.val_recon,     # updated
                "exist": self.val_exist,
                **({"edge_ce": self.val_edge_loss} if self.use_locality_supervision else {}),
                **({"aux_locality": self.val_aux_edge_loss} if self.use_auxiliary_locality_supervision else {}),
            },
            window=10,
            alpha=0.1
        )
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    
    # ───────────────────────────────────────────────────────────────────
    #  Classifier-guidance utilities  
    # ───────────────────────────────────────────────────────────────────
    def set_guidance_classifier(self, num_classes: int) -> None:
        """Initialise the lightweight guidance head that scores latent representations."""
        self.guidance_classifier = GuidanceMLP(
            input_dim=self.latent_embedding_dimension,
            hidden_dim=2 * self.latent_embedding_dimension,
            output_dim=num_classes
        ).to(self.device)

    
    def train_guidance_classifier(
        self,
        node_feats: List[np.ndarray],
        cond_vecs: np.ndarray,
        labels: np.ndarray,
        epochs: int = 20,
        lr: float = 1e-3,
        verbose: bool = True
    ):
        """Fit the guidance head on pooled latents and report its learning curve."""
        if self.guidance_classifier is None:
            raise RuntimeError("call set_guidance_classifier() first")

        self.eval()
        self.guidance_classifier.train()
        opt = torch.optim.Adam(self.guidance_classifier.parameters(), lr=lr)

        max_rows = self.number_of_rows_per_example
        padded_feats = []
        for f in node_feats:
            if f.shape[0] < max_rows:
                f = np.pad(f, ((0, max_rows - f.shape[0]), (0, 0)))
            padded_feats.append(f[:max_rows])

        X = torch.tensor(np.stack(padded_feats), dtype=torch.float32)
        cond_array = np.asarray(cond_vecs)
        if cond_array.ndim == 1:
            cond_array = cond_array[:, None]
        Y = torch.tensor(cond_array, dtype=torch.float32)
        L = torch.tensor(labels, dtype=torch.long)

        # --- Split into train/val ---
        X_tr, X_val, Y_tr, Y_val, L_tr, L_val = train_test_split(
            X.numpy(), Y.numpy(), L.numpy(), test_size=0.2, random_state=42
        )
        X_tr = torch.tensor(X_tr, device=self.device)
        Y_tr = torch.tensor(Y_tr, device=self.device)
        L_tr = torch.tensor(L_tr, device=self.device)
        X_val = torch.tensor(X_val, device=self.device)
        Y_val = torch.tensor(Y_val, device=self.device)
        L_val = torch.tensor(L_val, device=self.device)

        T_tr = torch.zeros(X_tr.size(0), 1, device=self.device)
        T_val = torch.zeros(X_val.size(0), 1, device=self.device)

        train_losses = []
        val_losses = []

        for _ in range(epochs):
            _, _, _, lat = self.forward(X_tr, Y_tr, T_tr, return_latents=True)
            logits = self.guidance_classifier(lat.mean(dim=1))
            loss = F.cross_entropy(logits, L_tr)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            train_losses.append(loss.item())

            with torch.no_grad():
                _, _, _, lat_val = self.forward(X_val, Y_val, T_val, return_latents=True)
                logits_val = self.guidance_classifier(lat_val.mean(dim=1))
                loss_val = F.cross_entropy(logits_val, L_val)
                val_losses.append(loss_val.item())

        if verbose:
            print(f"Trained guidance classifier for {epochs} epochs with learning rate {lr}.")
            print(f"Final train loss: {train_losses[-1]:.4f}, val loss: {val_losses[-1]:.4f}")
            # --- Plot losses ---
            plt.figure(figsize=(8, 5))
            plt.plot(train_losses, label="Train Loss")
            plt.plot(val_losses, label="Val Loss")
            plt.yscale('log')
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.title("Guidance Classifier Loss")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

    def generate(
        self,
        global_condition: torch.Tensor,
        total_steps: int = 200,
        desired_class: Optional[Union[int, Sequence[int]]] = None,
        use_heads_projection: bool = False,   # NEW: use exist/deg heads to “snap” outputs
        exist_threshold: float = 0.5,        # threshold in prob space
    ) -> torch.Tensor:
        """Run deterministic sampling that mirrors the training diffusion schedule."""
        self.eval()
        self._ensure_schedule_metadata()
        B = global_condition.size(0)
        device = global_condition.device
        self._last_edge_probability_matrices = None

        # --- sigma schedule consistent with training metadata ---
        sigmas = self._build_sigma_schedule(total_steps=total_steps, device=device)  # (T,)

        # Start from pure noise at the largest sigma
        x = torch.randn(
            B, self.number_of_rows_per_example, self.input_feature_dimension, device=device
        )

        # autocast for speed on GPU (skip on unsupported backends)
        if device.type == "cuda":
            autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16)
        elif device.type == "mps":
            autocast_ctx = torch.autocast(device_type="mps", dtype=torch.float16)
        else:
            autocast_ctx = contextlib.nullcontext()

        with autocast_ctx:
            for i in range(total_steps - 1):
                sigma_t = sigmas[i]
                sigma_next = sigmas[i + 1]
                # Map sigma_t back to training's normalized time t∈[0,1] for the time embedding
                t_norm = self._t_from_sigma(sigma_t).expand(B, 1)

                # Predict eps
                pred_eps, _, _, _, _ = self.forward(
                    x, global_condition, t_norm, add_noise=False
                )

                # Consistent additive update: x_{next} = x_t - (sigma_t - sigma_{next}) * epŝ
                x = (x - (sigma_t - sigma_next) * pred_eps).detach()

                # Optional classifier guidance (same as before, using t_norm)
                if self.use_guidance and self.guidance_classifier is not None:
                    x.requires_grad_(True)
                    _, _, _, lat, _, _ = self.forward(
                        x, global_condition, t_norm, return_latents=True, add_noise=False
                    )
                    pooled = lat.mean(dim=1)
                    logits_cls = self.guidance_classifier(pooled)
                    if desired_class is not None:
                        if isinstance(desired_class, int):
                            tgt = torch.full_like(logits_cls[:, 0], desired_class, dtype=torch.long)
                        else:
                            tgt = torch.as_tensor(desired_class, device=x.device)
                        sel = logits_cls[torch.arange(tgt.numel()), tgt]
                    else:
                        sel = logits_cls.softmax(-1).max(dim=-1).values
                    grad = torch.autograd.grad(sel.sum(), x)[0]
                    x = (x - self.guidance_weight * grad).detach()

        # Optional: project existence/degree using the auxiliary heads once at (approx) t=0
        if use_heads_projection:
            sigma_proj = sigmas[-1]
            t0 = self._t_from_sigma(sigma_proj).expand(B, 1)
            with torch.no_grad():
                _, logits_deg, logits_exist, latent_tokens, _, _ = self.forward(
                    x, global_condition, t0, return_latents=True, add_noise=False
                )

            # Existence: overwrite channel 0 with {0,1} from head
            exist_probs = torch.sigmoid(logits_exist)  # (B, N)
            exist_bin = (exist_probs >= exist_threshold).float()
            x[..., 0] = exist_bin

            # Degree: overwrite degree channel with predicted class index (in *original* scale after inverse)
            deg_classes = torch.argmax(logits_deg, dim=-1)  # (B, N)

            # Store degree classes so the wrapper can apply them after inverse transform if desired.
            self._last_deg_classes = deg_classes.detach().cpu()
            if self.use_locality_supervision:
                edge_probs = self._compute_edge_probability_matrices(latent_tokens)
                self._last_edge_probability_matrices = edge_probs.detach().cpu()

        return x

# =============================================================================
# Revised TransformerConditionalDiffusionGenerator with Piecewise Scheduling Parameters
# =============================================================================
class GraphWithEdgesDataset(Dataset):
    """Tensor dataset that stores graph node features, conditions, and edge labels.

    Args:
        X (np.ndarray): Node feature tensor shaped `(B, N, D)` where `B` is batch size,
            `N` the node count, and `D` the feature dimension.
        Y (np.ndarray): Conditioning feature tensor shaped `(B, C)` or `(B, N, C)`
            describing global context per graph.
        edge_pairs (List[Tuple[int, int, int]]): Triplets of `(batch_index, src, dst)`
            identifying locality-labelled node pairs.
        edge_targets (np.ndarray): Locality labels aligned with `edge_pairs`.
        node_mask (Optional[np.ndarray]): Boolean mask of shape `(B, N)` that flags
            which node slots are valid for each graph.
    """
    def __init__(
        self,
        X: np.ndarray,                    # (B, N, D)
        Y: np.ndarray,                    # (B, C)
        edge_pairs: List[Tuple[int, int, int]],
        edge_targets: np.ndarray,
        auxiliary_edge_pairs: Optional[List[Tuple[int, int, int]]] = None,
        auxiliary_edge_targets: Optional[np.ndarray] = None,
        node_mask: Optional[np.ndarray] = None   # (B, N) boolean
    ):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
        B, N, _ = X.shape
        if node_mask is None:
            node_mask = np.ones((B, N), dtype=bool)
        self.node_mask = torch.tensor(node_mask, dtype=torch.bool)
        self.edge_idx_by_graph = {b: [] for b in range(len(X))}
        self.edge_lbl_by_graph = {b: [] for b in range(len(X))}
        for (b, i, j), lbl in zip(edge_pairs, edge_targets):
            self.edge_idx_by_graph[b].append((i, j))
            self.edge_lbl_by_graph[b].append(lbl)
        self.aux_edge_idx_by_graph = {b: [] for b in range(len(X))}
        self.aux_edge_lbl_by_graph = {b: [] for b in range(len(X))}
        if auxiliary_edge_pairs is not None and auxiliary_edge_targets is not None:
            for (b, i, j), lbl in zip(auxiliary_edge_pairs, auxiliary_edge_targets):
                self.aux_edge_idx_by_graph[b].append((i, j))
                self.aux_edge_lbl_by_graph[b].append(lbl)

    def __len__(self) -> int:
        """Return the number of graph instances bundled in the dataset."""
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return node features, condition vector, locality pairs, and mask for one graph."""
        x = self.X[idx]  # (N, D)
        y = self.Y[idx]  # (C,)
        mask = self.node_mask[idx]  # (N,)
        
        # Get edges and labels for this graph
        edge_idxs = torch.tensor(self.edge_idx_by_graph[idx], dtype=torch.long) if self.edge_idx_by_graph[idx] else torch.empty((0, 2), dtype=torch.long)
        edge_lbls = torch.tensor(self.edge_lbl_by_graph[idx], dtype=torch.float32) if self.edge_lbl_by_graph[idx] else torch.empty((0,), dtype=torch.float32)
        aux_edge_idxs = torch.tensor(self.aux_edge_idx_by_graph[idx], dtype=torch.long) if self.aux_edge_idx_by_graph[idx] else torch.empty((0, 2), dtype=torch.long)
        aux_edge_lbls = torch.tensor(self.aux_edge_lbl_by_graph[idx], dtype=torch.float32) if self.aux_edge_lbl_by_graph[idx] else torch.empty((0,), dtype=torch.float32)

        return x, y, edge_idxs, edge_lbls, aux_edge_idxs, aux_edge_lbls, mask

def collate_graph_with_edges(batch):
    """Batch GraphWithEdgesDataset samples into tensors ready for Lightning loaders.

    Args:
        batch: Sequence of dataset items returned by `GraphWithEdgesDataset.__getitem__`.

    Returns:
        Tuple[torch.Tensor, ...]: Batched node features, conditions, edge indices,
            edge labels, and node masks.
    """
    xs, ys, masks = [], [], []
    local_edge_idxs, local_edge_lbls = [], []
    aux_local_edge_idxs, aux_local_edge_lbls = [], []
    for x, y, ei, el, aux_ei, aux_el, mask in batch:
        xs.append(x)
        ys.append(y)
        masks.append(mask)
        local_edge_idxs.append(ei)
        local_edge_lbls.append(el)
        aux_local_edge_idxs.append(aux_ei)
        aux_local_edge_lbls.append(aux_el)
    X = torch.stack(xs)         # (B, N, D)
    Y = torch.stack(ys)         # (B, C)
    M = torch.stack(masks)      # (B, N)
    def _pack_pairs(per_graph_idxs, per_graph_lbls):
        all_edge_idxs = []
        all_edge_lbls = []
        for b, (ei, el) in enumerate(zip(per_graph_idxs, per_graph_lbls)):
            if ei.numel() == 0:
                continue
            b_col = torch.full((ei.size(0), 1), b, dtype=torch.long)
            global_idx = torch.cat([b_col, ei], dim=1)
            all_edge_idxs.append(global_idx)
            all_edge_lbls.append(el)
        if all_edge_idxs:
            return torch.cat(all_edge_idxs, dim=0), torch.cat(all_edge_lbls, dim=0)
        return torch.empty((0, 3), dtype=torch.long), torch.empty((0,), dtype=torch.float32)

    edge_idx, edge_lbl = _pack_pairs(local_edge_idxs, local_edge_lbls)
    aux_edge_idx, aux_edge_lbl = _pack_pairs(aux_local_edge_idxs, aux_local_edge_lbls)
    return X, Y, edge_idx, edge_lbl, aux_edge_idx, aux_edge_lbl, M

class ConditionalNodeGenerator:
    """Scikit-learn friendly façade around the conditional diffusion pipeline.

    Args:
        latent_embedding_dimension (int): Width of the latent representation processed
            by the transformer backbone.
        number_of_transformer_layers (int): Number of cross-attention transformer
            layers used inside the Lightning module.
        transformer_attention_head_count (int): Number of attention heads allocated to
            each transformer layer.
        transformer_dropout (float): Dropout probability applied inside attention and
            feed-forward sublayers.
        learning_rate (float): Optimiser step size used when fitting the Lightning
            module.
        maximum_epochs (int): Maximum number of training epochs executed by the
            Lightning trainer.
        batch_size (int): Batch size for node graphs during training and evaluation.
        total_steps (int): Number of diffusion time steps sampled when training or
            drawing new graphs.
        verbose (bool): Whether to produce additional logs, plots, and progress output.
        important_feature_index (int): Column index of the degree feature in the input
            tensors; drives specialised scaling and head losses.
        lambda_degree_importance (float): Weight applied to the auxiliary degree loss
            during optimisation.
        noise_degree_factor (float): Divider that reduces injected noise on the degree
            channel relative to other features.
        degree_temperature (Optional[float]): Optional softmax temperature applied to
            degree logits before computing losses.
        lambda_node_exist_importance (float): Weight assigned to the node existence
            classification loss.
        default_exist_pos_weight (float): Positive class weight injected into the
            existence loss when no dataset-specific weights are supplied.
        lambda_locality_importance (float): Scaling factor for the optional locality
            edge supervision loss.
        use_guidance (bool): Train an auxiliary classifier on latent representations to
            enable classifier guidance during sampling.
        pool_condition_tokens (bool): Average condition tokens into a single vector
            before cross-attention instead of keeping them separate.
        use_locality_supervision (bool): Enable supervision for locality edge scores
            when training data provides labelled node pairs.
    """
    def __init__(self,
                 latent_embedding_dimension: int = 128,
                 number_of_transformer_layers: int = 4,
                 transformer_attention_head_count: int = 4,
                 transformer_dropout: float = 0.1,
                 learning_rate: float = 1e-3,
                 maximum_epochs: int = 10,
                 batch_size: int = 32,
                 total_steps: int = 1000,
                 verbose: bool = False,
                 important_feature_index: int = 1,
                 lambda_degree_importance: float = 1.0,
                 noise_degree_factor: float = 2.0,
                 degree_temperature: Optional[float] = None,
                 lambda_node_exist_importance: float = 1.0,
                 default_exist_pos_weight: float = 1.0,
                 lambda_locality_importance: float = 1.0,
                 lambda_auxiliary_locality_importance: float = 1.0,
                 use_guidance: bool = False,
                 pool_condition_tokens: bool = False,
                 use_locality_supervision: bool = False
    ):
        self.latent_embedding_dimension = latent_embedding_dimension
        self.number_of_transformer_layers = number_of_transformer_layers
        self.transformer_attention_head_count = transformer_attention_head_count
        self.transformer_dropout = transformer_dropout
        self.learning_rate = learning_rate
        self.maximum_epochs = maximum_epochs
        self.batch_size = batch_size
        self.total_steps = total_steps
        self.verbose = verbose
        self.important_feature_index = important_feature_index
        self.lambda_degree_importance = lambda_degree_importance
        self.noise_degree_factor = noise_degree_factor
        self.degree_temperature = degree_temperature
        self.lambda_node_exist_importance = lambda_node_exist_importance
        self.default_exist_pos_weight = default_exist_pos_weight
        self.lambda_locality_importance = lambda_locality_importance
        self.lambda_auxiliary_locality_importance = lambda_auxiliary_locality_importance
        self.use_guidance = use_guidance
        self.pool_condition_tokens = bool(pool_condition_tokens)
        self.use_locality_supervision = bool(use_locality_supervision)

        self.number_of_rows_per_example = None
        self.input_feature_dimension = None
        self.model = None
        # self.conditional_generator_estimator = None # This was not used
        self.x_scaler = None # Scaler for node features
        self.y_scaler = None # Scaler for conditional features
        self.D_max = None
        self.condition_token_count = 1
        self.condition_feature_dimension = None
        self.last_predicted_edge_probability_matrices_ = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # <-- FIXED

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
        auxiliary_edge_pairs: Optional[List[Tuple[int, int, int]]],
        auxiliary_edge_targets: Optional[np.ndarray],
    ) -> Tuple[bool, bool]:
        """Combine the supervision plan with the actually supplied arrays."""
        planned_direct_edges = self._planned_enabled("direct_edges", self.use_locality_supervision)
        planned_aux_locality = self._planned_enabled("auxiliary_locality", False)
        effective_locality = planned_direct_edges and edge_pairs is not None and edge_targets is not None
        effective_auxiliary_locality = (
            planned_aux_locality and auxiliary_edge_pairs is not None and auxiliary_edge_targets is not None
        )
        return effective_locality, effective_auxiliary_locality

    def _fit_scalers(self, X_array, y_array):
        B, n, d = X_array.shape
        # X_reshaped = X_array.reshape(-1, d) # No longer needed for CustomRobustScaler

        # For MinMaxScaler, we'll fit X and Y separately
        self.x_scaler = MinMaxScaler()
        self.y_scaler = MinMaxScaler()

        # Fit scaler for X (node features)
        # X_array has shape (B, N, D_x), reshape to (B*N, D_x) for scaler
        num_features_x = X_array.shape[2]
        self.x_scaler.fit(X_array.reshape(-1, num_features_x))

        # Fit scaler for Y (conditional features)
        if y_array.ndim == 3:
            self.y_scaler.fit(y_array.reshape(-1, y_array.shape[-1]))
        else:
            self.y_scaler.fit(y_array)

        # self.raw_degree_index is already set by __init__ (defaults to 1)
        # With MinMaxScaler, the important_feature_index (for scaled data)
        # will be the same as raw_degree_index, as MinMaxScaler doesn't change column order.

    def _transform_data(self, X_array, y_array):
        B, N, D_x = X_array.shape
        X_scaled_flat = self.x_scaler.transform(X_array.reshape(-1, D_x))
        X_scaled = X_scaled_flat.reshape(B, N, D_x)
        
        if y_array.ndim == 3:
            Bc, M, Dy = y_array.shape
            y_scaled = self.y_scaler.transform(y_array.reshape(-1, Dy)).reshape(Bc, M, Dy)
        else:
            y_scaled = self.y_scaler.transform(y_array)
        return X_scaled, y_scaled

    def _inverse_transform_input(self, X_array):
        B, N, D_x = X_array.shape # D_x is dimension of scaled features, same as original with MinMaxScaler
        X_orig_flat = self.x_scaler.inverse_transform(X_array.reshape(-1, D_x))
        X_orig = X_orig_flat.reshape(B, N, D_x) # Reshape back to (B, N, D_x_original)
        X_orig[..., self.important_feature_index] = np.clip(
            X_orig[..., self.important_feature_index], 0, self.D_max
        )
        return X_orig

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
        node_mask: Optional[np.ndarray] = None
    ):
        """Prepare scalers, metadata, and the underlying Lightning module for training."""
        del edge_label_pairs, edge_label_targets
        planned_direct_edges = self._planned_enabled("direct_edges", self.use_locality_supervision)
        planned_aux_locality = self._planned_enabled("auxiliary_locality", False)
        planned_edge_labels = self._planned_enabled(
            "edge_labels",
            edge_label_pairs is not None and edge_label_targets is not None,
        )
        effective_locality, effective_auxiliary_locality = self._effective_supervision_flags(
            edge_pairs,
            edge_targets,
            auxiliary_edge_pairs,
            auxiliary_edge_targets,
        )
        if planned_direct_edges and not effective_locality and self.verbose:
            print("Direct-edge channel planned as learned, but usable edge_pairs/edge_targets were not supplied.")
        if planned_aux_locality and not effective_auxiliary_locality and self.verbose:
            print("Auxiliary-locality channel planned as learned, but usable auxiliary_edge_pairs/auxiliary_edge_targets were not supplied.")
        if planned_edge_labels and self.verbose:
            print("Edge-label channel is planned, but ConditionalNodeGenerator does not implement edge-label prediction in this backend.")
        if not effective_locality:
            edge_pairs = None
            edge_targets = None
        if not effective_auxiliary_locality:
            auxiliary_edge_pairs = None
            auxiliary_edge_targets = None

        max_num_rows = max(x.shape[0] for x in node_encodings_list)
        self.number_of_rows_per_example = max_num_rows
        X_padded = []
        for x in node_encodings_list:
            n_rows = x.shape[0]
            if n_rows < max_num_rows:
                pad_width = ((0, max_num_rows - n_rows), (0, 0))
                x = np.pad(x, pad_width=pad_width, mode='constant', constant_values=0)
            X_padded.append(x)
        X_array = np.stack(X_padded, axis=0)
        y_array = np.array(conditional_graph_encodings)
        if y_array.ndim == 1:
            y_array = y_array[:, None]
        if y_array.ndim == 2:
            self.condition_token_count = 1
            self.condition_feature_dimension = y_array.shape[1]
        elif y_array.ndim == 3:
            self.condition_token_count = y_array.shape[1]
            self.condition_feature_dimension = y_array.shape[2]
        else:
            raise ValueError(
                "conditional_graph_encodings must be array-like of shape (B, C) or (B, M, C); "
                f"received shape {y_array.shape}"
            )
        
        self._fit_scalers(X_array, y_array)

        # ------------------------------------------------------------
        # Compute class-imbalance weight for BCEWithLogitsLoss
        # ------------------------------------------------------------
        exist_mask = (X_array[..., 0] >= 0.5)
        ones  = int(exist_mask.sum())                 # rows where exist == 1
        zeros = int(exist_mask.size) - ones           # rows where exist == 0

        # BCEWithLogitsLoss multiplies the *positive* (1-class) loss by
        # pos_weight.  It should be >1 **only when 1's are the minority**.
        if ones == 0:
            exist_pos_weight = 1.0                    # avoid div-by-zero
        elif zeros > ones:                            # positives rarer
            exist_pos_weight = float(zeros) / float(ones)
        else:                                         # positives majority or equal
            exist_pos_weight = 1.0

        # Get degree scaling parameters (min and range) from x_scaler for the degree column
        # self.important_feature_index is the column index for degree in the raw X_array
        deg_column_data_for_minmax = X_array[..., self.important_feature_index].reshape(-1, 1)
        temp_degree_scaler = MinMaxScaler().fit(deg_column_data_for_minmax) # Fit only on the degree column
        
        deg_min_val = temp_degree_scaler.data_min_[0]
        deg_range_val = temp_degree_scaler.data_range_[0]
        if deg_range_val == 0: # Handle case where all degrees are the same
            deg_range_val = 1e-8 # Avoid division by zero, effectively makes scaled value 0 if val == min
        
        X_scaled, y_scaled = self._transform_data(X_array, y_array)
        self.input_feature_dimension = X_scaled.shape[2]
        cond_feature_dim = self.condition_feature_dimension
        
        # Detect maximum degree from raw data
        raw_degrees = X_array[..., self.important_feature_index]  # shape (B, N)
        self.D_max = int(raw_degrees.max())  # global max
        
        # Initialize the model with updated flags for locality supervision
        self.model = IterativeDenoisingAutoencoderTransformerModel(
            number_of_rows_per_example=self.number_of_rows_per_example,
            input_feature_dimension=self.input_feature_dimension,
            condition_feature_dimension=cond_feature_dim,
            latent_embedding_dimension=self.latent_embedding_dimension,
            number_of_transformer_layers=self.number_of_transformer_layers,
            transformer_attention_head_count=self.transformer_attention_head_count,
            transformer_dropout=self.transformer_dropout,
            learning_rate=self.learning_rate,
            verbose=self.verbose,
            important_feature_index=self.important_feature_index,
            max_degree=self.D_max,
            lambda_degree_importance=self.lambda_degree_importance,
            noise_degree_factor=self.noise_degree_factor,
            degree_temperature=self.degree_temperature,
            degree_min_val=deg_min_val,
            degree_range_val=deg_range_val,
            lambda_node_exist_importance=self.lambda_node_exist_importance,
            use_locality_supervision=effective_locality,
            lambda_locality_importance=self.lambda_locality_importance,
            use_auxiliary_locality_supervision=effective_auxiliary_locality,
            lambda_auxiliary_locality_importance=self.lambda_auxiliary_locality_importance,
            exist_pos_weight=exist_pos_weight,
            use_guidance=self.use_guidance,      # NEW
            guidance_weight=1.0,                 # tweak as needed
            pool_condition_tokens=self.pool_condition_tokens,
        )
        self.model.use_guidance = self.use_guidance
        self.model.use_locality_supervision = effective_locality
        self.model.use_auxiliary_locality_supervision = effective_auxiliary_locality

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
        node_mask: Optional[np.ndarray] = None
    ):
        """Train the diffusion model, optionally consuming locality supervision pairs."""
        del edge_label_pairs, edge_label_targets
        X_array = np.stack([np.pad(x, ((0, self.number_of_rows_per_example - x.shape[0]), (0, 0)), mode='constant', constant_values=0)
                           for x in node_encodings_list], axis=0)
        y_array = np.array(conditional_graph_encodings)
        if y_array.ndim == 1:
            y_array = y_array[:, None]
        X_scaled, y_scaled = self._transform_data(X_array, y_array)

        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y_scaled, dtype=torch.float32)
        effective_locality, effective_auxiliary_locality = self._effective_supervision_flags(
            edge_pairs,
            edge_targets,
            auxiliary_edge_pairs,
            auxiliary_edge_targets,
        )
        if not effective_locality:
            edge_pairs = None
            edge_targets = None

        if effective_locality:
            if node_mask is None:
                B, N, _ = X_scaled.shape
                node_mask_arr = np.ones((B, N), dtype=bool)
            else:
                node_mask_arr = node_mask

            dataset = GraphWithEdgesDataset(
                X_scaled,
                y_scaled,
                edge_pairs,
                edge_targets,
                auxiliary_edge_pairs=auxiliary_edge_pairs,
                auxiliary_edge_targets=auxiliary_edge_targets,
                node_mask=node_mask_arr,
            )
            
            # Split into train/val
            dataset_size = len(node_encodings_list)  # <--- Use the length of the original data
            train_size = int(0.9 * dataset_size)
            val_size = dataset_size - train_size
            # Create indices for the split
            indices = torch.randperm(len(node_encodings_list)).tolist()
            train_indices = indices[:train_size]
            val_indices = indices[train_size:]

            train_dataset = torch.utils.data.Subset(dataset, train_indices)
            val_dataset = torch.utils.data.Subset(dataset, val_indices)
            
            train_loader = DataLoader(
                train_dataset, 
                batch_size=self.batch_size, 
                shuffle=True,
                collate_fn=collate_graph_with_edges
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,  # No need to shuffle validation
                collate_fn=collate_graph_with_edges
            )
        else:
            dataset = TensorDataset(X_tensor, y_tensor)
            
            # Split into train/val
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
            enable_progress_bar=False
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
        desired_class: Optional[Union[int, Sequence[int]]] = None
    ) -> List[np.ndarray]:
        """Sample node feature grids for each conditioning vector supplied."""
        if int(self.verbose) >= 2:
            print(f"Predicting node matrices for {len(conditional_graph_encodings)} graphs...")
            
        self.device = next(self.model.parameters()).device

        # ------------------------------------------------------------------
        # 1. Generate denoised node embeddings (scaled feature space)
        # ------------------------------------------------------------------
        with torch.no_grad():
            cond_array = np.asarray(conditional_graph_encodings)
            if cond_array.ndim == 1:
                cond_array = cond_array[:, None]
            cond_tensor = torch.tensor(
                cond_array, dtype=torch.float32, device=self.device
            )
            generated = self.model.generate(
                cond_tensor,
                total_steps=self.total_steps,
                desired_class=desired_class
            )

        # ------------------------------------------------------------------
        # 2. Convert to numpy and inverse-transform to original scale
        # ------------------------------------------------------------------
        gen_np = generated.detach().cpu().numpy()
        gen_orig = self._inverse_transform_input(gen_np)

        # ------------------------------------------------------------------
        # 3. Optional overwrite of existence / degree channels using heads
        # ------------------------------------------------------------------
        try:
            # Fetch stored head outputs (set inside generate)
            deg_classes = getattr(self.model, "_last_deg_classes", None)
            edge_probability_matrices = getattr(self.model, "_last_edge_probability_matrices", None)

            # Existence logits were already used inside generate (projected before inverse)
            # But if you prefer to re-compute existence probabilities here, uncomment below:
            # with torch.no_grad():
            #     t0 = torch.zeros(cond_tensor.size(0), 1, device=self.device)
            #     _, logits_deg, logits_exist, _, _, _ = self.model.forward(generated, cond_tensor, t0, return_latents=True)
            #     exist_probs = torch.sigmoid(logits_exist).cpu().numpy()
            #     exist_bin = (exist_probs >= 0.5).astype(float)
            #     for i in range(len(gen_orig)):
            #         gen_orig[i][..., 0] = exist_bin[i]

            # If degree logits were stored during generation, overwrite degree channel
            if deg_classes is not None:
                deg_classes = deg_classes.cpu().numpy()
                for i in range(len(gen_orig)):
                    # Overwrite the degree channel (assumed channel index 1)
                    gen_orig[i][..., 1] = np.clip(deg_classes[i], 0, None)

            self.last_predicted_edge_probability_matrices_ = None
            if edge_probability_matrices is not None:
                edge_probability_matrices = edge_probability_matrices.cpu().numpy()
                self.last_predicted_edge_probability_matrices_ = [
                    edge_probability_matrices[i]
                    for i in range(edge_probability_matrices.shape[0])
                ]

            if self.verbose:
                print("Applied head-based projection for existence/degree channels.")

        except Exception as e:
            if self.verbose:
                print(f"[Warning] Head projection skipped due to: {e}")

        return gen_orig

    def plot_metrics(self, window: int = 10, alpha: float = 0.3):
        if self.model is None:
            print("Model is not fitted yet.")
            return
        plot_metrics(
            train_metrics = {
                "total": self.model.train_losses,
                "deg_ce": self.model.train_deg_ce,
                "all": self.model.train_loss_all,
                "exist": self.model.train_exist,
                **({"edge_ce": self.model.train_edge_loss} if self.model.use_locality_supervision else {}),
                **({"aux_locality": self.model.train_aux_edge_loss} if self.model.use_auxiliary_locality_supervision else {}),
            },
            val_metrics = {
                "total": self.model.val_losses,
                "deg_ce": self.model.val_deg_ce,
                "all": self.model.val_loss_all,
                "exist": self.model.val_exist,
                **({"edge_ce": self.model.val_edge_loss} if self.model.use_locality_supervision else {}),
                **({"aux_locality": self.model.val_aux_edge_loss} if self.model.use_auxiliary_locality_supervision else {}),
            },
            window=window,
            alpha=alpha
        )
    
    # -------------- Guidance helpers --------------
    def set_guidance_classifier(self, num_classes: int):
        if self.model is None:
            raise RuntimeError("call setup() first")
        self.model.set_guidance_classifier(num_classes)

    def train_guidance_classifier(
        self, node_feats, cond_vecs, labels, *, epochs: int = 20, lr: float = 1e-3
    ):
        if self.model is None:
            raise RuntimeError("call setup() first")
        self.model.train_guidance_classifier(node_feats, cond_vecs, labels,
                                             epochs=epochs, lr=lr)

class MetricsLogger(pl.callbacks.Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        m = trainer.callback_metrics
        pl_module.train_losses.append(m.get("train_total", torch.tensor(0.0)).item())
        pl_module.train_deg_ce.append(m.get("train_deg_ce", torch.tensor(0.0)).item())
        pl_module.train_recon.append(m.get("train_recon", torch.tensor(0.0)).item())
        if hasattr(pl_module, "train_exist"):
            pl_module.train_exist.append(m.get("train_exist", torch.tensor(0.0)).item())
        if hasattr(pl_module, "train_node_label_ce"):
            pl_module.train_node_label_ce.append(m.get("train_node_label_ce", m.get("train_label_ce", torch.tensor(0.0))).item())
        elif hasattr(pl_module, "train_label_ce"):
            pl_module.train_label_ce.append(m.get("train_label_ce", m.get("train_node_label_ce", torch.tensor(0.0))).item())
        if hasattr(pl_module, "train_edge_label_ce"):
            pl_module.train_edge_label_ce.append(m.get("train_edge_label_ce", torch.tensor(0.0)).item())
        if pl_module.use_locality_supervision:
            pl_module.train_edge_loss.append(m.get("train_edge_ce", m.get("train_edge_loss", torch.tensor(0.0))).item())
            pl_module.train_edge_acc.append(m.get("train_edge_acc", torch.tensor(0.0)).item())
        if hasattr(pl_module, "use_auxiliary_locality_supervision") and pl_module.use_auxiliary_locality_supervision:
            pl_module.train_aux_edge_loss.append(m.get("train_aux_locality_ce", m.get("train_aux_edge_loss", torch.tensor(0.0))).item())
            pl_module.train_aux_edge_acc.append(m.get("train_aux_edge_acc", torch.tensor(0.0)).item())

    def on_validation_epoch_end(self, trainer, pl_module):
        m = trainer.callback_metrics
        pl_module.val_losses.append(m.get("val_total", torch.tensor(0.0)).item())
        pl_module.val_deg_ce.append(m.get("val_deg_ce", torch.tensor(0.0)).item())
        pl_module.val_recon.append(m.get("val_recon", torch.tensor(0.0)).item())
        if hasattr(pl_module, "val_exist"):
            pl_module.val_exist.append(m.get("val_exist", torch.tensor(0.0)).item())
        if hasattr(pl_module, "val_node_label_ce"):
            pl_module.val_node_label_ce.append(m.get("val_node_label_ce", m.get("val_label_ce", torch.tensor(0.0))).item())
        elif hasattr(pl_module, "val_label_ce"):
            pl_module.val_label_ce.append(m.get("val_label_ce", m.get("val_node_label_ce", torch.tensor(0.0))).item())
        if hasattr(pl_module, "val_edge_label_ce"):
            pl_module.val_edge_label_ce.append(m.get("val_edge_label_ce", torch.tensor(0.0)).item())
        if pl_module.use_locality_supervision:
            pl_module.val_edge_loss.append(m.get("val_edge_ce", m.get("val_edge_loss", torch.tensor(0.0))).item())
            pl_module.val_edge_acc.append(m.get("val_edge_acc", torch.tensor(0.0)).item())
        if hasattr(pl_module, "use_auxiliary_locality_supervision") and pl_module.use_auxiliary_locality_supervision:
            pl_module.val_aux_edge_loss.append(m.get("val_aux_locality_ce", m.get("val_aux_edge_loss", torch.tensor(0.0))).item())
            pl_module.val_aux_edge_acc.append(m.get("val_aux_edge_acc", torch.tensor(0.0)).item())
