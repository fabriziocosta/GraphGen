import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from pytorch_lightning.callbacks import RichProgressBar, ModelCheckpoint
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
import os
from sklearn.preprocessing import MinMaxScaler # Import MinMaxScaler
import contextlib
import logging

from .lightning_utils import run_trainer_fit

############################################
# Custom low‐rank linear layer (LowRankLinear)
############################################

class LowRankLinear(nn.Module):
    """
    A linear layer whose weight matrix is factorized as A @ B,
    where A is (in_features x thin_size) and B is (thin_size x out_features).
    """
    def __init__(self, in_features, out_features, thin_size, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.thin_size = thin_size
        self.A = nn.Parameter(torch.Tensor(in_features, thin_size))
        self.B = nn.Parameter(torch.Tensor(thin_size, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(self.in_features)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x):
        out = x @ self.A @ self.B
        if self.bias is not None:
            out = out + self.bias
        return out

############################################
# Residual block with dropout and LeakyReLU
############################################

class ResidualBlock(nn.Module):
    """
    A residual block with low-rank linear -> dropout -> LeakyReLU, plus skip.
    """
    def __init__(self, in_features, out_features, thin_size, dropout_prob, negative_slope, use_layernorm: bool = False):
        super().__init__()
        self.linear = LowRankLinear(in_features, out_features, thin_size)
        self.norm = nn.LayerNorm(out_features) if use_layernorm else nn.Identity()
        self.dropout = nn.Dropout(dropout_prob)
        self.activation = nn.LeakyReLU(negative_slope)
        self.skip = LowRankLinear(in_features, out_features, thin_size) if in_features != out_features else nn.Identity()
    
    def forward(self, x):
        identity = self.skip(x) # Calculate skip connection first
        out = self.linear(x)
        out = self.norm(out)    # Apply optional LayerNorm
        out = self.dropout(out) # Apply dropout
        out = self.activation(out) # Apply activation
        return out + identity   # Add skip connection

############################################
# LowRankMLP Network
############################################

class LowRankMLPNet(nn.Module):
    """
    MLP built from ResidualBlocks and a final LowRankLinear.
    """
    def __init__(self, input_dim, output_dim,
                 hidden_layers, hidden_dim, thin_size, dropout,
                 negative_slope, use_layernorm_in_residual: bool = False):
        super().__init__()
        self.hidden_layers = hidden_layers
        if hidden_layers > 0:
            blocks = [ResidualBlock(input_dim, hidden_dim, thin_size, dropout, negative_slope, use_layernorm_in_residual)]
            blocks += [
                ResidualBlock(hidden_dim, hidden_dim, thin_size, dropout, negative_slope, use_layernorm_in_residual)
                for _ in range(hidden_layers -1)
            ]
            self.blocks = nn.ModuleList(blocks)
            self.out_layer = LowRankLinear(hidden_dim, output_dim, thin_size)
        else:
            self.out_layer = LowRankLinear(input_dim, output_dim, thin_size)

    def forward(self, x):
        # Use getattr to safely access blocks, in case hidden_layers is 0
        for block in getattr(self, 'blocks', []): 
            x = block(x)
        return self.out_layer(x)

############################################
# StopWhenLRBelow callback with min_epochs
############################################

class StopWhenLRBelow(pl.Callback):
    """
    Stops training when LR falls below a threshold, but only after a minimum number
    of epochs.
    """
    def __init__(self, min_lr=1e-8, min_epochs=0):
        super().__init__()
        self.min_lr = min_lr
        self.min_epochs = min_epochs

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        if epoch < self.min_epochs:
            return
        lr = trainer.optimizers[0].param_groups[0]['lr']
        if lr < self.min_lr:
            print(f"LR {lr:.2e} below {self.min_lr:.2e} at epoch {epoch}; stopping.")
            trainer.should_stop = True

############################################
# Lightning Module
############################################

class LowRankMLPModule(pl.LightningModule):
    """
    Lightning wrapper around LowRankMLPNet, tracks losses.
    """
    def __init__(self, net, lr, task='classification', lr_patience=3, class_weights=None, verbose=False):
        super().__init__()
        self.net = net
        self.lr = lr
        self.task = task
        self.lr_patience = lr_patience
        self.verbose = verbose

        if task == 'classification':
            self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        elif task == 'regression':
            self.loss_fn = nn.MSELoss()
        else:
            raise ValueError("task must be 'classification' or 'regression'")

        self.train_losses = []
        self.val_losses = []
        self._train_batch = []
        self._val_batch = []

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss_fn(self(x), y)
        self.log('train_loss', loss)
        self._train_batch.append(loss.detach())
        return loss

    def on_train_epoch_end(self):
        if self._train_batch:
            avg = torch.stack(self._train_batch).mean().item()
            self.train_losses.append(avg)
            self._train_batch = []

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss_fn(self(x), y)
        self.log('val_loss', loss, prog_bar=True)
        self._val_batch.append(loss.detach())

    def on_validation_epoch_end(self):
        if self._val_batch:
            avg = torch.stack(self._val_batch).mean().item()
            self.val_losses.append(avg)
            self._val_batch = []

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, patience=self.lr_patience, factor=0.1
        )
        return {
            'optimizer': opt,
            'lr_scheduler': {'scheduler': sched, 'monitor': 'val_loss', 'interval': 'epoch'}
        }

    def on_train_end(self):
        if not self.verbose or not (self.train_losses and self.val_losses):
            return
        min_len = min(len(self.train_losses), len(self.val_losses))
        skip = 5 if min_len > 5 else 0
        t = self.train_losses[skip:min_len]
        v = self.val_losses[skip:min_len]
        epochs = range(skip + 1, skip + 1 + len(t))
        plt.figure(figsize=(10,5))
        plt.plot(epochs, t, label='Train Loss')
        plt.plot(epochs, v, label='Validation Loss')
        plt.yscale('log')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()

############################################
# sklearn-compatible estimator with partial optimizer reset
############################################

class LowRankMLP(BaseEstimator, ClassifierMixin):
    """
    sklearn-like estimator wrapping our Lightning MLP, supports partial optimizer reset:
    preserves model weights but reinitializes optimizer and LR scheduler on warm start.
    """
    def __init__(
        self,
        hidden_layers=2,
        hidden_dim=100,
        thin_size=10,
        dropout=0.5,
        negative_slope=0.01,
        lr=1e-3,
        max_epochs=30,
        batch_size=32,
        task='classification',
        lr_patience=5,
        verbose=False,
        enable_progress_bar=False,
        balance=True,
        min_lr=1e-6,
        warm_start=True,
        min_epochs=0,
        use_layernorm_in_residual: bool = True,
        use_minmax_scaler: bool = True 
    ):
        self.hidden_layers = hidden_layers
        self.hidden_dim = hidden_dim
        self.thin_size = thin_size
        self.dropout = dropout
        self.negative_slope = negative_slope
        self.lr = lr
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.task = task
        self.lr_patience = lr_patience
        self.verbose = verbose
        self.enable_progress_bar = enable_progress_bar
        self.balance = balance
        self.min_lr = min_lr
        self.warm_start = warm_start
        self.min_epochs = min_epochs
        self.use_layernorm_in_residual = use_layernorm_in_residual
        self.use_minmax_scaler = use_minmax_scaler # Store the new parameter
        self.scaler_ = None # Initialize scaler attribute

    def fit(self, X, y):
        # 1. Data & warm-start checks
        X_arr = np.asarray(X)
        y_arr = np.asarray(y)

        # 1.1. Fit and transform X using MinMaxScaler if enabled
        if self.use_minmax_scaler:
            if not (self.warm_start and hasattr(self, 'scaler_') and self.scaler_ is not None):
                 self.scaler_ = MinMaxScaler()
                 X_arr = self.scaler_.fit_transform(X_arr)
            elif self.scaler_ is not None: # Apply existing scaler if warm_start
                 X_arr = self.scaler_.transform(X_arr)

        if self.warm_start and hasattr(self, 'net_'):
            if X_arr.shape[1] != self.input_dim_:
                raise ValueError(f"Warm start input_dim={self.input_dim_}, got {X_arr.shape[1]}")
            if self.task == 'classification':
                new_cls = np.unique(y_arr.ravel())
                if not np.array_equal(new_cls, self.classes_):
                    raise ValueError(f"Warm start classes changed from {self.classes_} to {new_cls}")
            else:
                out_dim = y_arr.shape[1] if y_arr.ndim > 1 else 1
                if out_dim != self.output_dim_:
                    raise ValueError(f"Warm start output_dim={self.output_dim_}, got {out_dim}")

        # 2. Prepare target tensor
        if self.task == 'classification':
            flat = y_arr.ravel()
            if not (self.warm_start and hasattr(self, 'classes_')):
                self.classes_ = np.unique(flat)
            mapping = {c: i for i, c in enumerate(self.classes_)}
            mapped = np.vectorize(mapping.get)(flat)
            y_tensor = torch.from_numpy(mapped).long()
            class_weights = torch.tensor(1.0/np.bincount(mapped), dtype=torch.float32) if self.balance else None
            out_dim = len(self.classes_)
        else:
            y_tensor = torch.from_numpy(y_arr.astype(np.float32))
            class_weights = None
            out_dim = y_arr.shape[1] if y_arr.ndim > 1 else 1

        # 3. Build or reuse model weights
        if not (self.warm_start and hasattr(self, 'net_')):
            self.input_dim_ = X_arr.shape[1]
            self.output_dim_ = out_dim
            self.net_ = LowRankMLPNet(
                input_dim=self.input_dim_,
                output_dim=self.output_dim_,
                hidden_layers=self.hidden_layers,
                hidden_dim=self.hidden_dim,
                thin_size=self.thin_size,
                    dropout=self.dropout, # type: ignore
                    negative_slope=self.negative_slope,
                    use_layernorm_in_residual=self.use_layernorm_in_residual # Pass to LowRankMLPNet
            )
        net = self.net_

        # 4. Create new LightningModule (fresh optimizer & scheduler)
        self.module_ = LowRankMLPModule(
            net,
            lr=self.lr,
            task=self.task,
            lr_patience=self.lr_patience,
            class_weights=class_weights,
            verbose=self.verbose
        )

        # 5. DataLoaders
        X_tensor = torch.from_numpy(X_arr.astype(np.float32))
        ds = TensorDataset(X_tensor, y_tensor)
        val_size = int(0.1 * len(ds))
        train_size = len(ds) - val_size
        tr_ds, vl_ds = random_split(ds, [train_size, val_size])
        tr_ld = DataLoader(tr_ds, batch_size=self.batch_size, shuffle=True)
        vl_ld = DataLoader(vl_ds, batch_size=self.batch_size)

        # 6. Callbacks
        callbacks = [StopWhenLRBelow(min_lr=self.min_lr, min_epochs=self.min_epochs)]
        if self.enable_progress_bar:
            callbacks.append(RichProgressBar( refresh_rate=30 ))
        # keep checkpointing as reference, but not to resume
        ckpt = ModelCheckpoint(dirpath=os.getcwd(), filename='lowrankmlp_ckpt', save_last=True, save_top_k=0)
        callbacks.append(ckpt)

        # 7. Trainer (partial optimizer reset: new optimizer, new lr scheduler)
        trainer_kwargs = dict(
            max_epochs=self.max_epochs,
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=1,
            callbacks=callbacks,
            enable_checkpointing=True,
            enable_progress_bar=self.enable_progress_bar
        )
        if not self.verbose:
            log = logging.getLogger('pytorch_lightning'); prev = log.level; log.setLevel(logging.ERROR)
            with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                trainer = pl.Trainer(logger=False, **trainer_kwargs)
                run_trainer_fit(
                    trainer,
                    self.module_,
                    tr_ld,
                    vl_ld,
                    context=f"{self.__class__.__name__}.fit",
                )
            log.setLevel(prev)
        else:
            trainer = pl.Trainer(**trainer_kwargs)
            run_trainer_fit(
                trainer,
                self.module_,
                tr_ld,
                vl_ld,
                context=f"{self.__class__.__name__}.fit",
            )

        # 8. Record state & return
        self._epochs_trained_ = trainer.current_epoch
        self._checkpoint_path = ckpt.last_model_path
        self.net_ = self.module_.net.eval()
        return self

    def predict(self, X):
        """
        Predict class labels or regression outputs for X.
        """
        # Apply scaler if enabled and fitted
        if self.use_minmax_scaler and self.scaler_ is not None:
            X = self.scaler_.transform(X)
        X_tensor = torch.from_numpy(np.asarray(X).astype(np.float32))
        with torch.no_grad():
            out = self.net_(X_tensor)
        if self.task == 'classification':
            idx = out.argmax(dim=1).cpu().numpy()
            return self.classes_[idx]
        return out.cpu().numpy()

    def predict_proba(self, X):
        """
        Predict class probabilities for classification task.
        """
        if self.task != 'classification':
            raise AttributeError("predict_proba only available for classification tasks.")
        # Apply scaler if enabled and fitted
        if self.use_minmax_scaler and self.scaler_ is not None:
            X = self.scaler_.transform(X)
        X_tensor = torch.from_numpy(np.asarray(X).astype(np.float32))
        with torch.no_grad():
            logits = self.net_(X_tensor)
        return F.softmax(logits, dim=1).cpu().numpy()
    
    def score(self, X, y):
        """
        Returns accuracy for classification, R^2 for regression.
        """
        from sklearn.metrics import accuracy_score, r2_score
        y_pred = self.predict(X)
        if self.task == 'classification':
            return accuracy_score(y, y_pred)
        else:
            return r2_score(y, y_pred)
