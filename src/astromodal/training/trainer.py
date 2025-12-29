"""
Training infrastructure for astromodal.

This module provides a flexible Trainer class that handles the complete
training loop with support for:
- Meaningful reconstruction losses
- Mixed precision training (AMP)
- Gradient accumulation and clipping
- Learning rate scheduling
- Checkpointing and resumption
- Progress logging
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ..config import TrainingConfig, MeaningfulLossConfig
from ..core.vocab import VocabSpec
from .losses import combined_loss
from .utils import (
    cosine_schedule_with_warmup,
    count_parameters,
    format_time,
    log_gradient_stats,
)


class Trainer:
    """
    Trainer class for astromodal models.

    This class encapsulates the training loop and provides a clean interface
    for training models with meaningful reconstruction losses.

    Parameters
    ----------
    model : nn.Module
        PyTorch model to train
    vocab : VocabSpec
        Vocabulary specification
    train_loader : DataLoader
        Training data loader
    config : TrainingConfig
        Training configuration
    image_codebook : torch.Tensor
        Image codebook for reconstruction loss (V_IMAGE, D_img)
    spectrum_codebook : torch.Tensor
        Spectrum codebook for reconstruction loss (V_SPECTRA, D_spec)
    scalar_centers : torch.Tensor
        Scalar bin centers for reconstruction loss (V_SCALAR,)
    device : str, default="cuda"
        Device to train on
    output_dir : Optional[str], default=None
        Directory for checkpoints and logs
    use_amp : bool, default=True
        Whether to use automatic mixed precision
    use_ce : bool, default=True
        Whether to include cross-entropy loss
    compile_model : bool, default=False
        Whether to use torch.compile for speedup

    Attributes
    ----------
    global_step : int
        Current global training step
    current_epoch : int
        Current training epoch

    Examples
    --------
    >>> trainer = Trainer(
    ...     model=model,
    ...     vocab=vocab,
    ...     train_loader=train_loader,
    ...     config=config,
    ...     image_codebook=img_cb,
    ...     spectrum_codebook=spec_cb,
    ...     scalar_centers=sca_centers,
    ... )
    >>> trainer.train(num_epochs=10)
    """

    def __init__(
        self,
        model: nn.Module,
        vocab: VocabSpec,
        train_loader: DataLoader,
        config: TrainingConfig,
        image_codebook: torch.Tensor,
        spectrum_codebook: torch.Tensor,
        scalar_centers: torch.Tensor,
        device: str = "cuda",
        output_dir: Optional[str] = None,
        use_amp: bool = True,
        use_ce: bool = True,
        compile_model: bool = False,
    ):
        self.model = model
        self.vocab = vocab
        self.train_loader = train_loader
        self.config = config
        self.device = device
        self.output_dir = Path(output_dir) if output_dir else None
        self.use_amp = use_amp and (device == "cuda")
        self.use_ce = use_ce

        # Training state
        self.global_step = 0
        self.current_epoch = 0

        # Move model to device
        self.model.to(self.device)

        # Optional compilation
        if compile_model and device == "cuda":
            try:
                self.model = torch.compile(self.model, mode="max-autotune")
                print("[trainer] torch.compile enabled")
            except Exception as e:
                print(f"[trainer] torch.compile failed: {e}")

        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.95),
        )

        # Setup gradient scaler for mixed precision
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # Move codebooks to device
        dtype = torch.float16 if self.use_amp else torch.float32
        self.image_codebook = image_codebook.to(device=device, dtype=dtype)
        self.spectrum_codebook = spectrum_codebook.to(device=device, dtype=dtype)
        self.scalar_centers = scalar_centers.to(device=device, dtype=dtype)

        # Loss config
        self.loss_config = {
            "enabled": config.meaningful_loss.enabled,
            "sample_k": config.meaningful_loss.sample_k_per_modality,
            "weight_image": config.meaningful_loss.weight_image,
            "weight_scalar": config.meaningful_loss.weight_scalar,
            "weight_spectrum": config.meaningful_loss.weight_spectrum,
        }

        # Create output directory
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        # Log model info
        total_params, trainable_params = count_parameters(self.model)
        print(f"[trainer] Total parameters: {total_params:,}")
        print(f"[trainer] Trainable parameters: {trainable_params:,}")
        print(f"[trainer] Device: {self.device}")
        print(f"[trainer] Mixed precision: {self.use_amp}")
        print(f"[trainer] Use CE loss: {self.use_ce}")
        print(f"[trainer] Meaningful loss: {self.loss_config['enabled']}")

    def compute_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute loss with both CE and meaningful reconstruction.

        Parameters
        ----------
        logits : torch.Tensor
            Model logits (B, L, V)
        targets : torch.Tensor
            Target tokens (B, L)

        Returns
        -------
        Tuple[torch.Tensor, Dict[str, float]]
            (loss, stats_dict)
        """
        return combined_loss(
            logits=logits,
            targets=targets,
            vocab=self.vocab,
            image_codebook=self.image_codebook,
            spectrum_codebook=self.spectrum_codebook,
            scalar_centers=self.scalar_centers,
            config=self.loss_config,
            use_ce=self.use_ce,
        )

    def get_lr(self) -> float:
        """
        Compute learning rate for current step.

        Returns
        -------
        float
            Current learning rate
        """
        return cosine_schedule_with_warmup(
            step=self.global_step,
            base_lr=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
        )

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Execute a single training step.

        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            Batch dict with keys: tokens, types, attn_mask

        Returns
        -------
        Dict[str, float]
            Statistics dictionary
        """
        # Move batch to device
        tokens = batch["tokens"].to(self.device, non_blocking=True)
        types = batch["types"].to(self.device, non_blocking=True)
        attn_mask = batch["attn_mask"].to(self.device, non_blocking=True)

        # Next-token prediction setup
        x = tokens[:, :-1]
        x_types = types[:, :-1]
        x_mask = attn_mask[:, :-1]
        y = tokens[:, 1:].contiguous()

        # Update learning rate
        lr = self.get_lr()
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

        # Forward pass with mixed precision
        with torch.amp.autocast("cuda", enabled=self.use_amp):
            logits = self.model(x, x_types, x_mask)
            loss, stats = self.compute_loss(logits, y)

        # Backward pass
        self.scaler.scale(loss / self.config.grad_accum_steps).backward()

        # Update weights if accumulation is complete
        if (self.global_step + 1) % self.config.grad_accum_steps == 0:
            # Gradient clipping
            if self.config.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.grad_clip,
                )

            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

        # Add learning rate to stats
        stats["lr"] = lr
        stats["loss_total"] = float(loss.detach().cpu().item())

        return stats

    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.

        Returns
        -------
        Dict[str, float]
            Epoch statistics (averages)
        """
        self.model.train()
        self.current_epoch += 1

        # Statistics tracking
        epoch_stats = {}
        n_batches = 0

        # Progress bar
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch}",
            dynamic_ncols=True,
        )

        for batch in pbar:
            stats = self.train_step(batch)
            self.global_step += 1
            n_batches += 1

            # Accumulate stats
            for key, val in stats.items():
                epoch_stats[key] = epoch_stats.get(key, 0.0) + val

            # Update progress bar
            if self.global_step % self.config.log_every == 0:
                pbar.set_postfix({
                    "loss": f"{stats.get('loss_total', 0):.3f}",
                    "ce": f"{stats.get('loss_ce', 0):.3f}",
                    "img": f"{stats.get('loss_img', 0):.3f}",
                    "sca": f"{stats.get('loss_sca', 0):.3f}",
                    "spec": f"{stats.get('loss_spec', 0):.3f}",
                    "lr": f"{stats.get('lr', 0):.2e}",
                })

            # Save checkpoint
            if self.output_dir and self.global_step % self.config.save_every_steps == 0:
                self.save_checkpoint(f"ckpt_step_{self.global_step:08d}.pt")

        # Average stats over epoch
        if n_batches > 0:
            epoch_stats = {k: v / n_batches for k, v in epoch_stats.items()}

        return epoch_stats

    def train(self, num_epochs: Optional[int] = None) -> None:
        """
        Run full training loop.

        Parameters
        ----------
        num_epochs : Optional[int], default=None
            Number of epochs to train. If None, uses config.epochs
        """
        if num_epochs is None:
            num_epochs = self.config.epochs

        print(f"\n[trainer] Starting training for {num_epochs} epochs")
        print(f"[trainer] Total steps per epoch: {len(self.train_loader)}")
        print(f"[trainer] Warmup steps: {self.config.warmup_steps}")
        print(f"[trainer] Gradient accumulation: {self.config.grad_accum_steps}")
        print()

        start_time = time.time()

        for epoch in range(num_epochs):
            epoch_stats = self.train_epoch()

            # Log epoch summary
            elapsed = time.time() - start_time
            print(f"\n[trainer] Epoch {self.current_epoch} complete")
            print(f"  Loss: {epoch_stats.get('loss_total', 0):.4f}")
            if "loss_ce" in epoch_stats:
                print(f"  CE: {epoch_stats.get('loss_ce', 0):.4f}")
            if "loss_img" in epoch_stats:
                print(f"  Image: {epoch_stats.get('loss_img', 0):.4f}")
            if "loss_sca" in epoch_stats:
                print(f"  Scalar: {epoch_stats.get('loss_sca', 0):.4f}")
            if "loss_spec" in epoch_stats:
                print(f"  Spectrum: {epoch_stats.get('loss_spec', 0):.4f}")
            print(f"  Time: {format_time(elapsed)}")
            print()

            # Save epoch checkpoint
            if self.output_dir:
                self.save_checkpoint(f"ckpt_epoch_{self.current_epoch:03d}.pt")

        total_time = time.time() - start_time
        print(f"[trainer] Training complete! Total time: {format_time(total_time)}")

    def save_checkpoint(self, filename: str) -> None:
        """
        Save training checkpoint.

        Parameters
        ----------
        filename : str
            Checkpoint filename
        """
        if not self.output_dir:
            return

        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict() if self.use_amp else None,
            "global_step": self.global_step,
            "epoch": self.current_epoch,
            "config": {
                "vocab_size": self.vocab.vocab_size,
                "base_image": self.vocab.base_image,
                "base_scalar": self.vocab.base_scalar,
                "base_spectrum": self.vocab.base_spectrum,
                "loss_config": self.loss_config,
            },
        }

        path = self.output_dir / filename
        torch.save(checkpoint, path)
        print(f"[trainer] Saved checkpoint: {path}")

    def load_checkpoint(self, path: str) -> None:
        """
        Load training checkpoint and resume.

        Parameters
        ----------
        path : str
            Path to checkpoint file
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        if self.use_amp and checkpoint.get("scaler"):
            self.scaler.load_state_dict(checkpoint["scaler"])

        self.global_step = checkpoint.get("global_step", 0)
        self.current_epoch = checkpoint.get("epoch", 0)

        print(f"[trainer] Loaded checkpoint from {path}")
        print(f"[trainer] Resuming from step {self.global_step}, epoch {self.current_epoch}")

    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate model on validation set.

        Parameters
        ----------
        val_loader : DataLoader
            Validation data loader

        Returns
        -------
        Dict[str, float]
            Validation statistics
        """
        self.model.eval()

        val_stats = {}
        n_batches = 0

        pbar = tqdm(val_loader, desc="Validation", dynamic_ncols=True)

        for batch in pbar:
            # Move batch to device
            tokens = batch["tokens"].to(self.device, non_blocking=True)
            types = batch["types"].to(self.device, non_blocking=True)
            attn_mask = batch["attn_mask"].to(self.device, non_blocking=True)

            # Next-token prediction setup
            x = tokens[:, :-1]
            x_types = types[:, :-1]
            x_mask = attn_mask[:, :-1]
            y = tokens[:, 1:].contiguous()

            # Forward pass
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                logits = self.model(x, x_types, x_mask)
                loss, stats = self.compute_loss(logits, y)

            stats["loss_total"] = float(loss.detach().cpu().item())

            # Accumulate stats
            for key, val in stats.items():
                val_stats[key] = val_stats.get(key, 0.0) + val
            n_batches += 1

            # Update progress bar
            pbar.set_postfix({
                "loss": f"{stats.get('loss_total', 0):.3f}",
            })

        # Average stats
        if n_batches > 0:
            val_stats = {k: v / n_batches for k, v in val_stats.items()}

        self.model.train()
        return val_stats
