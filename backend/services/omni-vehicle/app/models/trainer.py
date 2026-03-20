"""
LPRv3 Training Pipeline
Focal CTC Loss + Curriculum Learning + Mixed Precision
Optimized for RTX 5090 with DDP
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass
from datetime import datetime
import json

logger = logging.getLogger(__name__)


# ============================================
# HARD CHARACTER INDICES (for confusion penalty)
# These are indices in the standard VN plate charset:
#   0123456789ABCDEFGHKLMNPRSTUVXYZ
# Confusion-prone pairs from DeepSeek/Gemini research:
#   8↔0, 8↔B, 0↔D, 1↔I, 5↔S, 6↔0, 2↔3, 4↔1
# ============================================
HARD_CHARS_VN = set('80BD1I5S6234')


class FocalCTCLoss(nn.Module):
    """
    Focal CTC Loss with confusion-pair penalty.
    
    Upgrades over standard CTC:
    1. Focal modulation: (1 - exp(-loss))^γ focuses on hard examples
    2. Confusion penalty: Extra weight for samples containing chars
       that are commonly confused (8↔0, B↔D, 1↔I, 5↔S, etc.)
    
    Based on research from DeepSeek/Gemini analysis:
    - Gemini: "Focal CTC introduces (1-p_t)^γ modulating factor"
    - DeepSeek: "hard negative mining focused on confusion pairs"
    """
    
    def __init__(self, blank: int = 0, reduction: str = 'mean', 
                 alpha: float = 0.25, gamma: float = 2.0,
                 confusion_boost: float = 1.5,
                 hard_chars: str = '80BD1I5S6234'):
        """
        Args:
            blank: Blank token index
            reduction: Loss reduction method
            alpha: Focal weight base
            gamma: Focusing parameter (higher = more focus on hard examples)
            confusion_boost: Extra weight multiplier for samples with confusable chars
            hard_chars: Characters that are confusion-prone
        """
        super().__init__()
        self.ctc_loss = nn.CTCLoss(blank=blank, reduction='none', zero_infinity=True)
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.confusion_boost = confusion_boost
        self.hard_chars = set(hard_chars)
        
        # Pre-compute hard character indices for the VN plate charset
        # Charset: 0123456789ABCDEFGHKLMNPRSTUVXYZ (1-based, 0=blank)
        self._charset = "0123456789ABCDEFGHKLMNPRSTUVXYZ"
        self._hard_indices = set()
        for i, ch in enumerate(self._charset):
            if ch in self.hard_chars:
                self._hard_indices.add(i + 1)  # +1 because 0 = blank
    
    def _compute_confusion_weights(self, targets: torch.Tensor,
                                    target_lengths: torch.Tensor) -> torch.Tensor:
        """
        Compute per-sample confusion penalty weight.
        Samples containing hard characters get higher weight.
        
        Returns:
            (N,) tensor of weights, 1.0 for normal, confusion_boost for hard samples
        """
        N = targets.shape[0]
        weights = torch.ones(N, device=targets.device)
        
        for i in range(N):
            length = target_lengths[i].item()
            target_seq = targets[i, :length]
            # Check if any character index maps to a hard character
            has_hard = any(idx.item() in self._hard_indices for idx in target_seq)
            if has_hard:
                weights[i] = self.confusion_boost
        
        return weights
    
    def forward(self, log_probs: torch.Tensor, targets: torch.Tensor,
               input_lengths: torch.Tensor, target_lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            log_probs: (T, N, C) log probabilities from CTC
            targets: (N, L) target sequences
            input_lengths: (N,) input lengths
            target_lengths: (N,) target lengths
        
        Returns:
            loss: Scalar loss
        """
        # 1. Base CTC loss per sample
        losses = self.ctc_loss(log_probs, targets, input_lengths, target_lengths)  # (N,)
        
        # 2. Focal modulation: focus on hard examples
        # weight = (1 - exp(-loss))^gamma
        focal_weight = torch.pow(1.0 - torch.exp(-losses.detach()), self.gamma)
        
        # 3. Confusion pair penalty: boost loss for samples with hard characters
        confusion_weight = self._compute_confusion_weights(targets, target_lengths)
        
        # 4. Combined weight = focal × confusion
        combined_weight = focal_weight * confusion_weight
        
        # 5. Apply weights
        weighted_losses = combined_weight * losses
        
        if self.reduction == 'mean':
            return weighted_losses.mean()
        elif self.reduction == 'sum':
            return weighted_losses.sum()
        else:
            return weighted_losses


class HardNegativeSampler:
    """
    Hard Negative Mining Sampler for LPR training.
    
    Over-samples training examples containing confusion-prone characters
    (8, 0, B, D, 1, I, 5, S, etc.) to force the model to learn distinguishing
    features for these hard pairs.
    
    Based on DeepSeek research: "hard negative mining tập trung vào các cặp
    ký tự dễ nhầm" + Gemini: "generate hard negative pairs to force network
    to learn subtle serif features"
    
    Usage:
        sampler = HardNegativeSampler(dataset, hard_ratio=0.4)
        loader = DataLoader(dataset, batch_sampler=sampler)
    """
    
    def __init__(self, dataset, hard_chars: str = '80BD1I5S6234',
                 hard_ratio: float = 0.4, batch_size: int = 32):
        """
        Args:
            dataset: Training dataset with 'label' attribute per sample
            hard_chars: Characters considered confusion-prone
            hard_ratio: Target ratio of hard samples in each batch (0.0-1.0)
            batch_size: Batch size
        """
        self.dataset = dataset
        self.hard_chars = set(hard_chars)
        self.hard_ratio = hard_ratio
        self.batch_size = batch_size
        
        # Pre-compute indices of hard vs easy samples
        self.hard_indices = []
        self.easy_indices = []
        
        for idx in range(len(dataset)):
            label = self._get_label(idx)
            if any(c in self.hard_chars for c in label):
                self.hard_indices.append(idx)
            else:
                self.easy_indices.append(idx)
        
        self.hard_indices = np.array(self.hard_indices)
        self.easy_indices = np.array(self.easy_indices)
        
        logger.info(
            f"🎯 HardNegativeSampler: {len(self.hard_indices)} hard / "
            f"{len(self.easy_indices)} easy samples (target ratio: {hard_ratio:.0%})"
        )
    
    def _get_label(self, idx: int) -> str:
        """Extract label string from dataset sample."""
        sample = self.dataset[idx]
        if isinstance(sample, dict) and 'label' in sample:
            return sample['label']
        if isinstance(sample, dict) and 'plate_text' in sample:
            return sample['plate_text']
        return ''
    
    def __iter__(self):
        """Yield batches with controlled hard/easy ratio."""
        n_hard = int(self.batch_size * self.hard_ratio)
        n_easy = self.batch_size - n_hard
        
        # Shuffle
        hard_perm = np.random.permutation(len(self.hard_indices))
        easy_perm = np.random.permutation(len(self.easy_indices))
        
        h_ptr, e_ptr = 0, 0
        
        total_batches = len(self.dataset) // self.batch_size
        
        for _ in range(total_batches):
            batch = []
            
            # Sample hard examples (with wrapping)
            for _ in range(n_hard):
                if h_ptr >= len(hard_perm):
                    hard_perm = np.random.permutation(len(self.hard_indices))
                    h_ptr = 0
                batch.append(self.hard_indices[hard_perm[h_ptr]])
                h_ptr += 1
            
            # Sample easy examples (with wrapping)
            for _ in range(n_easy):
                if e_ptr >= len(easy_perm):
                    easy_perm = np.random.permutation(len(self.easy_indices))
                    e_ptr = 0
                batch.append(self.easy_indices[easy_perm[e_ptr]])
                e_ptr += 1
            
            # Shuffle within batch
            np.random.shuffle(batch)
            yield batch
    
    def __len__(self):
        return len(self.dataset) // self.batch_size


@dataclass
class TrainingConfig:
    """Training hyperparameters"""
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    warmup_steps: int = 1000
    
    # Mixed precision
    use_amp: bool = True
    
    # Data augmentation
    augmentation_level: str = 'hard'  # light, medium, hard
    
    # Curriculum learning
    use_curriculum: bool = True
    curriculum_stages: int = 5
    
    # Model
    model_type: str = 'lprv3'
    checkpoint_dir: str = './checkpoints'
    log_dir: str = './logs'
    
    # Training specifics
    num_workers: int = 4
    pin_memory: bool = True
    gradient_clip: float = 1.0
    
    def to_dict(self) -> Dict:
        return {k: str(v) for k, v in self.__dict__.items()}


class LPRTrainer:
    """
    Complete training pipeline for LPRv3
    """
    
    def __init__(self, model: nn.Module, config: TrainingConfig,
                device: torch.device = None):
        """
        Args:
            model: LPRv3 model
            config: Training configuration
            device: Device to train on
        """
        self.model = model
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Setup directories
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.log_dir = Path(config.log_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Loss and optimizer
        self.criterion = FocalCTCLoss(blank=0, alpha=0.25, gamma=2.0)
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = self._create_scheduler()
        
        # Mixed precision
        self.scaler = GradScaler() if config.use_amp else None
        
        # Training stats
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.global_step = 0
        
        # Curriculum learning
        self.curriculum_stage = 0
        
        logger.info(f"✅ Trainer initialized on {self.device}")
    
    def _create_scheduler(self) -> optim.lr_scheduler.LambdaLR:
        """Create learning rate scheduler with warmup + cosine decay.
        
        Note: scheduler.step() is called per batch (global_step),
        so we estimate total_steps = num_epochs * approx_steps_per_epoch.
        """
        warmup = self.config.warmup_steps
        # Rough estimate: assume ~500 batches/epoch if dataset size unknown
        est_steps_per_epoch = 500
        total_steps = max(warmup + 1, self.config.num_epochs * est_steps_per_epoch)
        
        def lr_lambda(step):
            if step < warmup:
                return float(step) / float(max(1, warmup))
            # Linear decay from 1.0 to 0.0 over remaining steps
            progress = float(step - warmup) / float(max(1, total_steps - warmup))
            return max(0.0, 1.0 - progress)
        
        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def _get_curriculum_difficulty(self, epoch: int) -> float:
        """
        Get curriculum learning difficulty level
        Progresses from easy to hard samples over training
        
        Returns: Difficulty in range [0, 1] where 1 is hardest
        """
        if not self.config.use_curriculum:
            return 1.0
        
        total_epochs = self.config.num_epochs
        stage_epochs = total_epochs // self.config.curriculum_stages
        
        stage = min(epoch // stage_epochs, self.config.curriculum_stages - 1)
        return float(stage) / max(1, self.config.curriculum_stages - 1)
    
    def train_epoch(self, train_loader: DataLoader, 
                   validation_loader: Optional[DataLoader] = None) -> Dict:
        """
        Train for one epoch
        """
        self.model.train()
        self.current_epoch += 1
        
        epoch_losses = []
        epoch_metrics = {
            'loss': 0.0,
            'loss_top': 0.0,
            'loss_bottom': 0.0,
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'curriculum_stage': self._get_curriculum_difficulty(self.current_epoch),
        }
        
        logger.info(f"\n📚 Epoch {self.current_epoch}/{self.config.num_epochs}")
        logger.info(f"   LR: {epoch_metrics['learning_rate']:.6f}")
        logger.info(f"   Curriculum Stage: {epoch_metrics['curriculum_stage']:.2f}")
        
        for batch_idx, batch in enumerate(train_loader):
            images = batch['image'].to(self.device)
            targets_top = batch['target_top'].to(self.device)
            targets_bottom = batch['target_bottom'].to(self.device)
            input_lengths = batch['input_length'].to(self.device)
            target_lengths_top = batch['target_length_top'].to(self.device)
            target_lengths_bottom = batch['target_length_bottom'].to(self.device)
            
            # Forward pass with mixed precision
            self.optimizer.zero_grad()
            
            if self.config.use_amp and torch.cuda.is_available():
                with autocast():
                    output = self.model(images)
                    ctc_top = output['output_top']
                    ctc_bottom = output['output_bottom']
                    
                    loss_top = self.criterion(ctc_top, targets_top, 
                                            input_lengths, target_lengths_top)
                    loss_bottom = self.criterion(ctc_bottom, targets_bottom,
                                               input_lengths, target_lengths_bottom)
                    loss = 0.5 * loss_top + 0.5 * loss_bottom
                
                # Backward with loss scaling
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                             self.config.gradient_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(images)
                ctc_top = output['output_top']
                ctc_bottom = output['output_bottom']
                
                loss_top = self.criterion(ctc_top, targets_top, 
                                         input_lengths, target_lengths_top)
                loss_bottom = self.criterion(ctc_bottom, targets_bottom,
                                            input_lengths, target_lengths_bottom)
                loss = 0.5 * loss_top + 0.5 * loss_bottom
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                             self.config.gradient_clip)
                self.optimizer.step()
            
            # Update scheduler (LambdaLR tracks its own internal step counter)
            self.scheduler.step()
            self.global_step += 1
            
            epoch_losses.append({
                'loss': loss.item(),
                'loss_top': loss_top.item(),
                'loss_bottom': loss_bottom.item(),
            })
            
            # Logging
            if (batch_idx + 1) % 10 == 0:
                avg_loss = np.mean([l['loss'] for l in epoch_losses[-10:]])
                logger.info(f"   Batch {batch_idx+1}/{len(train_loader)} | Loss: {avg_loss:.4f}")
        
        # Compute epoch metrics
        epoch_metrics['loss'] = np.mean([l['loss'] for l in epoch_losses])
        epoch_metrics['loss_top'] = np.mean([l['loss_top'] for l in epoch_losses])
        epoch_metrics['loss_bottom'] = np.mean([l['loss_bottom'] for l in epoch_losses])
        
        logger.info(f"✅ Epoch {self.current_epoch} - Loss: {epoch_metrics['loss']:.4f}")
        
        # Validation
        if validation_loader is not None:
            val_metrics = self.validate(validation_loader)
            epoch_metrics.update(val_metrics)
            
            # Save best model
            if val_metrics['val_loss'] < self.best_loss:
                self.best_loss = val_metrics['val_loss']
                self.save_checkpoint(is_best=True)
                logger.info(f"🎯 New best model! Loss: {self.best_loss:.4f}")
        else:
            self.save_checkpoint()
        
        return epoch_metrics
    
    def validate(self, val_loader: DataLoader) -> Dict:
        """Validation loop"""
        self.model.eval()
        
        val_losses = []
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(self.device)
                targets_top = batch['target_top'].to(self.device)
                targets_bottom = batch['target_bottom'].to(self.device)
                input_lengths = batch['input_length'].to(self.device)
                target_lengths_top = batch['target_length_top'].to(self.device)
                target_lengths_bottom = batch['target_length_bottom'].to(self.device)
                
                output = self.model(images)
                ctc_top = output['output_top']
                ctc_bottom = output['output_bottom']
                
                loss_top = self.criterion(ctc_top, targets_top,
                                         input_lengths, target_lengths_top)
                loss_bottom = self.criterion(ctc_bottom, targets_bottom,
                                            input_lengths, target_lengths_bottom)
                loss = 0.5 * loss_top + 0.5 * loss_bottom
                
                val_losses.append(loss.item())
        
        val_loss = np.mean(val_losses)
        
        return {
            'val_loss': val_loss,
        }
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'global_step': self.global_step,
            'config': self.config.to_dict(),
        }
        
        if is_best:
            path = self.checkpoint_dir / 'best_model.pt'
        else:
            path = self.checkpoint_dir / f'checkpoint_epoch_{self.current_epoch}.pt'
        
        torch.save(checkpoint, path)
        logger.info(f"💾 Saved checkpoint: {path}")
    
    def load_checkpoint(self, path: str):
        """Load checkpoint"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)  # noqa: S-torch-load — trusted local checkpoints only
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_loss = checkpoint['best_loss']
        self.global_step = checkpoint['global_step']
        logger.info(f"✅ Loaded checkpoint from {path}")


# Example training script
"""
from app.models.lprv3 import LPRv3
from app.services.training.plate_generator import DatasetGenerator

# Generate dataset
gen = DatasetGenerator()
gen.generate_dataset(num_images=200000)

# Create model
model = LPRv3(num_classes=256)

# Create trainer
config = TrainingConfig(
    batch_size=32,
    num_epochs=100,
    learning_rate=0.001,
    use_curriculum=True,
)

trainer = LPRTrainer(model, config)

# Create dataloaders (implement LPRDataset first)
# train_loader = DataLoader(...)
# val_loader = DataLoader(...)

# Train
for epoch in range(config.num_epochs):
    metrics = trainer.train_epoch(train_loader, val_loader)
    print(metrics)

# Export for TensorRT
model.switch_to_deploy()
model.export_onnx('lprv3_fortress.onnx')
"""
