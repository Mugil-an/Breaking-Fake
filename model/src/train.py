"""Training script for Breaking-Fake ViT model."""
import os
import sys
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import timm
from tqdm import tqdm
import numpy as np

# Add parent to path so we can import data module
sys.path.insert(0, str(Path(__file__).parent))
from data import get_dataloaders


class BreakingFakeTrainer:
    """Trainer class for Breaking-Fake model."""
    
    def __init__(
        self,
        model,
        device,
        learning_rate=1e-4,
        weight_decay=1e-4,
        checkpoint_dir=None,
    ):
        self.model = model
        self.device = device
        # Use a path relative to this script so saving is stable regardless of CWD
        # Save checkpoints into the artifacts folder so all saved models are together
        if checkpoint_dir is None:
            self.checkpoint_dir = Path(__file__).resolve().parent.parent / 'artifacts'
        else:
            self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=10)
        
        # put logs under the model folder next to checkpoints
        logs_dir = Path(__file__).resolve().parent.parent / 'logs'
        logs_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(logs_dir))
        self.global_step = 0
    
    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        pbar = tqdm(train_loader, desc="Training")
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            logits = self.model(images)
            loss = self.criterion(logits, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Metrics
            with torch.no_grad():
                preds = logits.argmax(dim=1)
                correct = (preds == labels).sum().item()
            
            total_loss += loss.item()
            total_correct += correct
            total_samples += images.size(0)
            
            # Logging
            self.writer.add_scalar('train/loss', loss.item(), self.global_step)
            self.global_step += 1
            
            pbar.set_postfix({
                'loss': f'{total_loss / (total_samples / images.size(0)):.4f}',
                'acc': f'{total_correct / total_samples:.4f}',
            })
        
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = total_correct / total_samples
        return epoch_loss, epoch_acc
    
    def eval_epoch(self, val_loader):
        """Evaluate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validating")
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                
                preds = logits.argmax(dim=1)
                correct = (preds == labels).sum().item()
                
                total_loss += loss.item()
                total_correct += correct
                total_samples += images.size(0)
                
                pbar.set_postfix({'loss': f'{total_loss / (total_samples / images.size(0)):.4f}'})
        
        epoch_loss = total_loss / len(val_loader)
        epoch_acc = total_correct / total_samples
        return epoch_loss, epoch_acc

    def test_epoch(self, test_loader):
        """Run the trained model on the test split."""
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            pbar = tqdm(test_loader, desc="Testing")
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(images)
                loss = self.criterion(logits, labels)

                preds = logits.argmax(dim=1)
                correct = (preds == labels).sum().item()

                total_loss += loss.item()
                total_correct += correct
                total_samples += images.size(0)

                pbar.set_postfix({'loss': f'{total_loss / (total_samples / images.size(0)):.4f}'})

        epoch_loss = total_loss / len(test_loader)
        epoch_acc = total_correct / total_samples
        return epoch_loss, epoch_acc
    
    def save_checkpoint(self, epoch, val_acc, is_best=False):
        """Save model checkpoint.
        
        Args:
            epoch: Current epoch number
            val_acc: Validation accuracy for this checkpoint
            is_best: Whether this is the best checkpoint so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': val_acc,
            'best_val_acc': getattr(self, 'best_val_acc', val_acc),
            'scheduler_state_dict': getattr(self.scheduler, 'state_dict', lambda: None)(),
            'global_step': self.global_step,
        }
        
        # Save epoch-specific checkpoint (keeps full history)
        epoch_path = self.checkpoint_dir / f'breaking_fake_model_epoch_{epoch:03d}.pth'
        torch.save(checkpoint, epoch_path)
        print(f"Checkpoint saved: {epoch_path}")
        
        # Save as best checkpoint if this is the best so far (safety: only overwrite on explicit best=True)
        if is_best:
            best_path = self.checkpoint_dir / 'breaking_fake_model_best.pth'
            torch.save(checkpoint, best_path)
            print(f"Best checkpoint saved: {best_path} (val_acc={val_acc:.4f})")

    def load_checkpoint(self, path):
        """Load checkpoint and restore model + optimizer + scheduler state.

        Returns: checkpoint dict
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        ck = torch.load(path, map_location='cpu')
        # load model
        self.model.load_state_dict(ck['model_state_dict'])
        # load optimizer
        if 'optimizer_state_dict' in ck:
            try:
                self.optimizer.load_state_dict(ck['optimizer_state_dict'])
            except ValueError:
                # move optimizer tensors to current device
                for state in self.optimizer.state.values():
                    for k, v in list(state.items()):
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(self.device)
                self.optimizer.load_state_dict(ck['optimizer_state_dict'])

        # load scheduler if available
        if 'scheduler_state_dict' in ck and ck['scheduler_state_dict'] is not None:
            try:
                self.scheduler.load_state_dict(ck['scheduler_state_dict'])
            except Exception:
                pass

        # restore helper attrs
        self.global_step = ck.get('global_step', getattr(self, 'global_step', 0))
        self.best_val_acc = ck.get('best_val_acc', ck.get('val_acc', 0.0))
        print(f"Loaded checkpoint {path} (epoch={ck.get('epoch')}, val_acc={ck.get('val_acc')})")
        return ck
    
    def train(self, train_loader, val_loader, num_epochs=10, start_epoch=0, best_val_acc=0.0):
        """Train for multiple epochs.

        Args:
            num_epochs: total number of epochs (like `--epochs`)
            start_epoch: epoch index to start from (0-based)
            best_val_acc: best validation accuracy so far
        """
        self.best_val_acc = best_val_acc

        for epoch in range(start_epoch, num_epochs):
            print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")
            
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.eval_epoch(val_loader)
            
            self.scheduler.step()
            
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
            
            self.writer.add_scalar('epoch/train_loss', train_loss, epoch)
            self.writer.add_scalar('epoch/train_acc', train_acc, epoch)
            self.writer.add_scalar('epoch/val_loss', val_loss, epoch)
            self.writer.add_scalar('epoch/val_acc', val_acc, epoch)
            
            # Save best model (only overwrites if truly better)
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint(epoch, val_acc, is_best=True)
            else:
                # Still save epoch checkpoint for history, but not as "best"
                self.save_checkpoint(epoch, val_acc, is_best=False)
        
        print(f"\nTraining complete. Best Val Acc: {best_val_acc:.4f}")
        self.writer.close()



def main():
    parser = argparse.ArgumentParser(description='Train Breaking-Fake model')
    parser.add_argument('--data-dir', default='data/raw', help='Path to data/raw folder')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num-workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--resume', default=None, help='Path to checkpoint or "auto" to resume from latest in artifacts')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Create model (ViT Base)
    model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=2)
    model = model.to(device)
    print(f"Model: ViT Base | Params: {sum(p.numel() for p in model.parameters()):,}")
    
    # Data: resolve data_dir (try provided path, then script-relative)
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        alt = Path(__file__).resolve().parent.parent / args.data_dir
        if alt.exists():
            data_dir = alt
        else:
            print(f"Warning: data dir {args.data_dir} not found. Tried {alt}.")

    print(f"\nLoading data from {data_dir}...")
    train_loader, val_loader, test_loader = get_dataloaders(
        str(data_dir),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    
    # Train
    trainer = BreakingFakeTrainer(model, device, learning_rate=args.lr)

    start_epoch = 0
    best_val_acc = 0.0
    # Handle resume
    if args.resume:
        # determine artifacts/checkpoint directory
        ck_dir = trainer.checkpoint_dir
        resume_path = None
        if args.resume == 'auto':
            # Try to load best checkpoint first, then latest epoch checkpoint
            best_path = ck_dir / 'breaking_fake_model_best.pth'
            if best_path.exists():
                resume_path = best_path
            else:
                # find latest epoch .pth in artifacts
                cands = list(ck_dir.glob('breaking_fake_model_epoch_*.pth'))
                if cands:
                    resume_path = max(cands, key=lambda p: p.stat().st_mtime)
        else:
            p = Path(args.resume)
            if not p.is_absolute():
                p = Path.cwd() / args.resume
            if p.exists():
                resume_path = p

        if resume_path is None:
            print(f"No checkpoint found to resume from (looked in {ck_dir}). Continuing from scratch.")
        else:
            ck = trainer.load_checkpoint(resume_path)
            start_epoch = ck.get('epoch', 0) + 1
            best_val_acc = ck.get('best_val_acc', ck.get('val_acc', 0.0))

    trainer.train(train_loader, val_loader, num_epochs=args.epochs, start_epoch=start_epoch, best_val_acc=best_val_acc)

    test_loss, test_acc = trainer.test_epoch(test_loader)
    print(f"Test Loss:  {test_loss:.4f} | Test Acc:  {test_acc:.4f}")
    
    # Export final model into script-relative artifacts folder
    artifacts_dir = Path(__file__).resolve().parent.parent / 'artifacts'
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    save_path = artifacts_dir / 'breaking_fake_vit.pth'
    torch.save(model.state_dict(), save_path)
    print(f"\nModel saved to {save_path}")


if __name__ == '__main__':
    main()
