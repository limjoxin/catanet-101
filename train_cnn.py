import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
import os
import argparse
import numpy as np
import sys
import time
import wandb
import math
from models.catRSDNet import CatRSDNet, EnhancedCatRSDNet
from utils.dataset_utils import DatasetCataract1k
from utils.logging_utils import timeSince
import glob
from sklearn.metrics import confusion_matrix
from torchvision.transforms import Compose, RandomResizedCrop, RandomVerticalFlip, RandomHorizontalFlip, ToPILImage, \
    ToTensor, Resize, ColorJitter, RandomRotation, Normalize
from torchvision.models import resnet50, ResNet50_Weights


# Learning rate tracker to monitor changes
class LRTracker:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.initial_lr = optimizer.param_groups[0]['lr']
        self.current_lr = self.initial_lr
        print(f"Initial learning rate: {self.initial_lr}")
        
    def after_scheduler_step(self, epoch, iteration, log=False):
        # Check if learning rate has changed
        new_lr = self.optimizer.param_groups[0]['lr']
        if new_lr != self.current_lr:
            print(f"Learning rate changed: {self.current_lr} -> {new_lr} (epoch {epoch}, iteration {iteration})")
            if log:
                wandb.log({'learning_rate': new_lr, 'epoch': epoch, 'iteration': iteration})
            self.current_lr = new_lr


# Focal Loss for addressing class imbalance
class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# Cosine learning rate scheduler with warmup
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr=0.0):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function
    between the initial lr and min_lr, after a warmup period.
    """
    def lr_lambda(current_step):
        # Fix for zero learning rate at step 0
        if current_step == 0:
            return 1.0  # Return 1.0 for the first step to keep original LR
            
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
            
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        
        # Make sure we don't divide by zero
        initial_lr = optimizer.param_groups[0]['lr']
        if initial_lr == 0:
            return cosine_decay  # Avoid division by zero
            
        return max(min_lr / initial_lr, cosine_decay)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# Mixup data augmentation for improving generalization
def mixup_data(x, y, alpha=0.2, device='cuda'):
    """Applies Mixup augmentation to the batch."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Applies Mixup criterion to the predictions."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def compute_label_distribution(train_path, n_classes):
    """
    Calculate class distribution directly from label files
    """
    label_sum = np.zeros(n_classes)
    total_labels = 0
    file_count = 0
    
    for fname_label in glob.glob(os.path.join(train_path, '**', '*.csv')):
        try:
            file_count += 1
            data = np.genfromtxt(fname_label, delimiter=',', skip_header=1)
            
            if data.size == 0:
                continue
                
            if data.ndim == 1:
                data = data.reshape(1, -1)
                
            if data.shape[1] < 2:
                continue
                
            labels_col0 = data[:, 0]
            labels_col1 = data[:, 1]
            
            col0_valid = np.all((labels_col0 >= 0) & (labels_col0 < n_classes))
            col1_valid = np.all((labels_col1 >= 0) & (labels_col1 < n_classes))
            
            if col0_valid:
                labels = labels_col0
            elif col1_valid:
                labels = labels_col1
            else:
                continue
            
            total_labels += len(labels)
            for l in range(n_classes):
                count = np.sum(labels == l)
                label_sum[l] += count
                
        except Exception as e:
            continue
    
    print("\nClass distribution:")
    for i in range(n_classes):
        percentage = 0 if total_labels == 0 else (label_sum[i] / total_labels * 100)
        print(f"Class {i}: {int(label_sum[i])} examples ({percentage:.2f}%)")
    
    return label_sum


def compute_balanced_weights(label_sum):
    """
    Compute class weights with numerical stability in mind:
    - Non-empty classes get weights inversely proportional to their count
    - Empty classes get a tiny weight (near-zero but not exactly zero)
    - Weights are normalized considering only non-empty classes
    """
    epsilon = 1e-7
    non_empty_mask = label_sum > 0

    weights = np.full_like(label_sum, epsilon, dtype=float)
    weights[non_empty_mask] = 1.0 / (label_sum[non_empty_mask] + epsilon)

    num_active_classes = non_empty_mask.sum()
    non_zero_sum = weights[non_empty_mask].sum()
    weights[non_empty_mask] *= (num_active_classes / non_zero_sum)
    
    print("\nClass weights:")
    for i in range(len(weights)):
        print(f"Class {i}: {weights[i]:.8f} ({int(label_sum[i])} examples)")
            
    return torch.tensor(weights).float()


def compute_class_metrics(conf_mat):
    """
    Compute per-class precision, recall, and F1-score from a confusion matrix
    
    Args:
        conf_mat: numpy array of shape (n_classes, n_classes)
    
    Returns:
        Dictionary with precision, recall, and F1 for each class
    """
    n_classes = conf_mat.shape[0]
    metrics = {}
    
    # Compute true positives, false positives, false negatives for each class
    for i in range(n_classes):
        # True positives: diagonal elements
        tp = conf_mat[i, i]
        
        # False positives: sum of column i excluding the diagonal element
        fp = np.sum(conf_mat[:, i]) - tp
        
        # False negatives: sum of row i excluding the diagonal element
        fn = np.sum(conf_mat[i, :]) - tp
        
        # Calculate metrics with handling for division by zero
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        metrics[i] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': tp + fn  # Total number of true instances for this class
        }
    
    return metrics


def train_with_gradient_accumulation(model, dataloader, optimizer, criterion, device, 
                                   accumulation_steps=4, mixup_alpha=0.2, use_mixup=True, 
                                   max_grad_norm=1.0, scheduler=None):
    """Training with gradient accumulation and mixup"""
    model.train()
    total_loss = 0
    step_count = 0
    optimizer.zero_grad()  # Zero gradients at the beginning
    
    for i, (img, labels) in enumerate(dataloader):
        img = img.to(device)
        step_label = labels[:, 0].long().to(device)
        
        # Apply mixup if enabled
        if use_mixup:
            img, y_a, y_b, lam = mixup_data(img, step_label, alpha=mixup_alpha, device=device)
            
        # Forward pass
        prediction = model(img)
        
        # Compute loss (with or without mixup)
        if use_mixup:
            loss = mixup_criterion(criterion, prediction, y_a, y_b, lam)
        else:
            loss = criterion(prediction, step_label)
        
        # Normalize loss by accumulation steps
        loss = loss / accumulation_steps
        loss.backward()
        
        # Update weights after accumulating gradients
        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(dataloader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            
            # Step the scheduler after each optimizer step
            if scheduler is not None:
                scheduler.step()
                
            step_count += 1
            
        total_loss += loss.item() * accumulation_steps
        
    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device, n_classes=13):
    """Comprehensive evaluation function"""
    model.eval()
    total_loss = 0
    conf_mat = np.zeros((n_classes, n_classes))
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for img, labels in dataloader:
            img = img.to(device)
            step_label = labels[:, 0].long().to(device)
            
            prediction = model(img)
            loss = criterion(prediction, step_label)
            total_loss += loss.item()
            
            # Get predictions
            preds = torch.argmax(prediction, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(step_label.cpu().numpy())
    
    # Concatenate all predictions and labels
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    # Compute confusion matrix
    conf_mat = confusion_matrix(all_labels, all_preds, labels=np.arange(n_classes))
    
    # Calculate metrics
    class_metrics = compute_class_metrics(conf_mat)
    avg_f1 = np.mean([m['f1'] for m in class_metrics.values()])
    
    return total_loss / len(dataloader), conf_mat, class_metrics, avg_f1


def main(output_folder, log, basepath):
    # Define configuration
    config = {'train': {}, 'val': {}, 'data': {}}
    config["train"]['batch_size'] = 64
    config["train"]['epochs'] = 15
    config["train"]['weighted_loss'] = True
    config['train']['sub_epoch_validation'] = 100
    config['train']['learning_rate'] = 0.001  # Updated to 0.001 for RMSprop
    config["val"]['batch_size'] = 128
    config['input_size'] = [224, 224]
    config['data']['base_path'] = basepath
    config['train']['min_frames_per_phase'] = 30
    
    config['train']['use_enhanced_model'] = True
    config['train']['gradient_accumulation_steps'] = 4
    config['train']['warmup_steps'] = 500
    config['train']['use_mixup'] = True
    config['train']['mixup_alpha'] = 0.2
    config['train']['dropout_rate'] = 0.3
    config['train']['use_focal_loss'] = True
    config['train']['focal_gamma'] = 2.0
    config['train']['early_stopping_patience'] = 5

    # RMSprop optimizer settings
    config['train']['optimizer'] = 'rmsprop'
    config['train']['momentum'] = 0.0  # RMSprop can use momentum
    config['train']['weight_decay'] = 0.0  # L2 regularization
    config['train']['alpha'] = 0.99  # RMSprop smoothing constant
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_devices = torch.cuda.device_count()
    
    n_step_classes = 13
    training_phases = ['train', 'val']
    
    # Enhanced image transformations
    img_transform = {}
    img_transform['train'] = Compose([
        ToPILImage(),
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.3),
        RandomResizedCrop(size=config['input_size'][0], scale=(0.4, 1.0), ratio=(0.9, 1.1)),
        ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        RandomRotation(20),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_transform['val'] = Compose([
        ToPILImage(),
        Resize(config['input_size']),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    if log:
        run = wandb.init(project='cataract_rsd', group='enhanced_catnet_rmsprop')
        run.config.update(config)
        run.name = run.id
    
    # Calculate class distribution
    print("Calculating class distribution for balanced weights...")
    label_sum = compute_label_distribution(os.path.join(config['data']['base_path'], 'train'), n_step_classes)
    
    # Data loading
    dataLoader = {}
    underrepresented_classes = [0, 1, 6, 7, 8, 9]
    
    for phase in training_phases:
        data_folders = sorted(glob.glob(os.path.join(config['data']['base_path'], phase, '*')))
        labels = sorted(glob.glob(os.path.join(config['data']['base_path'], phase, '**', '*.csv')))
        
        if phase == 'train':
            dataset = DatasetCataract1k(
                data_folders,
                label_files=labels,
                img_transform=img_transform[phase],
                min_frames_per_phase=config['train']['min_frames_per_phase'],
                balanced_sampling=True,
                training_mode=True,
                underrepresented_classes=underrepresented_classes
            )
        else:
            dataset = DatasetCataract1k(
                data_folders,
                label_files=labels,
                img_transform=img_transform[phase],
                training_mode=False
            )
        
        dataLoader[phase] = DataLoader(
            dataset,
            batch_size=config[phase]['batch_size'],
            shuffle=(phase == 'train'),
            num_workers=4,
            pin_memory=True
        )
    
    # Model setup
    if config['train']['use_enhanced_model']:
        output_model_name = os.path.join(output_folder, 'enhanced_catRSDNet_rmsprop.pth')
        print('Start training enhanced model with RMSprop optimizer...')
        model = EnhancedCatRSDNet(n_classes=n_step_classes, dropout_rate=config['train']['dropout_rate'])
    else:
        output_model_name = os.path.join(output_folder, 'catRSDNet_rmsprop.pth')
        print('Start training original model with RMSprop optimizer...')
        base_model = CatRSDNet()
        model = base_model.cnn
    
    if num_devices > 1:
        model = nn.DataParallel(model).to(device)
    else:
        model = model.to(device)
    
    # Setup RMSprop optimizer
    optimizer = torch.optim.RMSprop(
        model.parameters(),
        lr=config['train']['learning_rate'],
        alpha=config['train']['alpha'],
        momentum=config['train']['momentum'],
        weight_decay=config['train']['weight_decay']
    )
    
    # Learning rate scheduler
    total_steps = len(dataLoader['train']) * config['train']['epochs'] // config['train']['gradient_accumulation_steps']
    warmup_steps = config['train']['warmup_steps']
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
        min_lr=1e-6
    )
    
    lr_tracker = LRTracker(optimizer)
    
    # Loss function
    if config['train']['weighted_loss']:
        loss_weights = compute_balanced_weights(label_sum).to(device)
    else:
        loss_weights = None
    
    if config['train']['use_focal_loss']:
        criterion = FocalLoss(weight=loss_weights, gamma=config['train']['focal_gamma'])
    else:
        criterion = nn.CrossEntropyLoss(weight=loss_weights)
    
    # Training setup
    best_val_metric = float('inf')
    best_val_f1 = 0.0
    early_stopping_counter = 0
    start_time = time.time()
    
    # Training loop
    for epoch in range(config['train']['epochs']):
        train_loss = train_with_gradient_accumulation(
            model, 
            dataLoader['train'],
            optimizer,
            criterion,
            device,
            accumulation_steps=config['train']['gradient_accumulation_steps'],
            mixup_alpha=config['train']['mixup_alpha'],
            use_mixup=config['train']['use_mixup'],
            scheduler=scheduler
        )
        
        lr_tracker.after_scheduler_step(epoch, 0, log)
            
        val_loss, conf_mat, class_metrics, avg_f1 = evaluate(
            model, 
            dataLoader['val'], 
            criterion, 
            device, 
            n_classes=n_step_classes
        )
        
        # Logging
        log_text = '\n%s ([%d/%d] %d%%), train loss: %.4f, val loss: %.4f, avg F1: %.4f' % (
            timeSince(start_time, (epoch + 1) / config['train']['epochs']),
            epoch + 1, config['train']['epochs'],
            (epoch + 1) / config['train']['epochs'] * 100,
            train_loss, val_loss, avg_f1
        )
        print(log_text)
        
        # Log per-class metrics
        print("\nPer-class performance:")
        worst_classes = []
        for class_idx, metric in class_metrics.items():
            print(f"Class {class_idx}: Precision: {metric['precision']:.4f}, "
                  f"Recall: {metric['recall']:.4f}, F1: {metric['f1']:.4f}")
            worst_classes.append((class_idx, metric['f1']))
        
        # Find worst classes
        worst_classes = sorted(worst_classes, key=lambda x: x[1])[:3]
        print("\nWorst performing classes by F1 score:")
        for class_idx, f1 in worst_classes:
            print(f"Class {class_idx}: F1 = {f1:.4f}")
        
        if log:
            wandb.log({
                'epoch': epoch,
                'train/loss': train_loss,
                'val/loss': val_loss,
                'val/avg_f1': avg_f1,
                'learning_rate': optimizer.param_groups[0]['lr']
            })
            
            # Also log per-class metrics
            for class_idx, metric in class_metrics.items():
                wandb.log({
                    f'val/class_{class_idx}_precision': metric['precision'],
                    f'val/class_{class_idx}_recall': metric['recall'],
                    f'val/class_{class_idx}_f1': metric['f1']
                })
        
        # Model saving - can loss as metric
        if val_loss < best_val_metric:
            best_val_metric = val_loss
            early_stopping_counter = 0
            
            # Save model
            if num_devices > 1:
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
                
            state = {
                'epoch': epoch + 1,
                'model_dict': state_dict,
                'optimizer_dict': optimizer.state_dict(),
                'class_metrics': class_metrics,
                'best_val_metric': best_val_metric,
                'avg_f1': avg_f1
            }
            
            torch.save(state, output_model_name)
            print(f"Model saved with val_loss: {best_val_metric:.4f}")
        else:
            early_stopping_counter += 1
            print(f"EarlyStopping counter: {early_stopping_counter} out of {config['train']['early_stopping_patience']}")
            
            if early_stopping_counter >= config['train']['early_stopping_patience']:
                print("Early stopping triggered!")
                break
    
    print('...finished training')
    
    # Final evaluation on validation set
    print("\n--- Final Evaluation on Validation Set ---")
    model.load_state_dict(torch.load(output_model_name, weights_only=False)['model_dict'])
    final_loss, final_conf_mat, final_metrics, final_f1 = evaluate(
        model, 
        dataLoader['val'], 
        criterion, 
        device, 
        n_classes=n_step_classes
    )
    print(f"Final validation loss: {final_loss:.4f}, F1: {final_f1:.4f}")
    
    # Print final confusion matrix (normalized)
    print("\nConfusion Matrix (Normalized by True Labels):")
    row_sums = final_conf_mat.sum(axis=1, keepdims=True)
    norm_conf_mat = np.zeros_like(final_conf_mat, dtype=float)
    np.divide(final_conf_mat, row_sums, out=norm_conf_mat, where=row_sums!=0)
    
    # Print matrix in a readable format
    print("    " + " ".join(f"{i:5d}" for i in range(n_step_classes)))
    for i in range(n_step_classes):
        print(f"{i:2d} |" + " ".join(f"{x:5.2f}" for x in norm_conf_mat[i]))
    
    # Highlight classes with most confusion
    confusion_pairs = []
    for i in range(n_step_classes):
        for j in range(n_step_classes):
            if i != j and norm_conf_mat[i, j] > 0.1:
                confusion_pairs.append((i, j, norm_conf_mat[i, j]))
    
    confusion_pairs.sort(key=lambda x: x[2], reverse=True)
    print("\nMost Confused Class Pairs:")
    for true_class, pred_class, rate in confusion_pairs[:5]:
        print(f"Class {true_class} confused with Class {pred_class}: {rate:.2f}")


if __name__ == "__main__":
    """The program's entry point."""
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    script_dir = os.path.dirname(sys.argv[0])

    parser = argparse.ArgumentParser(description='Training CNN for Cataract Tool Detection')

    parser.add_argument(
        '--out',
        type=str,
        default='output',
        help='Path to output file, ignored if log is true (use wandb directory instead).'
    )
    parser.add_argument(
        '--log',
        type=str2bool,
        default='False',
        help='If true log with wandb.'
    )
    parser.add_argument(
        '--basepath',
        type=str,
        default='data/cataract101',
        help='Path to data.'
    )
    parser.add_argument(
        '--enhanced',
        type=str2bool,
        default='True',
        help='Use enhanced model (ResNet backbone) instead of original CNN.'
    )
    
    args = parser.parse_args()
    
    # Run training with RMSprop optimizer
    main(output_folder=args.out, log=args.log, basepath=args.basepath)