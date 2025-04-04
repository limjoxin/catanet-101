import torch
from torch import nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, ReduceLROnPlateau
import os
import argparse
import numpy as np
import sys
import time
import wandb
from torch.utils.data import DataLoader
import glob
from models.catRSDNet import CatRSDNet
from utils.dataset_utils import DatasetCataract1k
from utils.logging_utils import timeSince
from torchvision import transforms
from sklearn.metrics import confusion_matrix
import random
import warnings
import numpy as np
from numpy._core.multiarray import scalar as numpy_scalar
from models.catRSDNet import CatRSDNet
import logging

float32_dtype = np.dtype('float32').__class__
torch.serialization.add_safe_globals([float32_dtype])
logging.getLogger("wandb").setLevel(logging.WARNING)


class FocalLoss(nn.Module):
    """
    Focal Loss implementation for handling class imbalance more effectively.
    Applies a modulating factor to the standard cross-entropy loss.
    """
    def __init__(self, weight=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target, 
                                 weight=self.weight, 
                                 reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def conf2metrics(conf_mat):
    """ Confusion matrix to performance metrics conversion """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        precision = np.diag(conf_mat) / np.sum(conf_mat, axis=0)
    precision[np.isnan(precision)] = 0.0

    recall = np.diag(conf_mat) / np.sum(conf_mat, axis=1)
    recall[np.isnan(recall)] = 0.0

    class_totals = np.sum(conf_mat, axis=1)
    empty_classes = np.where(class_totals == 0)[0]
    if len(empty_classes) > 0:
        print(f"Warning: No examples found for classes: {empty_classes}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f1 = 2 * (precision * recall) / (precision + recall)
    f1[np.isnan(f1)] = 0.0

    accuracy = np.trace(conf_mat) / np.sum(conf_mat)
    return precision, recall, f1, accuracy


def compute_class_weights(label_counts):
    eps = 1e-6
    n_samples = np.sum(label_counts)
    class_frequencies = label_counts / n_samples
    inverse_frequencies = 1.0 / (class_frequencies + eps)
    normalized_weights = inverse_frequencies / np.mean(inverse_frequencies)
    
    # Print detailed diagnostic information
    print("\nClass weighting diagnostics:")
    print("Class | Samples | % of Data |  Weight  | Weighted %")
    print("-" * 60)
    
    for i in range(len(label_counts)):
        percentage = class_frequencies[i] * 100
        weighted_percentage = (normalized_weights[i] * label_counts[i]) / np.sum(normalized_weights * label_counts) * 100
        print(f"{i:5} | {int(label_counts[i]):7} | {percentage:8.2f}% | {normalized_weights[i]:8.3f} | {weighted_percentage:8.2f}%")
    
    return torch.tensor(normalized_weights).float()


def main(output_folder, log, pretrained_model):
    # Enhanced configuration with additional parameters
    config = {'train': {}, 'val': {}, 'data': {}}
    config["train"]['batch_size'] = 32
    config["train"]['epochs'] = [50, 15, 25]
    config["train"]["learning_rate"] = [0.0003, 0.0001, 0.00005]
    config['train']['weighted_loss'] = True
    config['train']['use_focal_loss'] = True  # New option for focal loss
    config['train']['focal_gamma'] = 1.0
    config['train']['use_mixed_precision'] = True
    config['train']['early_stopping_patience'] = 15
    config['train']['min_epochs'] = 20
    config["val"]['batch_size'] = 32
    config["pretrained_model"] = pretrained_model
    config["data"]["base_path"] = 'data/cataract1k'
    config["train"]["sequence"] = ['train_rnn', 'train_partial_cnn', 'train_all']
    config['train']['window'] = 32
    config['input_size'] = 224
    config['val']['test_every'] = 1
    config['train']['weight_decay'] = 1e-4  # Weight decay for regularization

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_model_name = os.path.join(output_folder, 'catRSDNet.pth')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_devices = torch.cuda.device_count()

    # Initialize model
    n_step_classes = 13
    model = CatRSDNet()

    # Load the pretrained CNN model
    if pretrained_model and os.path.exists(pretrained_model):
        device_map = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        try:
            checkpoint = torch.load(pretrained_model, map_location=device_map, weights_only=False)
            model.cnn.load_state_dict(checkpoint['model_dict'])
        except Exception as e:
            print(f"Error loading pretrained CNN model: {e}")
            print("Starting with random weights")
    else:
        print("No pretrained model provided. Starting with random weights.")

    model.set_cnn_as_feature_extractor()
    model = model.to(device)

    # Set up training phases
    training_phases = ['train', 'val']
    validation_phases = ['val']

    # Initialize WandB logging
    if log:
        run = wandb.init(project='cataract_rsd', group='catnet')
        run.name = run.id

    output_model_name = os.path.join(output_folder, 'catRSDNet.pth')

    # Image transformations
    img_transform = transforms.Compose([
                                        transforms.ToPILImage(),
                                        transforms.Resize(config['input_size']),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
                                        ])
    
    sequences_path = {key:{} for key in training_phases}
    sequences_path['train']['label'] = sorted(glob.glob(os.path.join(config['data']['base_path'], 'train','**','*.csv')))
    sequences_path['val']['label'] = sorted(glob.glob(os.path.join(config['data']['base_path'], 'val', '**', '*.csv')))
    sequences_path['train']['video'] = sorted(glob.glob(os.path.join(config['data']['base_path'], 'train', '*/')))
    sequences_path['val']['video'] = sorted(glob.glob(os.path.join(config['data']['base_path'], 'val', '*/')))

    # Mixed precision training
    if config['train']['use_mixed_precision'] and torch.cuda.is_available():
        scaler = GradScaler()
        print("Using mixed precision training")
    else:
        scaler = None

    # --- Loss and Optimizer
    if config['train']['weighted_loss']:
        label_sum = np.zeros(n_step_classes)
        for fname_label in sequences_path['train']['label']:
            labels = np.genfromtxt(fname_label, delimiter=',', skip_header=1)[:, 0]
            for l in range(n_step_classes):
                label_sum[l] += np.sum(labels==l)
        loss_weights = compute_class_weights(label_sum).to(device)
    else:
        loss_weights = None
    
    # Print class distribution
    print("\nClass distribution in training data:")
    for i in range(len(label_sum)):
        percentage = (label_sum[i] / label_sum.sum()) * 100
        print(f"Class {i}: {int(label_sum[i])} samples ({percentage:.2f}%)")

    # Step criterion: regular cross-entropy and focal loss
    if config['train']['use_focal_loss']:
        step_criterion = FocalLoss(weight=loss_weights, gamma=config['train']['focal_gamma'])
        print(f"Using Focal Loss with gamma={config['train']['focal_gamma']}")
    else:
        step_criterion = nn.CrossEntropyLoss(weight=loss_weights)
        print("Using Cross Entropy Loss")
    
    rsd_criterion = nn.L1Loss()

    training_steps = config['train']['sequence']
    remaining_steps = training_steps.copy()
    print(' start training... ')
    print(' training with sequence:', training_steps)

    start_time = time.time()
    non_improving_val_counter = 0
    features = {}
    
    for step_count, training_step in enumerate(training_steps):
        print(f"\n{'='*80}")
        print(f"Starting {training_step} phase (Phase {step_count+1}/{len(training_steps)})")
        print(f"{'='*80}")
    
        # Always load the best model from the previous phase
        if step_count > 0 and os.path.exists(output_model_name):
            try:
                print(f"Loading model from previous phase: {output_model_name}")
                checkpoint = torch.load(output_model_name, map_location=device, weights_only=False)
                model.load_state_dict(checkpoint['model_dict'])
                print("Successfully loaded model from previous phase")
            except Exception as e:
                print(f"Error loading model from previous phase: {e}")
                print("Continuing with current model state")
        elif step_count > 0:
            print(f"No previous model found at {output_model_name}")
            print("Starting this phase with current model state")
        
        # Initialize tracking variables for this phase
        best_loss_on_val = np.inf
        best_f1_on_val = 0
        non_improving_val_counter = 0
        stop_epoch = config['train']['epochs'][step_count]

        if training_step == 'train_rnn':
            # Pre-compute CNN features
            if len(features) == 0:
                model.eval()   
                for phase in training_phases:
                    sequences = list(zip(sequences_path[phase]['label'], 
                                        sequences_path[phase]['video']))
                    
                    for ii, (label_path, input_path) in enumerate(sequences):
                        # For training phase, use balanced sampling
                        if phase == 'train':
                            data = DatasetCataract1k(
                                [input_path], 
                                [label_path], 
                                img_transform=img_transform,
                                min_frames_per_phase=20,
                                balanced_sampling=True
                            )
                        else:
                            data = DatasetCataract1k(
                                [input_path], 
                                [label_path], 
                                img_transform=img_transform
                            )
                        
                        # Determine the feature dimensions
                        with torch.no_grad():
                            sample_X, _ = data[0]
                            sample_X = sample_X.unsqueeze(0).float().to(device)
                            feature_dim = model.cnn(sample_X).shape[1]
                        
                        # Calculate total number of frames in this video
                        total_frames = len(data)
                        print(f"    Total frames: {total_frames}, feature dimensions: {feature_dim}")
                    
                        # Pre-allocate numpy array of the right size
                        features[input_path] = np.zeros((total_frames, feature_dim), dtype=np.float32)
                        
                        # Process in smaller batches
                        batch_size = 128
                        dataloader = DataLoader(data, batch_size=batch_size, shuffle=(phase=='train'), num_workers=4, pin_memory=True)
                        current_idx = 0
                        for i, (X, _) in enumerate(dataloader):
                            with torch.no_grad():
                                batch_size_actual = X.shape[0]
                                batch_features = model.cnn(X.float().to(device)).cpu().numpy()
                                features[input_path][current_idx:current_idx + batch_size_actual] = batch_features
                                current_idx += batch_size_actual
                    
                        # Verify expected number of frames
                        assert current_idx == total_frames, f"{current_idx} / {total_frames} frames processed"
                        
                        # Free up memory
                        torch.cuda.empty_cache()
                        
                    print("CNN feature extraction complete.")
                
            # RNN-only training
            model.freeze_cnn(True)
            model.freeze_rnn(False)
            print("Frozen CNN, training only RNN")
            
            # Create optimizer
            trainable_params = [{
                'params': model.rnn.parameters(), 
                'lr': config['train']['learning_rate'][step_count]
            }]
            
        elif training_step == 'train_partial_cnn':
            features = {}

            model.freeze_early_cnn_layers(True)
            model.freeze_late_cnn_layers(False)
            model.freeze_rnn(False)
            
            # Create optimizer
            trainable_params = [
                {'params': model.rnn.parameters(), 
                'lr': config['train']['learning_rate'][step_count]},
                {'params': CatRSDNet.get_trainable_cnn_params(model),
                'lr': config['train']['learning_rate'][step_count] / 10}
            ]
            
        elif training_step == 'train_all':
            features = {}

            # Configure model for full training
            model.freeze_cnn(False)
            model.freeze_rnn(False)
            print("Training full model (CNN + RNN)")
            
            # Create optimizer
            trainable_params = [
                {'params': model.rnn.parameters(), 
                'lr': config['train']['learning_rate'][step_count]},
                {'params': model.cnn.parameters(),
                'lr': config['train']['learning_rate'][step_count] / 20}
            ]
            
        else:
            raise RuntimeError(f'Training step {training_step} not implemented')
    
        # Set initial_lr
        for param_group in trainable_params:
            param_group['initial_lr'] = param_group['lr']
        
        # Create optimizer with weight decay for regularization
        optim = torch.optim.AdamW(
            trainable_params,
            weight_decay=config['train']['weight_decay']
        )
        
        # Create learning rate scheduler for smoother training
        if training_step == 'train_rnn':
            scheduler = CosineAnnealingLR(
                optim, 
                T_max=stop_epoch, 
                eta_min=config['train']['learning_rate'][step_count] / 10
            )
        elif training_step == 'train_partial_cnn':
            # Use ReduceLROnPlateau to adapt learning rate based on validation performance
            scheduler = ReduceLROnPlateau(
                optim,
                mode='min',
                factor=0.5,
                patience=5,
            )
        elif training_step == 'train_all':
            # Cosine annealing for full model fine-tuning
            scheduler = CosineAnnealingLR(
                optim, 
                T_max=stop_epoch, 
                eta_min=config['train']['learning_rate'][step_count] / 10
            )

        # Main training loop for current phase
        for epoch in range(stop_epoch):
            epoch_start_time = time.time()
            
            # Initialize metrics tracking
            all_precision = {}
            average_precision = {}
            all_recall = {}
            average_recall = {}
            all_f1 = {}
            average_f1 = {}
            conf_mat = {key: np.zeros((n_step_classes, n_step_classes)) for key in validation_phases}
            all_loss = {key: torch.zeros(0).to(device) for key in training_phases}
            all_loss_step = {key: torch.zeros(0).to(device) for key in training_phases}
            all_loss_rsd = {key: torch.zeros(0).to(device) for key in training_phases}

            # Implement learning rate warmup for first few epochs
            warmup_epochs = 5
            if epoch < warmup_epochs:
                warmup_factor = 0.1 + 0.9 * (epoch / warmup_epochs)
                for param_group in optim.param_groups:
                    param_group['lr'] = param_group['initial_lr'] * warmup_factor
                print(f"Warmup epoch {epoch+1}/{warmup_epochs}, LR: {optim.param_groups[0]['lr']:.6f}")

            # Training/validation loop
            for phase in training_phases:
                sequences = list(zip(sequences_path[phase]['label'], sequences_path[phase]['video']))
                
                # Set model mode
                if phase == 'train':
                    model.train()
                    random.shuffle(sequences)
                    if training_step == 'train_rnn':
                        model.cnn.eval()
                else:
                    model.eval()

                # Process each video sequence
                for ii, (label_path, input_path) in enumerate(sequences):
                    if (training_step == 'train_rnn') | (training_step == 'train_fc'):
                        raw_data = np.genfromtxt(label_path, delimiter=',', skip_header=1)
                        if raw_data.shape[1] <= 3:
                            print(f"Warning: CSV file {os.path.basename(label_path)} has only {raw_data.shape[1]} columns")
                            # Use column 0 for class labels and column 2 for RSD if column 3 doesn't exist
                            last_col_idx = min(2, raw_data.shape[1] - 1)
                            raw_labels = raw_data[:, [0, last_col_idx]]
                        else:
                            # Original behavior - use columns 0 and 3
                            raw_labels = raw_data[:, [0, 3]]
                        sample_ratio = 59.94 / 2.5  # Original fps / New fps
                        
                        # Subsample the labels to match the CNN features
                        sampled_indices = np.arange(0, len(raw_labels), sample_ratio).astype(int)
                        sampled_labels = raw_labels[sampled_indices]

                        # Convert to tensors
                        label = torch.tensor(sampled_labels)
                        features_tensor = torch.tensor(features[input_path])

                        # Verify both have the same length
                        if label.shape[0] != features_tensor.shape[0]:
                            min_len = min(features_tensor.shape[0], label.shape[0])
                            features_tensor = features_tensor[:min_len]
                            label = label[:min_len]
                        
                        dataloader = [(features_tensor.unsqueeze(0), label)]
                        skip_features = True
                    else:
                        # Process full frames for CNN+RNN training
                        batch_size = config['train']['window']
                        data = DatasetCataract1k([input_path], [label_path], img_transform=img_transform)
                        dataloader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)
                        skip_features = False

                    # Process each batch in the sequence
                    for i, (X, y) in enumerate(dataloader):
                        if len(y.shape) > 2:
                            y = y.squeeze()

                        # Prepare labels
                        y_step = y[:, 0].long().to(device)
                        y_rsd = (y[:, 1]/60.0/25.0).float().to(device)  # Normalize RSD
                        X = X.float().to(device)

                        # Forward and backward passes
                        with torch.set_grad_enabled(phase == 'train'):
                            with autocast("cuda", enabled=scaler is not None):
                                stateful = (i > 0)
                                step_prediction, rsd_prediction = model.forwardRNN(
                                    X, 
                                    stateful=stateful,
                                    skip_features=skip_features
                                )
                                
                                # Calculate losses
                                loss_step = step_criterion(step_prediction, y_step)
                                rsd_prediction = rsd_prediction.squeeze(1)
                                loss_rsd = rsd_criterion(rsd_prediction, y_rsd)
                                rsd_weight = 0.2
                                loss = loss_step + rsd_weight * loss_rsd

                            # Optimization step
                            if phase == 'train':
                                optim.zero_grad()
                                
                                if scaler is not None:
                                    scaler.scale(loss).backward()
                                    scaler.step(optim)
                                    scaler.update()
                                else:
                                    loss.backward()
                                    optim.step()

                            # Track losses
                            all_loss[phase] = torch.cat((all_loss[phase], loss.detach().view(1, -1)))
                            all_loss_step[phase] = torch.cat((all_loss_step[phase], loss_step.detach().view(1, -1)))
                            all_loss_rsd[phase] = torch.cat((all_loss_rsd[phase], loss_rsd.detach().view(1, -1)))

                        # Calculate metrics for validation phases
                        if phase in validation_phases:
                            hard_prediction = torch.argmax(step_prediction.detach(), dim=1).cpu().numpy()
                            conf_mat[phase] += confusion_matrix(
                                y_step.cpu().numpy(), 
                                hard_prediction, 
                                labels=np.arange(n_step_classes)
                            )

                # Calculate mean losses for this phase
                all_loss[phase] = all_loss[phase].cpu().numpy().mean()
                all_loss_step[phase] = all_loss_step[phase].cpu().numpy().mean()
                all_loss_rsd[phase] = all_loss_rsd[phase].cpu().numpy().mean()

                # Calculate metrics for validation phases
                if phase in validation_phases:
                    precision, recall, f1, accuracy = conf2metrics(conf_mat[phase])
                    all_precision[phase] = precision
                    all_recall[phase] = recall
                    average_precision[phase] = np.mean(all_precision[phase])
                    average_recall[phase] = np.mean(all_recall[phase])
                    all_f1[phase] = f1
                    average_f1[phase] = np.mean(all_f1[phase])
                
                if i % 10 == 0 or phase == 'val':  # Only print every 10th batch in training or all validation batches
                    print(f"Batch {i} - {os.path.basename(input_path)}")
                    print(f"Prediction distribution: {torch.softmax(step_prediction, dim=1).mean(dim=0)}")
                    print(f"Unique predictions: {torch.argmax(step_prediction, dim=1).unique()}")

                # Log metrics to WandB
                if log:
                    log_epoch = step_count*stop_epoch + epoch
                    log_dict = {
                        'epoch': log_epoch, 
                        f'{phase}/loss': all_loss[phase],
                        f'{phase}/loss_rsd': all_loss_rsd[phase],
                        f'{phase}/loss_step': all_loss_step[phase],
                    }
                    
                    # Log learning rate
                    if phase == 'train':
                        log_dict['learning_rate'] = optim.param_groups[0]['lr']
                    
                    wandb.log(log_dict)
                    
                    # Log additional validation metrics
                    if ((epoch % config['val']['test_every']) == 0) & (phase in validation_phases):
                        val_metrics = {
                            'epoch': log_epoch, 
                            f'{phase}/precision': average_precision[phase],
                            f'{phase}/recall': average_recall[phase], 
                            f'{phase}/f1': average_f1[phase],
                            f'{phase}/accuracy': accuracy
                        }
                        
                        # Add per-class metrics
                        for i in range(len(all_precision[phase])):
                            val_metrics.update({
                                f'{phase}/precision_class_{i}': all_precision[phase][i],
                                f'{phase}/recall_class_{i}': all_recall[phase][i],
                                f'{phase}/f1_class_{i}': all_f1[phase][i]
                            })
                        
                        wandb.log(val_metrics)
                        
                        # Log confusion matrix
                        try:
                            wandb.log({
                                f'{phase}/confusion_matrix': wandb.plot.confusion_matrix(
                                    probs=None,
                                    y_true=np.repeat(np.arange(n_step_classes), conf_mat[phase].sum(axis=1).astype(int)),
                                    preds=np.concatenate([np.repeat(i, int(conf_mat[phase][j, i])) 
                                                        for j in range(n_step_classes) for i in range(n_step_classes)]),
                                    class_names=[f"Class {i}" for i in range(n_step_classes)]
                                )
                            })
                        except Exception as e:
                            print(f"Warning: Could not log confusion matrix: {e}")

            # Update learning rate scheduler
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(all_loss['val'])
            else:
                scheduler.step()
                
            # Print epoch results
            epoch_time = time.time() - epoch_start_time
            log_text = '%s ([%d/%d] %d%%), train loss: %.4f val loss: %.4f lp: %.4f (%.1f s/epoch)' % \
                        (timeSince(start_time, (epoch + 1) / stop_epoch),
                        epoch + 1, stop_epoch, (epoch + 1) / stop_epoch * 100,
                        all_loss['train'], all_loss['val'], all_loss_step['val'],
                        epoch_time)
            log_text += ' val precision: {0:.4f}, recall: {1:.4f}, f1: {2:.4f}'.format(
                average_precision['val'],
                average_recall['val'],
                average_f1['val'])
            
            # Check if current model is the best so far
            is_best_loss = all_loss["val"] < best_loss_on_val
            is_best_f1 = average_f1['val'] > best_f1_on_val
            
            # Get model state dict
            if num_devices > 1:
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            
            # Only start early stopping after minimum epochs
            if epoch >= config['train']['min_epochs']:
                if is_best_loss:
                    best_loss_on_val = all_loss["val"]
                    non_improving_val_counter = 0
                    print(log_text + '  ** (best loss)')
                elif is_best_f1:
                    best_f1_on_val = average_f1['val']
                    non_improving_val_counter = 0
                    print(log_text + '  ** (best F1)')
                else:
                    non_improving_val_counter += 1
                    print(log_text)
                    
                    # Check for early stopping
                    if non_improving_val_counter >= config['train']['early_stopping_patience']:
                        print(f"\nEarly stopping at epoch {epoch+1} due to no improvement for "
                                f"{config['train']['early_stopping_patience']} epochs")
                        break
            else:
                print(log_text)
            
            # Save model if it's the best so far (by loss or F1 score)
            if is_best_loss or is_best_f1:
                state = {
                    'epoch': epoch + 1,
                    'model_dict': state_dict,
                    'remaining_steps': remaining_steps,
                    'val_loss': all_loss["val"],
                    'val_f1': average_f1['val'],
                    'train_loss': all_loss["train"],
                    'phase': training_step
                }
                
                torch.save(state, output_model_name)
                if log:
                    wandb.summary['best_epoch'] = epoch + 1
                    wandb.summary['best_loss_on_val'] = best_loss_on_val
                    wandb.summary['best_f1'] = best_f1_on_val
                
        # After completing current phase
        remaining_steps.pop(0)
        print(f"\nCompleted phase: {training_step}")
            
        print('Training complete!')
    
    # Save final model configuration
    if log:
        wandb.summary.update({
            'completed': True,
            'total_training_time': time.time() - start_time,
            'final_phase': training_steps[-1]
        })
        wandb.finish()


def str2bool(v):
    """Convert string to boolean."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    """The program's entry point."""
    script_dir = os.path.dirname(sys.argv[0])

    parser = argparse.ArgumentParser(description='Training RNN for Cataract Tool Detection')

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
        help='if true log with wandb.'
    )
    parser.add_argument(
        '--pretrained',
        type=str,
        help='path to pre-trained CNN for CatNet.'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility.'
    )
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Run main function
    try:
        main(output_folder=args.out, log=args.log, pretrained_model=args.pretrained)
    except Exception as e:
        print(f"Error occurred during training: {e}")
        import traceback
        traceback.print_exc()
        if 'wandb' in sys.modules and wandb.run is not None:
            wandb.finish(exit_code=1)