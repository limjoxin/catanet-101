import argparse
import glob
import logging
import os
import random
import sys
import time
import math

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from sklearn.metrics import confusion_matrix
import wandb

from models.catRSDNet import CombinedEnhancedModel
from utils.dataset_utils import DatasetCataract1k
from utils.logging_utils import timeSince
from torchvision import transforms
from sklearn.metrics import confusion_matrix
import random
import warnings
import numpy as np
from numpy._core.multiarray import scalar as numpy_scalar
import logging

float32_dtype = np.dtype('float32').__class__
torch.serialization.add_safe_globals([float32_dtype])
logging.getLogger("wandb").setLevel(logging.WARNING)

import torch
import torch.nn.functional as F
from torch import nn


class MultiClassFocalLoss(nn.Module):
    """
    Multi-class Focal Loss.
    FL = αₜ (1 – pₜ)^γ ⋅ (−log pₜ)

    Args:
        alpha: 1D Tensor of shape [num_classes], giving weight αᵢ for each class.
               If None, all αᵢ = 1.
        gamma: focusing parameter γ ≥ 0.
        reduction: 'none' | 'mean' | 'sum'
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        if alpha is not None:
            # ensure it's a float tensor
            self.alpha = alpha.float()
        else:
            self.alpha = None
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        """
        inputs: [batch, C] logits (raw scores)
        targets: [batch] long with class indices in [0..C-1]
        """
        # Compute per-sample cross-entropy: ce = –log pₜ
        ce_loss = F.cross_entropy(inputs, targets, weight=None, reduction='none')
        # Compute pₜ = exp(–ce)
        pt = torch.exp(-ce_loss)

        # α factor for each target
        if self.alpha is not None:
            # gather αᵢ for each sample’s true class
            alpha_t = self.alpha.to(inputs.device)[targets]
        else:
            alpha_t = 1.0

        # Focal loss
        loss = alpha_t * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss



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


# Learning rate tracking function
def log_lr(optimizer, epoch, iteration, wandb_log=False):
    """Track and log learning rates for all parameter groups"""
    for i, param_group in enumerate(optimizer.param_groups):
        current_lr = param_group['lr']
        group_name = f"group_{i+1}" if i > 0 else "rnn"
        
        print(f"Epoch {epoch+1} - LR for {group_name}: {current_lr:.6f}")
        
        if wandb_log:
            wandb.log({
                f'learning_rate/{group_name}': current_lr,
                'epoch': epoch,
                'iteration': iteration
            })


def init_rnn_weights(m):
    if isinstance(m, (nn.GRU, nn.LSTM, nn.RNN)):
        for name, param in m.named_parameters():
            if 'weight_ih' in name: nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name: nn.init.orthogonal_(param.data)
            elif 'bias' in name: param.data.fill_(0)
    elif hasattr(m, 'reset_parameters'):
        m.reset_parameters()


def _save_epoch_results_csv(frame_idxs, true_labels, preds, output_folder, epoch, tag):
    """
    Dump a per-frame results CSV.
      - frame_idxs, true_labels, preds: lists of equal length
      - output_folder: where to write
      - epoch: integer epoch number
      - tag: e.g. 'last' or 'best' to distinguish files
    """
    df = pd.DataFrame({
        'frame_index': frame_idxs,
        'true_phase':   true_labels,
        'pred_phase':   preds,
    })
    fname = f'epoch_{epoch:02d}_{tag}_results.csv'
    path  = os.path.join(output_folder, fname)
    df.to_csv(path, index=False)
    print(f"→ Saved results CSV to {path}")


def main(output_folder, log, pretrained_model):
    # Enhanced configuration with additional parameters
    config = {'train': {}, 'val': {}, 'data': {}}
    config["train"]['batch_size'] = 64
    config["train"]['epochs'] = [5, 10]
    config["train"]["learning_rate"] = [0.003, 0.0001]
    config['train']['weighted_loss'] = True
    config['train']['use_focal_loss'] = True  # New option for focal loss
    config['train']['focal_gamma'] = 1.0
    config['train']['use_mixed_precision'] = True
    config['train']['early_stopping_patience'] = 15
    config['train']['min_epochs'] = 20
    config["val"]['batch_size'] = 32
    config["pretrained_model"] = pretrained_model
    config["data"]["base_path"] = 'data/cataract1k'
    config["train"]["sequence"] = ['train_rnn', "train_all"]
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
    best_ckpt_name   = os.path.join(output_folder, "catRSDNet_best.pth")   # best‑metric model
    last_ckpt_name   = os.path.join(output_folder, "catRSDNet_last.pth")   # last‑epoch model

    best_f1_global   = -1.0                      # highest val‑F1 seen so far
    best_loss_global = float("inf")              # lowest val‑loss seen so far
    
    n_step_classes = 13
    model = CombinedEnhancedModel()

    # Load pretrained CNN if available
    if pretrained_model and os.path.exists(pretrained_model):
        device_map = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        try:
            checkpoint = torch.load(pretrained_model, map_location=device_map, weights_only=False)
            model.cnn.load_state_dict(checkpoint['model_dict'])
        except Exception as e:
            print(f"Error loading pretrained CNN model: {e}")
            print("Starting with random weights")
    else:
        print("No pretrained model found, using random initialization")

    # Always initialize the RNN
    model.rnn.apply(init_rnn_weights)
    model.set_cnn_as_feature_extractor()
    model.freeze_cnn(True)
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
    img_transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomResizedCrop(config['input_size'], scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
    ])
    img_transform_val = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(config['input_size']),
        transforms.CenterCrop(config['input_size']),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
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
        step_criterion = MultiClassFocalLoss(alpha=loss_weights, gamma=config['train']['focal_gamma'])
        print(f"Using MultiClass Focal Loss with gamma={config['train']['focal_gamma']}")
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
                                img_transform=img_transform_train,  # Use training transforms
                                min_frames_per_phase=1,
                                balanced_sampling=True,
                                return_index=True
                            )
                        else:
                            data = DatasetCataract1k(
                                [input_path], 
                                [label_path], 
                                img_transform=img_transform_val,
                                return_index=True
                            )
                        
                        # Determine the feature dimensions
                        with torch.no_grad():
                            sample_X = data[0][0]
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
                        for i, batch in enumerate(dataloader):
                            X = batch[0]
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
            
        elif training_step == 'train_all':
            # Load pre-computed features if possible
            features = {}
            feature_files_missing = False
            
            for phase in training_phases:
                sequences = list(zip(sequences_path[phase]['label'], 
                                    sequences_path[phase]['video']))
                
                for ii, (label_path, input_path) in enumerate(sequences):
                    feature_file = f"{os.path.basename(input_path).replace('/', '_')}_features.npy"
                    if os.path.exists(feature_file):
                        features[input_path] = np.load(feature_file)
                    else:
                        feature_files_missing = True
                        print(f"WARNING: Feature file missing for {os.path.basename(input_path)}")
            else:  # train_all
                # If all feature files are available, use the pre-computed features
                if not feature_files_missing and len(features) > 0:
                    print("Using pre-computed features for train_all phase")
                    # Configure model for using pre-computed features but still train CNN
                    model.freeze_cnn(False)
                    model.freeze_rnn(False)
                    
                    # Create optimizer with Adam
                    trainable_params = [
                        {'params': model.rnn.parameters(), 
                        'lr': config['train']['learning_rate'][step_count] * 0.1},
                        {'params': model.cnn.parameters(),
                        'lr': config['train']['learning_rate'][step_count] * 0.005}
                    ]
                else:
                    features = {}
                    model.freeze_cnn(False)
                    model.freeze_rnn(False)
                    print("Training full model (CNN + RNN)")
                    
                    # Create optimizer with Adam
                    trainable_params = [
                        {'params': model.rnn.parameters(), 
                        'lr': config['train']['learning_rate'][step_count] * 0.1},
                        {'params': model.cnn.parameters(),
                        'lr': config['train']['learning_rate'][step_count] * 0.005}
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
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optim,
                T_max=stop_epoch,
                eta_min=1e-6,
                )
        elif training_step == 'train_all':
            max_lr_values = []
            for i, param_group in enumerate(optim.param_groups):
                if i == 0:
                    max_lr_values.append(config['train']['learning_rate'][step_count] * 0.1)
                else:
                    max_lr_values.append(config['train']['learning_rate'][step_count] * 0.005)
            
            total_batches = 0
            for label_path in sequences_path['train']['label']:
                labels = np.genfromtxt(label_path, delimiter=',', skip_header=1)
                n_frames = labels.shape[0]
                # how many window-sized mini-batches that video creates:
                batches = math.ceil(n_frames / config['train']['window'])
                total_batches += batches

            updates_per_epoch = total_batches // config['train']['grad_accum_steps']

            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optim,
                max_lr=max_lr_values,
                epochs=stop_epoch,
                steps_per_epoch=updates_per_epoch,
                pct_start=0.3,
                anneal_strategy='cos',
                div_factor=25.0,
                final_div_factor=10000.0
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

            all_frame_idxs   = []
            all_true_labels  = []
            all_preds        = []


            log_lr(optim, epoch, 0, log)

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
                        if phase == 'train':
                            raw = np.genfromtxt(label_path, delimiter=',', skip_header=1)
                            labels = raw[:, 0].astype(int)
                            data = DatasetCataract1k(
                                [input_path],
                                [label_path],
                                img_transform=current_transform,
                                min_frames_per_phase=config['train']['min_frames_per_phase'],
                                balanced_sampling=False,
                                return_index=True
                            )
                            # Make sure sampler and dataset agree on N
                            dataset_len = len(data)
                            class_counts = np.bincount(labels, minlength=n_step_classes)
                            sample_weights = 1.0 / (class_counts[labels] + 1e-6)
                            sample_weights = sample_weights[:dataset_len]
                            sampler = WeightedRandomSampler(
                                weights=sample_weights,
                                num_samples=dataset_len,
                                replacement=True
                            )
                            dataloader = DataLoader(
                                data,
                                batch_size=config['train']['batch_size'],
                                sampler=sampler,
                                num_workers=4,
                                pin_memory=True
                            )
                            skip_features = False
                    else:
                        # Process full frames for CNN+RNN training
                        batch_size = config['train']['window']
                        data = DatasetCataract1k([input_path], [label_path], img_transform=img_transform)
                        dataloader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)
                        skip_features = False

                    # Track a batch counter for the current sequence
                    batch_counter = 0
                    
                    # Process each batch in the sequence
                    for i, batch in enumerate(dataloader):
                        if len(batch) == 3:
                            X, y, idxs = batch
                        else:
                            X, y = batch
                            idxs = torch.arange(X.size(0))
                        X      = X.float().to(device)
                        y_step = y[:, 0].long().to(device)
                        # y_rsd  = (y[:, 1] / (60.0 * 25.0)).float().to(device)
                        y_rsd_raw = (y[:, 1] / (60.0 * 25.0)).float().to(device)
                        y_rsd     = torch.log1p(y_rsd_raw)

                        with torch.set_grad_enabled(phase == 'train'):
                            with autocast("cuda", enabled=(scaler is not None)):
                                # Forward RNN
                                step_pred, rsd_pred = model.forwardRNN(
                                    X,
                                    stateful=(i > 0),
                                    skip_features=skip_features
                                )

                                # Reduce step_pred to [batch, classes]
                                if step_pred.dim() == 3:
                                    step_pred = step_pred[:, -1, :]
                                elif step_pred.dim() == 1:
                                    step_pred = step_pred.unsqueeze(0)

                                # Reduce rsd_pred similarly
                                if rsd_pred.dim() == 3:
                                    rsd_pred = rsd_pred[:, -1, :]
                                elif rsd_pred.dim() == 2 and rsd_pred.size(1) == 1:
                                    rsd_pred = rsd_pred.squeeze(1)

                                # Compute losses
                                loss_step = step_criterion(step_pred, y_step)
                                # loss_rsd = rsd_criterion(rsd_pred, y_rsd)
                                loss_rsd  = nn.L1Loss()(rsd_pred, y_rsd)
                                loss = loss_step + 0.2 * loss_rsd

                            # Backprop + optimizer step
                            if phase == 'train':
                                scaled = loss / config['train']['grad_accum_steps']
                                if scaler is not None:
                                    scaler.scale(scaled).backward()
                                    if (i + 1) % config['train']['grad_accum_steps'] == 0:
                                        scaler.unscale_(optim)
                                        torch.nn.utils.clip_grad_norm_(model.parameters(), config['train']['clip_value'])
                                        scaler.step(optim); scaler.update(); optim.zero_grad()
                                else:
                                    scaled.backward()
                                    if (i + 1) % config['train']['grad_accum_steps'] == 0:
                                        torch.nn.utils.clip_grad_norm_(model.parameters(), config['train']['clip_value'])
                                        optim.step(); optim.zero_grad()

                                # If using OneCycleLR, step scheduler here:
                                if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR) and (i + 1) % config['train']['grad_accum_steps'] == 0:
                                    scheduler.step()


                        # Stash predictions for CSV
                        hard = step_pred.argmax(dim=1).cpu().tolist()
                        all_frame_idxs  .extend(idxs.cpu().tolist())
                        all_true_labels .extend(y_step.cpu().tolist())
                        all_preds       .extend(hard)

                        # Track losses & confusion for logging/metrics…
                        all_loss[phase]     = torch.cat((all_loss[phase], loss.detach().view(1, -1)))
                        all_loss_step[phase]= torch.cat((all_loss_step[phase], loss_step.detach().view(1, -1)))
                        all_loss_rsd[phase] = torch.cat((all_loss_rsd[phase], loss_rsd.detach().view(1, -1)))
                        if phase in validation_phases:
                            hp = step_pred.argmax(dim=1).cpu().numpy()
                            conf_mat[phase] += confusion_matrix(y_step.cpu().numpy(), hp, labels=np.arange(n_step_classes))

                        # only debug the very first epoch and a couple of batches
                        if epoch == 0 and i < 2 and phase == 'train':
                            # labels in this batch
                            ys = y_step.cpu().numpy()

                        # Track losses
                        all_loss[phase] = torch.cat((all_loss[phase], loss.detach().view(1, -1)))
                        all_loss_step[phase] = torch.cat((all_loss_step[phase], loss_step.detach().view(1, -1)))
                        all_loss_rsd[phase] = torch.cat((all_loss_rsd[phase], loss_rsd.detach().view(1, -1)))

                        # Calculate metrics for validation phases
                        if phase in validation_phases:
                            hard_prediction = torch.argmax(step_pred.detach(), dim=1).cpu().numpy()
                            
                            conf_mat[phase] += confusion_matrix(
                                y_step.cpu().numpy(), 
                                hard_prediction, 
                                labels=np.arange(n_step_classes)
                            )

                # Calculate mean losses for this phase
                if all_loss[phase].numel() > 0:  # Check if tensor is not empty
                    all_loss[phase] = all_loss[phase].cpu().numpy().mean()
                else:
                    all_loss[phase] = float('nan')
                    
                if all_loss_step[phase].numel() > 0:
                    all_loss_step[phase] = all_loss_step[phase].cpu().numpy().mean()
                else:
                    all_loss_step[phase] = float('nan')
                    
                if all_loss_rsd[phase].numel() > 0:
                    all_loss_rsd[phase] = all_loss_rsd[phase].cpu().numpy().mean()
                else:
                    all_loss_rsd[phase] = float('nan')

                # Calculate metrics for validation phases
                if phase in validation_phases and any(conf_mat[phase].sum(axis=1) > 0):  # Ensure we have valid data
                    precision, recall, f1, accuracy = conf2metrics(conf_mat[phase])
                    all_precision[phase] = precision
                    all_recall[phase] = recall
                    average_precision[phase] = np.mean(all_precision[phase])
                    average_recall[phase] = np.mean(all_recall[phase])
                    all_f1[phase] = f1
                    average_f1[phase] = np.mean(all_f1[phase])
                    
                    # DEBUG: Print detailed per-class metrics at end of phase
                    print("\n[DEBUG] End of phase per-class metrics:")
                    for c in range(n_step_classes):
                        print(f"Class {c}: Precision={precision[c]:.4f}, Recall={recall[c]:.4f}, F1={f1[c]:.4f}")
                else:
                    # Initialize with default values if no data
                    if phase in validation_phases:
                        all_precision[phase] = np.zeros(n_step_classes)
                        all_recall[phase] = np.zeros(n_step_classes)
                        all_f1[phase] = np.zeros(n_step_classes)
                        average_precision[phase] = 0
                        average_recall[phase] = 0
                        average_f1[phase] = 0
                
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
                            f'{phase}/accuracy': accuracy if 'accuracy' in locals() else 0
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
            elif not isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                # For other schedulers like CosineAnnealingLR, step per epoch
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
            is_best_loss = all_loss["val"] < best_loss_on_val and not np.isnan(all_loss["val"])
            is_best_f1 = average_f1['val'] > best_f1_on_val and not np.isnan(average_f1['val'])
            
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
            
            # Save model state
            state = {
                'epoch': epoch + 1,
                'model_dict': state_dict,
                'phase': training_step,
                'val_loss': all_loss["val"],
                'val_f1': average_f1['val'],
                'train_loss': all_loss["train"],
            }

            # Always keep a "last" checkpoint (for resuming)
            torch.save(state, last_ckpt_name)
            _save_epoch_results_csv(
                all_frame_idxs, all_true_labels, all_preds,
                output_folder, epoch+1, tag='last'
            )

            # Track best metrics
            better_f1 = average_f1['val'] > best_f1_global and not np.isnan(average_f1['val'])
            better_loss = all_loss["val"] < best_loss_global and not np.isnan(all_loss["val"])

            if better_f1 or better_loss:             
                torch.save(state, best_ckpt_name)          # overwrite best
                _save_epoch_results_csv(
                    all_frame_idxs, all_true_labels, all_preds,
                    output_folder, epoch+1, tag='best'
                )
                best_f1_global = max(best_f1_global, average_f1['val']) if not np.isnan(average_f1['val']) else best_f1_global
                best_loss_global = min(best_loss_global, all_loss["val"]) if not np.isnan(all_loss["val"]) else best_loss_global
                print(f"  ↳  Saved new {'best F1' if better_f1 else 'best loss'}: {best_ckpt_name} (…)")
                torch.save(state, output_model_name)

                if log:
                    wandb.summary['best_epoch'] = epoch + 1
                    wandb.summary['best_loss_on_val'] = best_loss_on_val
                    wandb.summary['best_f1'] = best_f1_on_val
                
        # Update remaining steps
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