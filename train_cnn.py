import torch
from torch import nn
from torch.utils.data import DataLoader
import os
import argparse
import numpy as np
import sys
import time
import wandb
from models.catRSDNet import CatRSDNet
from utils.dataset_utils import DatasetCataract101
from utils.logging_utils import timeSince
import glob
from sklearn.metrics import confusion_matrix
from torchvision.transforms import Compose, RandomResizedCrop, RandomVerticalFlip, RandomHorizontalFlip, ToPILImage, \
    ToTensor, Resize

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def create_transforms(input_size):
    return {
        'train': Compose([
            ToPILImage(),
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            RandomResizedCrop(size=input_size[0], scale=(0.4, 1.0), ratio=(1.0, 1.0)),
            ToTensor()
        ]),
        'val': Compose([
            ToPILImage(),
            Resize(input_size),
            ToTensor()
        ])
    }

def calculate_class_weights(base_path, n_classes):
    """Calculate class weights and analyze class distribution"""
    print("Analyzing class labels in:", base_path)
    label_files = glob.glob(os.path.join(base_path, 'train', '**', '*.csv'))
    
    # Store all unique labels we find
    all_labels = set()
    label_counts = {}
    
    for fname_label in label_files:
        try:
            # Read only the step column (first column)
            labels = np.genfromtxt(fname_label, delimiter=',', skip_header=1)
            if labels.size == 0:
                continue
                
            # Extract step column
            class_labels = labels[:, 0] if labels.ndim > 1 else labels
            
            # Only count valid labels
            valid_labels = class_labels[(class_labels >= 0) & (class_labels < n_classes)]
            unique_labels = np.unique(valid_labels)
            all_labels.update(unique_labels)
            
            # Count occurrences
            for label in unique_labels:
                count = np.sum(class_labels == label)
                if label in label_counts:
                    label_counts[label] += count
                else:
                    label_counts[label] = count
                    
        except Exception as e:
            print(f"Warning: Error processing {fname_label}: {str(e)}")
            continue
    
    # Report findings
    print("\nDetailed label distribution:")
    for label in sorted(all_labels):
        count = label_counts.get(label, 0)
        print(f"Label {label:2.0f}: {count:8.0f} samples")
    
    # Calculate weights if we have all classes represented
    if len(all_labels) == n_classes:
        total_samples = sum(label_counts.values())
        weights = torch.zeros(n_classes)
        for i in range(n_classes):
            if i in label_counts:
                weights[i] = total_samples / (n_classes * label_counts[i])
        return weights
    
    print(f"\nWarning: Not all {n_classes} classes are represented in the data")
    return None

def create_dataloaders(config, img_transform):
    """Create training and validation data loaders"""
    dataloaders = {}
    phases = ['train', 'val']
    
    for phase in phases:
        try:
            data_folders = sorted(glob.glob(os.path.join(config['data']['base_path'], phase, '*')))
            labels = sorted(glob.glob(os.path.join(config['data']['base_path'], phase, '**', '*.csv')))
            
            if not data_folders:
                raise ValueError(f"No data folders found for {phase} phase")
            if not labels:
                raise ValueError(f"No label files found for {phase} phase")
            
            dataset = DatasetCataract101(data_folders, img_transform=img_transform[phase], label_files=labels)
            
            dataloaders[phase] = DataLoader(
                dataset,
                batch_size=config[phase]['batch_size'],
                shuffle=(phase == 'train'),
                num_workers=4,
                pin_memory=True,
                drop_last=False
            )
            
        except Exception as e:
            print(f"Error creating {phase} dataloader: {str(e)}")
            raise
            
    return dataloaders

def filter_invalid_samples(batch_data, n_classes, device):
    """
    Filter out samples with invalid class labels.
    
    Parameters:
    -----------
    batch_data : tuple
        Batch containing (images, labels)
    n_classes : int
        Number of valid classes
    device : torch.device
        Device to put filtered tensors on
        
    Returns:
    --------
    tuple
        Filtered (images, labels) or (None, None) if all samples are invalid
    """
    img, step_label = batch_data[0], batch_data[1]
    
    # Find valid label indices
    valid_mask = (step_label >= 0) & (step_label < n_classes)
    
    if not valid_mask.any():
        return None, None
        
    # Filter the batch
    valid_img = img[valid_mask].to(device)
    valid_labels = step_label[valid_mask].long().to(device)
    
    return valid_img, valid_labels

def validate(model, val_loader, criterion, device, n_classes):
    """
    Validate model with invalid label handling.
    """
    model.eval()
    val_loss = torch.zeros(0).to(device)
    conf_mat = np.zeros((n_classes, n_classes))
    
    with torch.no_grad():
        for batch_data in val_loader:
            img, step_label = filter_invalid_samples(batch_data, n_classes, device)
            if img is None:
                continue
                
            prediction = model(img)
            loss = criterion(prediction, step_label)
            val_loss = torch.cat((val_loss, loss.detach().view(1, -1)))
            
            hard_prediction = torch.argmax(prediction.detach(), dim=1).cpu().numpy()
            conf_mat += confusion_matrix(
                step_label.cpu().numpy(),
                hard_prediction,
                labels=np.arange(n_classes)
            )
    
    return val_loss.cpu().numpy().mean(), conf_mat

def train_epoch(model, dataloader, criterion, optimizer, device, n_classes, config, best_loss_on_test, output_model_name, num_devices, log=False):
    """
    Train for one epoch with invalid label handling.
    Returns the average training loss and updated best_loss_on_test
    """
    model.train()
    all_loss_train = torch.zeros(0).to(device)
    
    for ii, batch_data in enumerate(dataloader['train']):
        try:
            # Filter invalid samples
            img, step_label = filter_invalid_samples(batch_data, n_classes, device)
            if img is None:
                print(f"Skipping batch {ii} - all labels invalid")
                continue
                
            # Forward pass
            with torch.set_grad_enabled(True):
                prediction = model(img)
                loss = criterion(prediction, step_label)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            all_loss_train = torch.cat((all_loss_train, loss.detach().view(1, -1)))
            
            # Sub-epoch validation
            if ii % config['train']['sub_epoch_validation'] == 0:
                val_loss, conf_mat = validate(model, dataloader['val'], criterion, device, n_classes)
                print(f'Validation loss: {val_loss:.4f}', end='')
                
                if log:
                    wandb.log({'/val/loss': val_loss})
                
                if val_loss < best_loss_on_test:
                    best_loss_on_test = val_loss
                    print('  ** New best model saved **')
                    state = {
                        'model_dict': model.module.state_dict() if num_devices > 1 else model.state_dict()
                    }
                    torch.save(state, output_model_name)
                else:
                    print('')
                
                model.train()
                
        except Exception as e:
            print(f"Error in training batch {ii}: {str(e)}")
            continue
            
    return all_loss_train.cpu().numpy().mean(), best_loss_on_test

def main(output_folder, log, basepath):
    """Main training function with improved error handling"""
    config = {
        'train': {
            'batch_size': 50,
            'epochs': 3,
            'weighted_loss': True,
            'sub_epoch_validation': 100,
            'learning_rate': 0.0001
        },
        'val': {
            'batch_size': 150
        },
        'input_size': [224, 224],
        'data': {
            'base_path': basepath
        }
    }
    
    os.makedirs(output_folder, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_devices = torch.cuda.device_count()
    print(f"Using device: {device} ({num_devices} GPUs available)")
    
    n_step_classes = 13
    img_transform = create_transforms(config['input_size'])
    dataloader = create_dataloaders(config, img_transform)
    
    if log:
        run = wandb.init(project='cataract_rsd', group='catnet')
        run.config.data = config['data']['base_path']
        run.name = run.id
    
    # Initialize model
    base_model = CatRSDNet(n_step_classes=n_step_classes)
    model = base_model.cnn
    
    if num_devices > 1:
        model = nn.DataParallel(model)
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['train']['learning_rate'])
    
    # Calculate class weights if needed
    loss_weights = calculate_class_weights(config['data']['base_path'], n_step_classes) if config['train']['weighted_loss'] else None
    loss_weights = loss_weights.to(device) if loss_weights is not None else None
    
    criterion = nn.CrossEntropyLoss(weight=loss_weights)
    
    best_loss_on_test = float('inf')
    start_time = time.time()
    output_model_name = os.path.join(output_folder, 'catRSDNet_CNN.pth')
    
    print('Starting training...')
    
    try:
        for epoch in range(config['train']['epochs']):
            epoch_loss, best_loss_on_test = train_epoch(
                model, dataloader, criterion, optimizer, device, n_step_classes, 
                config, best_loss_on_test, output_model_name, num_devices, log
            )
            
            if log:
                wandb.log({'epoch': epoch, '/train/loss': epoch_loss})
            
            log_text = '%s ([%d/%d] %d%%), train loss: %.4f' % (
                timeSince(start_time, (epoch + 1) / config['train']['epochs']),
                epoch + 1,
                config['train']['epochs'],
                (epoch + 1) / config['train']['epochs'] * 100,
                epoch_loss
            )
            print(log_text)
            
    except KeyboardInterrupt:
        print('\nTraining interrupted by user')
    except Exception as e:
        print(f'\nTraining stopped due to error: {str(e)}')
    finally:
        print('\nTraining finished')

if __name__ == "__main__":
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
        help='if true log with wandb.'
    )
    parser.add_argument(
        '--basepath',
        type=str,
        default='data/cataract1k',
        help='path to data.'
    )
    
    args = parser.parse_args()
    main(output_folder=args.out, log=args.log, basepath=args.basepath)