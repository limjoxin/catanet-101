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
from utils.dataset_utils import DatasetCataract1k, debug_label_files
from utils.logging_utils import timeSince
import glob
from sklearn.metrics import confusion_matrix
from torchvision.transforms import Compose, RandomResizedCrop, RandomVerticalFlip, RandomHorizontalFlip, ToPILImage, \
    ToTensor, Resize


def main(output_folder, log, basepath):
    # specify videos of surgeons for training and validation
    config = {'train': {}, 'val': {}, 'data': {}}
    config["train"]['batch_size'] = 50
    config["train"]['epochs'] = 5
    config["train"]['weighted_loss'] = True
    config['train']['sub_epoch_validation'] = 100
    config['train']['learning_rate'] = 0.00001
    config["val"]['batch_size'] = 150
    config['input_size'] = [224, 224]
    config['data']['base_path'] = basepath
    config['train']['min_frames_per_phase'] = 30

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # specify if we should use a GPU (cuda) or only the CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_devices = torch.cuda.device_count()

    # --- training params
    n_step_classes = 13
    training_phases = ['train', 'val']
    img_transform = {}
    img_transform['train'] = Compose([ToPILImage(), RandomHorizontalFlip(), RandomVerticalFlip(),
                                      RandomResizedCrop(size=config['input_size'][0], scale=(0.4,1.0), ratio=(1.0,1.0)),
                                      ToTensor()])
    img_transform['val'] = Compose([ToPILImage(), Resize(config['input_size']), ToTensor()])

    # --- logging
    if log:
        run = wandb.init(project='cataract_rsd', group='catnet')
        run.config.data = config['data']['base_path']
        run.name = run.id
    # --- glob data set
    dataLoader = {}
    underrepresented_classes = [0, 1, 6, 7, 8, 9]

    for phase in training_phases:
        data_folders = sorted(glob.glob(os.path.join(config['data']['base_path'], phase, '*')))
        labels = sorted(glob.glob(os.path.join(config['data']['base_path'], phase, '**', '*.csv')))

        # Only use balanced sampling and enhanced transformations for training
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

    output_model_name = os.path.join(output_folder, 'catRSDNet_CNN.pth')

    print('start training... ')
    # --- model
    base_model = CatRSDNet()
    model = base_model.cnn

    if num_devices > 1:
        model = nn.DataParallel(model).to(device)
    else:
        model = model.to(device)

    # Add weight decay to optimizer
    optim = torch.optim.Adam(
        model.parameters(),
        lr=config['train']['learning_rate'],
        weight_decay=0.01  # L2 regularization
    )

    # --- optimizer
    max_grad_norm = 1.0

    # debugging label files
    print("Analyzing label files before computing weights...")
    label_sum = debug_label_files(config['data']['base_path'], n_step_classes)

    # --- loss
    # loss function
    if config['train']['weighted_loss']:
        loss_weights = compute_balanced_weights(label_sum).to(device)
    else:
        loss_weights = None

    # Add label smoothing to loss function
    criterion = nn.CrossEntropyLoss(weight=loss_weights)

    # --- training
    best_loss_on_test = np.inf
    start_time = time.time()
    stop_epoch = config['train']['epochs']

    for epoch in range(stop_epoch):
        #zero out epoch based performance variables
        all_loss_train = torch.zeros(0).to(device)
        model.train()  # Set model to training mode

        for ii, (img, labels) in enumerate(dataLoader['train']):
            img = img.to(device)  # input data
            step_label = labels[:, 0].long().to(device)
            with torch.set_grad_enabled(True):
                prediction = model(img)
                loss = criterion(prediction, step_label)
                # update weights
                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optim.step()

            all_loss_train = torch.cat((all_loss_train, loss.detach().view(1, -1)))

            # compute sub-epoch validation loss for early stopping
            if ii % config['train']['sub_epoch_validation'] == 0:
                model.eval()
                with torch.no_grad():
                    val_subepoch_loss = torch.zeros(0).to(device)
                    conf_mat = np.zeros((13, 13))

                    for jj, (img, label) in enumerate(dataLoader['val']):  # for each of the batches
                        img = img.to(device)  # input data
                        step_label = label[:, 0].long().to(device)
                        prediction = model(img)
                        loss = criterion(prediction, step_label)
                        val_subepoch_loss = torch.cat((val_subepoch_loss, loss.detach().view(1, -1)))
                        hard_prediction = torch.argmax(prediction.detach(), dim=1).cpu().numpy()
                        conf_mat += confusion_matrix(step_label.cpu().numpy(), hard_prediction,
                                                            labels=np.arange(13))
                    
                # compute metrics
                val_subepoch_loss = val_subepoch_loss.cpu().numpy().mean()
                print('val loss: {0:.4f}'.format(val_subepoch_loss), end='')
                if log:
                    wandb.log({'/val/loss': val_subepoch_loss})

                if val_subepoch_loss < best_loss_on_test:
                    # if current loss is the best we've seen, save model state
                    if num_devices > 1:
                        state_dict = model.module.state_dict()
                    else:
                        state_dict = model.state_dict()
                    best_loss_on_test = val_subepoch_loss
                    print('  **')
                    state = {'epoch': epoch + 1,
                             'model_dict': state_dict
                             }

                    torch.save(state, output_model_name)
                else:
                    print('')
                model.train()

        all_loss_train = all_loss_train.cpu().numpy().mean()
        if log:
            wandb.log({'epoch': epoch, '/train/loss': all_loss_train})

        log_text = '%s ([%d/%d] %d%%), train loss: %.4f' %\
                   (timeSince(start_time, (epoch+1) / stop_epoch),
                    epoch + 1, stop_epoch , (epoch + 1) / stop_epoch * 100,
                    all_loss_train)
        print(log_text)

    print('...finished training')

def compute_balanced_weights(label_sum):
    """
    Compute class weights with numerical stability in mind:
    - Non-empty classes get weights inversely proportional to their count
    - Empty classes get a tiny weight (near-zero but not exactly zero)
    - Weights are normalized considering only non-empty classes
    """
    epsilon = 1e-7

    # Create mask for non-empty classes
    non_empty_mask = label_sum > 0
    
    # Initialize weights array with minimum weight
    weights = np.full_like(label_sum, epsilon, dtype=float)
    
    # Compute weights only for non-empty classes
    weights[non_empty_mask] = 1.0 / (label_sum[non_empty_mask] + epsilon)
    
    # Normalize non-zero weights
    num_active_classes = non_empty_mask.sum()
    non_zero_sum = weights[non_empty_mask].sum()
    weights[non_empty_mask] *= (num_active_classes / non_zero_sum)
    
    print("\nClass weights:")
    for i in range(len(weights)):
        print(f"Class {i}: {weights[i]:.8f} ({int(label_sum[i])} examples)")
            
    return torch.tensor(weights).float()


# def analyze_label_files(base_path, n_classes):
#     print("\nAnalyzing training data labels...")
#     label_sum = np.zeros(n_classes)
#     file_count = 0
    
#     # Look through all CSV files
#     for fname_label in glob.glob(os.path.join(base_path, 'train', '**', '*.csv')):
#         file_count += 1
#         # Load labels and print first few for inspection
#         labels = np.genfromtxt(fname_label, delimiter=',', skip_header=1)[:, 1]
#         if file_count == 1:  # Print sample from first file
#             print(f"\nSample labels from {os.path.basename(fname_label)}:")
#             print(f"First 10 labels: {labels[:10]}")
#             print(f"Label range: {labels.min()} to {labels.max()}")
        
#         # Count labels
#         for l in range(n_classes):
#             count = np.sum(labels==l)
#             label_sum[l] += count
            
#         # Print per-file statistics occasionally
#         if file_count % 100 == 0:
#             print(f"Processed {file_count} files...")
                
#     # Print final analysis
#     print(f"\nFound {file_count} label files")
#     print("\nClass distribution:")
#     for i in range(n_classes):
#         print(f"Class {i}: {int(label_sum[i])} examples")
        
#     if 0 in label_sum:
#         print("\nWARNING: Found classes with zero examples:")
#         empty_classes = np.where(label_sum == 0)[0]
#         print(f"Empty classes: {empty_classes}")
    


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
        help='if true log with wandb.'
    )
    parser.add_argument(
        '--basepath',
        type=str,
        default='data/cataract101',
        help='path to data.'
    )
    args = parser.parse_args()

    main(output_folder=args.out, log=args.log, basepath=args.basepath)