import torch
from torch import nn
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
import csv
from sklearn.metrics import confusion_matrix
import random
import warnings


def conf2metrics(conf_mat):
    """ Confusion matrix to performance metrics conversion """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        precision = np.diag(conf_mat) / np.sum(conf_mat, axis=0)
    precision[np.isnan(precision)] = 0.0

    recall = np.diag(conf_mat) / np.sum(conf_mat, axis=1)
    recall[np.isnan(recall)] = 0.0#

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


def validate_batch_dimensions(predictions, targets, sequence_name):
    """
    Comprehensive validation of tensor dimensions before loss calculation.
    
    Args:
        predictions: Model output tensor
        targets: Ground truth tensor
        sequence_name: String identifier for logging
    """
    if predictions.size(0) != targets.size(0):
        raise ValueError(
            f"Batch size mismatch in {sequence_name}:\n"
            f"Predictions shape: {predictions.shape}\n"
            f"Targets shape: {targets.shape}\n"
            f"This indicates a problem in the data pipeline."
        )

def main(output_folder, log, pretrained_model):
    config = {'train': {}, 'val': {}, 'data': {}}
    config["train"]['batch_size'] = 1
    config["train"]['epochs'] = [50, 10, 20]
    config["train"]["learning_rate"] = [0.001, 0.0001, 0.0005]
    config['train']['weighted_loss'] = True
    config["val"]['batch_size'] = 1
    config["pretrained_model"] = pretrained_model
    config["data"]["base_path"] = 'data/cataract1k'
    config["train"]["sequence"] = ['train_rnn', 'train_all', 'train_rnn']
    config['train']['window'] = 48
    config['input_size'] = 224
    config['val']['test_every'] = 1

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_devices = torch.cuda.device_count()

    # --- model
    n_step_classes = 13
    model = CatRSDNet()
    model.cnn.load_state_dict(torch.load(config['pretrained_model'])['model_dict'])
    model.set_cnn_as_feature_extractor()
    model = model.to(device)

    training_phases = ['train', 'val']
    validation_phases = ['val']

    if log:
        run = wandb.init(project='cataract_rsd', group='catnet')
        run.name = run.id

    output_model_name = os.path.join(output_folder, 'catRSDNet.pth')

    img_transform = transforms.Compose([transforms.ToPILImage(),
                                      transforms.Resize(config['input_size']),
                                      transforms.ToTensor()])
    
    sequences_path = {key:{} for key in training_phases}
    sequences_path['train']['label'] = sorted(glob.glob(os.path.join(config['data']['base_path'], 'train','**','*.csv')))
    sequences_path['val']['label'] = sorted(glob.glob(os.path.join(config['data']['base_path'], 'val', '**', '*.csv')))
    sequences_path['train']['video'] = sorted(glob.glob(os.path.join(config['data']['base_path'], 'train', '*/')))
    sequences_path['val']['video'] = sorted(glob.glob(os.path.join(config['data']['base_path'], 'val', '*/')))

    # --- Loss and Optimizer
    if config['train']['weighted_loss']:
        label_sum = np.zeros(n_step_classes)
        for fname_label in sequences_path['train']['label']:
            labels = np.genfromtxt(fname_label, delimiter=',', skip_header=1)[:, 0]
            for l in range(n_step_classes):
                label_sum[l] += np.sum(labels==l)
        # loss_weights = 1 / label_sum
        # loss_weights[label_sum == 0] = 0.0
        # loss_weights = torch.tensor(loss_weights / np.max(loss_weights)).float().to(device)
        loss_weights = compute_balanced_weights(label_sum).to(device)
    else:
        loss_weights = None

    step_criterion = nn.CrossEntropyLoss(weight=loss_weights)
    rsd_criterion = nn.L1Loss()

    training_steps = config['train']['sequence']
    remaining_steps = training_steps.copy()
    print(' start training... ')

    start_time = time.time()
    non_improving_val_counter = 0
    features = {}
    
    for step_count, training_step in enumerate(training_steps):
        print(training_step)
        if step_count > 0:
            checkpoint = torch.load(output_model_name, map_location=device, weights_only=True)
            model.load_state_dict(checkpoint['model_dict'])
        best_loss_on_val = np.Infinity
        stop_epoch = config['train']['epochs'][step_count]
        
        optim = torch.optim.Adam([{'params': model.rnn.parameters()},
                                {'params': model.cnn.parameters(), 'lr': config['train']['learning_rate'][step_count] / 20}],
                                lr=config['train']['learning_rate'][step_count])

        if training_step == 'train_rnn':
            if len(features) == 0:
                model.eval()
                sequences = list(zip(sequences_path['train']['label']+sequences_path['val']['label'],
                                   sequences_path['train']['video']+sequences_path['val']['video']))
                for ii, (label_path, input_path) in enumerate(sequences):
                    data = DatasetCataract1k([input_path], [label_path], img_transform=img_transform)
                    dataloader = DataLoader(data, batch_size=500, shuffle=False, num_workers=1, pin_memory=True)
                    features[input_path] = []
                    for i, (X, _) in enumerate(dataloader):
                        with torch.no_grad():
                            features[input_path].append(model.cnn(X.float().to(device)).cpu().numpy())
                    features[input_path] = np.concatenate(features[input_path])
            model.freeze_cnn(True)
            model.freeze_rnn(False)
        elif training_step == 'train_all':
            model.freeze_cnn(False)
            model.freeze_rnn(False)
            features = {}
        else:
            raise RuntimeError('training step {0} not implemented'.format(training_step))

        for epoch in range(stop_epoch):
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

            for phase in training_phases:
                sequences = list(zip(sequences_path[phase]['label'], sequences_path[phase]['video']))
                if phase == 'train':
                    model.train()
                    random.shuffle(sequences)
                    model.cnn.eval()
                else:
                    model.eval()

                for ii, (label_path, input_path) in enumerate(sequences):
                    if (training_step == 'train_rnn') | (training_step == 'train_fc'):
                        raw_labels = np.genfromtxt(label_path, delimiter=',', skip_header=1)[:, [0, 3]]
                        sample_ratio = 59.94 / 2.5  # Original fps / New fps
                        
                        # Subsample the labels to match the CNN features
                        sampled_indices = np.arange(0, len(raw_labels), sample_ratio).astype(int)
                        sampled_labels = raw_labels[sampled_indices]

                        # Convert to tensors
                        label = torch.tensor(sampled_labels)
                        features_tensor = torch.tensor(features[input_path])

                        # verify both have same length
                        if label.shape[0] != features_tensor.shape[0]:
                            min_len = min(features_tensor.shape[0], label.shape[0])
                            features_tensor = features_tensor[:min_len]
                            label = label[:min_len]
                        
                        dataloader = [(features_tensor.unsqueeze(0), label)]
                        skip_features = True
                    else:
                        batch_size = config['train']['window']
                        data = DatasetCataract1k([input_path], [label_path], img_transform=img_transform)
                        dataloader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)
                        skip_features = False

                    for i, (X, y) in enumerate(dataloader):
                        if len(y.shape) > 2:
                            y = y.squeeze()

                        validate_labels(y, n_step_classes, os.path.basename(input_path))

                        y_step = y[:, 0].long().to(device)
                        y_rsd = (y[:, 1]/60.0/25.0).float().to(device)
                        X = X.float().to(device)

                        with torch.set_grad_enabled(phase == 'train'):
                            stateful = (i > 0)
                            step_prediction, rsd_prediction = model.forwardRNN(X, stateful=stateful,
                                                                            skip_features=skip_features)
                            
                            validate_batch_dimensions(
                                step_prediction,
                                y_step,
                                f"{os.path.basename(input_path)} - batch {i}"
                            )
                            
                            loss_step = step_criterion(step_prediction, y_step)
                            rsd_prediction = rsd_prediction.squeeze(1)
                            loss_rsd = rsd_criterion(rsd_prediction, y_rsd)
                            loss = loss_step + loss_rsd

                            if phase == 'train':
                                optim.zero_grad()
                                loss.backward()
                                optim.step()

                            all_loss[phase] = torch.cat((all_loss[phase], loss.detach().view(1, -1)))
                            all_loss_step[phase] = torch.cat((all_loss_step[phase], loss_step.detach().view(1, -1)))
                            all_loss_rsd[phase] = torch.cat((all_loss_rsd[phase], loss_rsd.detach().view(1, -1)))

                        if phase in validation_phases:
                            hard_prediction = torch.argmax(step_prediction.detach(), dim=1).cpu().numpy()
                            conf_mat[phase] += confusion_matrix(y_step.cpu().numpy(), hard_prediction, 
                                                             labels=np.arange(n_step_classes))

                all_loss[phase] = all_loss[phase].cpu().numpy().mean()
                all_loss_step[phase] = all_loss_step[phase].cpu().numpy().mean()
                all_loss_rsd[phase] = all_loss_rsd[phase].cpu().numpy().mean()

                if phase in validation_phases:
                    precision, recall, f1, accuracy = conf2metrics(conf_mat[phase])
                    all_precision[phase] = precision
                    all_recall[phase] = recall
                    average_precision[phase] = np.mean(all_precision[phase])
                    average_recall[phase] = np.mean(all_recall[phase])
                    all_f1[phase] = f1
                    average_f1[phase] = np.mean(all_f1[phase])

                if log:
                    log_epoch = step_count*epoch+epoch
                    wandb.log({'epoch': log_epoch, 
                             f'{phase}/loss': all_loss[phase],
                             f'{phase}/loss_rsd': all_loss_rsd[phase],
                             f'{phase}/loss_step': all_loss_step[phase]})
                    if ((epoch % config['val']['test_every']) == 0) & (phase in validation_phases):
                        wandb.log({'epoch': log_epoch, 
                                 f'{phase}/precision': average_precision[phase],
                                 f'{phase}/recall': average_recall[phase], 
                                 f'{phase}/f1': average_f1[phase]})

            log_text = '%s ([%d/%d] %d%%), train loss: %.4f val loss: %.4f lp: %.4f' % \
                      (timeSince(start_time, (epoch + 1) / stop_epoch),
                       epoch + 1, stop_epoch, (epoch + 1) / stop_epoch * 100,
                       all_loss['train'], all_loss['val'], all_loss_step['val'])
            log_text += ' val precision: {0:.4f}, recall: {1:.4f}, f1: {2:.4f}'.format(
                average_precision['val'],
                average_recall['val'],
                average_f1['val'])
            print(log_text, end='')

            # if current loss is the best we've seen, save model state
            if num_devices > 1:
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

            if all_loss["val"] < best_loss_on_val:
                # if current loss is the best we've seen, save model state
                non_improving_val_counter = 0
                best_loss_on_val = all_loss["val"]
                print('  **')
                state = {'epoch': epoch + 1,
                         'model_dict': state_dict,
                         'remaining_steps': remaining_steps}

                torch.save(state, output_model_name)
                if log:
                    wandb.summary['best_epoch'] = epoch + 1
                    wandb.summary['best_loss_on_val'] = best_loss_on_val
                    wandb.summary['f1'] = average_f1['val']

            else:
                print('')
                non_improving_val_counter += 1
        remaining_steps.pop(0)
    print('...finished training ')

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

def validate_labels(y, n_step_classes, label_path):
    """
    Validate the labels and print detailed information about any issues found.
    
    Args:
        y: Input labels tensor
        n_step_classes: Number of expected classes
        label_path: Full path to the CSV file being processed
    """
    # Get step labels (first column)
    step_labels = y[:, 0]
    
    # Get unique labels and their counts
    unique_labels, counts = torch.unique(step_labels, return_counts=True)
    
    # Check for invalid labels
    invalid_mask = (step_labels >= n_step_classes) | (step_labels < 0)
    if invalid_mask.any():
        print("\n" + "="*50)
        print(f"INVALID LABELS FOUND IN: {label_path}")
        print(f"Total samples in file: {len(step_labels)}")
        print("\nLabel distribution:")
        for label, count in zip(unique_labels, counts):
            print(f"Label {label:.0f}: {count} samples")
        print("\nFirst few invalid labels appear at indices:", 
              torch.where(invalid_mask)[0][:5].tolist())
        print("="*50 + "\n")


def conf2metrics(conf_mat):
    # Add small epsilon to avoid division by zero
    eps = 1e-7
    
    # Calculate metrics with zero handling
    recall = np.diag(conf_mat) / (np.sum(conf_mat, axis=1) + eps)
    precision = np.diag(conf_mat) / (np.sum(conf_mat, axis=0) + eps)
    
    # Optionally mask out classes with no examples
    recall = np.where(np.sum(conf_mat, axis=1) > 0, recall, 0)
    precision = np.where(np.sum(conf_mat, axis=0) > 0, precision, 0)
    
    return precision, recall


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
    args = parser.parse_args()

    main(output_folder=args.out, log=args.log, pretrained_model=args.pretrained)

