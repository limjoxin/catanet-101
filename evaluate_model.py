import torch
from torch.utils.data import DataLoader
import glob
import os
from models.catRSDNet import CatRSDNet
from utils.dataset_utils import DatasetCataract101
from torchvision.transforms import Compose, ToPILImage, ToTensor, Resize
from sklearn.metrics import confusion_matrix
import numpy as np

def evaluate_model(model_path, data_path):
    """
    Evaluates a saved CatRSDNet model on validation data and prints performance metrics.
    
    Args:
        model_path: Path to the .pth model file
        data_path: Base path to the cataract101 dataset
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load the model
    model = CatRSDNet()
    print("Model structure:")
    print(model)

    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_dict'])
    model = model.to(device)
    model.eval()
    
    # Print training info
    print(f"\nModel was trained for {checkpoint['epoch']} epochs")
    
    # Setup validation data
    img_transform = Compose([
        ToPILImage(),
        Resize((224, 224)),
        ToTensor()
    ])
    
    # Get validation data paths
    val_video_paths = sorted(glob.glob(os.path.join(data_path, 'val', '*/')))
    val_label_paths = sorted(glob.glob(os.path.join(data_path, 'val', '**', '*.csv')))
    
    # Initialize metrics
    all_phase_preds = []
    all_phase_labels = []
    all_exp_preds = []
    all_exp_labels = []
    all_rsd_preds = []
    all_rsd_labels = []
    
    print("\nEvaluating model on validation data...")
    
    # Evaluate each validation sequence
    for video_path, label_path in zip(val_video_paths, val_label_paths):
        dataset = DatasetCataract101([video_path], [label_path], img_transform=img_transform)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        # Process each batch
        model.hidden = None  # Reset LSTM state
        for images, labels in dataloader:
            images = images.to(device)
            with torch.no_grad():
                phase_pred, exp_pred, rsd_pred = model.forwardRNN(images)
                
            # Convert predictions to class indices
            phase_pred = torch.argmax(phase_pred, dim=1).cpu().numpy()
            exp_pred = torch.argmax(exp_pred, dim=1).cpu().numpy()
            rsd_pred = rsd_pred.squeeze().cpu().numpy()
            
            # Store predictions and labels
            all_phase_preds.extend(phase_pred)
            all_phase_labels.extend(labels[:, 0].numpy())
            all_exp_preds.extend(exp_pred)
            all_exp_labels.extend(labels[:, 2].numpy() - 1)  # Adjust for 0-based indexing
            all_rsd_preds.extend(rsd_pred)
            all_rsd_labels.extend((labels[:, 5]/60.0/25.0).numpy())
    
    # Calculate metrics
    phase_conf_mat = confusion_matrix(all_phase_labels, all_phase_preds, labels=range(11))
    exp_conf_mat = confusion_matrix(all_exp_labels, all_exp_preds, labels=range(2))
    
    # Calculate precision, recall, F1 for phases
    phase_precision = np.diag(phase_conf_mat) / np.sum(phase_conf_mat, axis=0)
    phase_recall = np.diag(phase_conf_mat) / np.sum(phase_conf_mat, axis=1)
    phase_f1 = 2 * (phase_precision * phase_recall) / (phase_precision + phase_recall)
    
    # Calculate experience accuracy
    exp_accuracy = np.trace(exp_conf_mat) / np.sum(exp_conf_mat)
    
    # Calculate RSD error
    rsd_mae = np.mean(np.abs(np.array(all_rsd_preds) - np.array(all_rsd_labels)))
    
    # Print results
    print("\nPerformance Metrics:")
    print(f"Phase Classification:")
    print(f"  Average Precision: {np.nanmean(phase_precision):.4f}")
    print(f"  Average Recall: {np.nanmean(phase_recall):.4f}")
    print(f"  Average F1: {np.nanmean(phase_f1):.4f}")
    print(f"\nExperience Classification:")
    print(f"  Accuracy: {exp_accuracy:.4f}")
    print(f"\nRSD Prediction:")
    print(f"  Mean Absolute Error: {rsd_mae:.4f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate saved CatRSDNet model')
    parser.add_argument('--model', type=str, required=True, help='Path to .pth model file')
    parser.add_argument('--data', type=str, required=True, help='Path to cataract101 dataset')
    
    args = parser.parse_args()
    evaluate_model(args.model, args.data)