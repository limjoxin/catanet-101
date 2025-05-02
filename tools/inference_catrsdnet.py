import os
import argparse
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, ToPILImage, Resize, Normalize
import sys
sys.path.append('..')
from models.catRSDNet import CombinedEnhancedModel
import glob

def load_label_map(labels_dir, case_id):
    """
    Load ground-truth labels from CSV for a given case.
    Expects CSV with columns: 'frame' and 'step'.
    Returns a dict mapping frame indices to step labels.
    """
    # Search for label file in directory and subdirectories
    matches = glob.glob(os.path.join(labels_dir, f"**/{case_id}.csv"), recursive=True)
    if matches:
        lbl_file = matches[0]
        print(f"Debug: Found label file {lbl_file} for case {case_id}")
        df = pd.read_csv(lbl_file)
        # Validate columns
        if not {'frame', 'step'}.issubset(df.columns):
            raise ValueError(f"Label file {lbl_file!r} must contain 'frame' and 'step' columns.")
        df['frame'] = df['frame'].astype(int)
        df['step'] = df['step'].astype(int)
        print(f"Debug: Loaded {len(df)} label entries from {lbl_file}")
        return dict(zip(df['frame'], df['step']))
    else:
        print(f"Debug: No label file found for case {case_id} in {labels_dir}")
        return {}


def main(out, input_dir, checkpoint, labels_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- load model checkpoint
    assert os.path.isfile(checkpoint), f"Checkpoint not found: {checkpoint}"
    checkpoint_data = torch.load(checkpoint, map_location='cpu', weights_only=False)

    # --- initialize model
    model = CombinedEnhancedModel().to(device)
    model.set_cnn_as_feature_extractor()
    model.load_state_dict(checkpoint_data['model_dict'])
    print("CombinedEnhancedModel loaded")
    model.eval()

    # --- prepare input folders
    assert os.path.isdir(input_dir), 'Input directory not found'
    video_folders = sorted(glob.glob(os.path.join(input_dir, '*/')))
    if not video_folders:
        video_folders = [input_dir]

    # --- FPS scaling
    ORIGINAL_FPS = 25.0
    SAMPLE_FPS   = 2.5
    scale = ORIGINAL_FPS / SAMPLE_FPS

    # --- process each case
    for case_folder in video_folders:
        folder_name = os.path.basename(os.path.dirname(case_folder))
        case_id = folder_name[len('case_'):] if folder_name.startswith('case_') else folder_name
        print(f"Starting inference for {folder_name}")

        # --- load ground truth mapping
        if labels_dir:
            step_map = load_label_map(labels_dir, case_id)
            if not step_map:
                print(f"Warning: no labels found for {case_id}, defaulting to -1")
        else:
            step_map = {}
            print("Debug: No labels_dir provided; ground-truth labels will be skipped.")

        # --- prepare data loader
        from utils.dataset_utils import DatasetNoLabel
        img_transform = Compose([
            ToPILImage(),
            Resize(224),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        data = DatasetNoLabel(datafolders=[case_folder], img_transform=img_transform)
        dataloader = DataLoader(data, batch_size=200, shuffle=False, num_workers=1, pin_memory=True)

        all_outputs = []
        for ii, (X, elapsed_time, frame_no, _) in enumerate(dataloader):
            X = X.to(device)
            elapsed_time = elapsed_time.unsqueeze(0).float().to(device)

            # compute original frame indices with rounding
            orig_frame_idx = np.round(frame_no.cpu().numpy() * scale).astype(int)

            with torch.no_grad():
                step_pred, rsd_pred = model(X, elapsed_time, stateful=(ii > 0))
                if rsd_pred.dim() == 2:
                    rsd_pred = rsd_pred[:, -1]
                step_hard = torch.argmax(step_pred, dim=-1).cpu().numpy()
                rsd_np = rsd_pred.view(-1).cpu().numpy()
                elapsed_np = elapsed_time.view(-1).cpu().numpy()
                progress = elapsed_np / (elapsed_np + rsd_np + 1e-5)

            # --- lookup GT step
            if step_map:
                gt_step = np.array([step_map.get(idx, -1) for idx in orig_frame_idx])
            else:
                gt_step = np.full(len(orig_frame_idx), -1)

            batch_out = np.vstack([
                orig_frame_idx,
                elapsed_np,
                progress,
                rsd_np,
                step_hard,
                gt_step
            ]).T
            all_outputs.append(batch_out)

        # --- concatenate and save CSV
        out_arr = np.concatenate(all_outputs, axis=0)
        os.makedirs(out, exist_ok=True)
        save_path = os.path.join(out, f"{folder_name}.csv")
        np.savetxt(
            save_path,
            out_arr,
            delimiter=',',
            header='frame_idx,elapsed,progress,predicted_rsd,predicted_step,ground_truth_step',
            comments=''
        )
        print(f"Saved results to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, default='output', help='root folder for outputs')
    parser.add_argument('--input', type=str, required=True, help='directory with case folders')
    parser.add_argument('--checkpoint', type=str, required=True, help='model checkpoint path')
    parser.add_argument('--labels', type=str, default=None, help='directory of label CSVs')
    args = parser.parse_args()

    main(
        out=args.out,
        input_dir=args.input,
        checkpoint=args.checkpoint,
        labels_dir=args.labels
    )


