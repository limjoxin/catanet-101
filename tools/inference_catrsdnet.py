import numpy as np
import os
import argparse
from torch.utils.data import DataLoader
import torch
from torchvision.transforms import Compose, ToTensor, ToPILImage, Resize, Normalize
import sys
sys.path.append('..')
from models.catRSDNet import CombinedEnhancedModel
import glob

def main(out, input_dir, checkpoint, labels):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- load model checkpoint
    assert os.path.isfile(checkpoint), f"Checkpoint not found: {checkpoint}"
    checkpoint_data = torch.load(checkpoint, map_location='cpu', weights_only=False)

    # --- initialize CombinedEnhancedModel
    model = CombinedEnhancedModel().to(device)
    model.set_cnn_as_feature_extractor()
    model.load_state_dict(checkpoint_data['model_dict'])
    print("CombinedEnhancedModel loaded")
    model.eval()

    # --- prepare input folders
    assert os.path.isdir(input_dir), 'No valid input provided; needs to be a folder'
    video_folders = sorted(glob.glob(os.path.join(input_dir, '*/')))
    if len(video_folders) == 0:
        video_folders = [input_dir]

    # --- FPS scaling
    ORIGINAL_FPS = 25.0
    SAMPLE_FPS   = 2.5
    scale = ORIGINAL_FPS / SAMPLE_FPS

    # --- process each case
    for case_folder in video_folders:
        folder_name = os.path.basename(os.path.dirname(case_folder))
        if folder_name.startswith('case_'):
            case_id = folder_name[len('case_'):]
        else:
            case_id = folder_name
        print(f"Starting inference for {folder_name}")

        # --- load ground truth mapping (optional)
        step_map = None
        if labels:
            lbl_file = os.path.join(labels, f"{case_id}.csv")
            if not os.path.isfile(lbl_file):
                matches = glob.glob(os.path.join(labels, '**', f'{case_id}.csv'), recursive=True)
                lbl_file = matches[0] if matches else None
            if lbl_file and os.path.isfile(lbl_file):
                import pandas as pd
                lbl_df = pd.read_csv(lbl_file)
                lbl_df['step'] = lbl_df['step'].astype(int)
                step_map = lbl_df['step'].to_dict()
            else:
                print(f"Warning: label CSV not found for {case_id}. Using -1 for ground_truth_step.")

        # --- prepare data loader with normalization
        from utils.dataset_utils import DatasetNoLabel
        img_transform = Compose([
            ToPILImage(),
            Resize(224),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        data = DatasetNoLabel(datafolders=[case_folder], img_transform=img_transform)
        dataloader = DataLoader(data, batch_size=200, shuffle=False,
                                num_workers=1, pin_memory=True)

        all_outputs = []
        for ii, (X, elapsed_time, frame_no, _) in enumerate(dataloader):
            X = X.to(device)
            elapsed_time = elapsed_time.unsqueeze(0).float().to(device)

            # compute original frame indices
            orig_frame_idx = (frame_no.cpu().numpy() * scale).astype(int)

            with torch.no_grad():
                step_pred, rsd_pred = model(X, elapsed_time, stateful=(ii > 0))
                step_hard = torch.argmax(step_pred, dim=-1).cpu().numpy()
                rsd_np = rsd_pred.view(-1).cpu().numpy()
                elapsed_np = elapsed_time.view(-1).cpu().numpy()
                progress = elapsed_np / (elapsed_np + rsd_np + 1e-5)

            # --- lookup GT phase or default using case_id
            if step_map is not None:
                gt_step = np.array([step_map.get(int(fid), -1) for fid in orig_frame_idx])
            else:
                gt_step = np.full(len(orig_frame_idx), -1)

            # --- collect outputs
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
        # use original folder_name for output filename
        save_path = os.path.join(out, f"{folder_name}.csv")
        np.savetxt(
            save_path,
            out_arr,
            delimiter=',',
            header='frame_idx,elapsed,progress,predicted_rsd,predicted_step,ground_truth_step',
            comments=''
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--out', type=str, default='output',
        help='root folder for case-specific outputs'
    )
    parser.add_argument(
        '--input', type=str, required=True,
        help='directory containing case folders'
    )
    parser.add_argument(
        '--checkpoint', type=str, required=True,
        help='path to model checkpoint (.pth)'
    )
    parser.add_argument(
        '--labels', type=str, default=None,
        help='directory containing per-video label CSVs'
    )
    args = parser.parse_args()

    main(
        out=args.out,
        input_dir=args.input,
        checkpoint=args.checkpoint,
        labels=args.labels
    )

