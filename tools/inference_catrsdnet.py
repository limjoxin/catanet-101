import numpy as np
import os
import argparse
from torch.utils.data import DataLoader
import torch
from torchvision.transforms import Compose, ToTensor, ToPILImage, Resize
import sys
sys.path.append('../')
from models.catRSDNet import CatRSDNet
import glob
from utils.dataset_utils import DatasetNoLabel

def main(out, input, checkpoint, labels):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- load model checkpoint
    assert os.path.isfile(checkpoint), f"Checkpoint not found: {checkpoint}"
    checkpoint_data = torch.load(checkpoint, map_location='cpu')
    model = CatRSDNet().to(device)
    model.set_cnn_as_feature_extractor()
    model.load_state_dict(checkpoint_data['model_dict'])
    model_type = 'CatNet'
    print(model_type, 'loaded')
    model.eval()

    # --- list case folders
    assert os.path.isdir(input), f"Input directory not found: {input}"
    video_folders = sorted(glob.glob(os.path.join(input, '*/')))
    if not video_folders:
        video_folders = [input]

    # --- prepare output root
    os.makedirs(out, exist_ok=True)

    # --- debug configuration
    ORIGINAL_FPS = 25.0
    SAMPLE_FPS   = 2.5

    for case_folder in video_folders:
        vname = os.path.basename(os.path.dirname(case_folder))
        print(f"Starting inference for case {vname}")

        # --- load ground truth for this case
        step_map = None
        if labels:
            lbl_file = os.path.join(labels, f"{vname}.csv")
            if not os.path.isfile(lbl_file):
                matches = glob.glob(os.path.join(labels, '**', f'{vname}.csv'), recursive=True)
                lbl_file = matches[0] if matches else None
            if lbl_file and os.path.isfile(lbl_file):
                import pandas as pd
                lbl_df = pd.read_csv(lbl_file)
                lbl_df['step'] = lbl_df['step'].astype(int)
                step_map = lbl_df['step'].to_dict()
            else:
                print(f"Warning: label CSV not found for {vname}. Filling ground_truth_step with -1.")

        # --- prepare data loader
        data = DatasetNoLabel(
            datafolders=[case_folder],
            img_transform=Compose([ToPILImage(), Resize(224), ToTensor()])
        )
        dataloader = DataLoader(
            data, batch_size=200, shuffle=False,
            num_workers=1, pin_memory=True
        )

        all_outputs = []
        for ii, (X, elapsed_time, frame_no, rsd) in enumerate(dataloader):
            # sampled frame indices
            scale = ORIGINAL_FPS / SAMPLE_FPS       # = 25.0 / 2.5 = 10.0
            orig_frame_idx = (frame_no.cpu().numpy() * scale).astype(int)
            X = X.to(device)
            elapsed_time = elapsed_time.unsqueeze(0).float().to(device)
            elapsed_np = elapsed_time.view(-1).cpu().numpy()

            # --- lookup GT phase or default
            if step_map is not None:
                gt_step = np.array([step_map.get(int(fid), -1) for fid in orig_frame_idx])
            else:
                gt_step = np.full(len(orig_frame_idx), -1)

            with torch.no_grad():
                if model_type == 'CatNet':
                    step_pred, exp_pred, rsd_pred = model(X, stateful=(ii > 0))
                    step_hard = torch.argmax(step_pred, dim=-1).cpu().numpy()
                    exp_soft = exp_pred.clone().cpu().numpy()
                    exp_hard = (torch.argmax(exp_pred, dim=-1) + 1).cpu().numpy()
                else:
                    rsd_pred = model(X, stateful=(ii > 0))
                    step_hard = np.zeros(len(rsd_pred))
                    exp_hard = np.zeros(len(rsd_pred))
                    exp_soft = np.zeros((len(rsd_pred), 2))

                elapsed_np = elapsed_time.view(-1).cpu().numpy()
                rsd_np = rsd_pred.view(-1).cpu().numpy()
                progress = elapsed_np / (elapsed_np + rsd_np + 1e-5)

            batch_out = np.vstack([
                orig_frame_idx,       # 0
                elapsed_np,         # 1
                progress,           # 2
                rsd_np,             # 3
                step_hard,          # 4
                exp_hard,           # 5
                exp_soft[:, 0],     # 6
                exp_soft[:, 1],     # 7
                gt_step             # 8
            ]).T
            all_outputs.append(batch_out)

        # --- concatenate and save CSV
        out_arr = np.concatenate(all_outputs, axis=0)
        save_path = os.path.join(out, f"{vname}.csv")
        np.savetxt(
            save_path,
            out_arr,
            delimiter=',',
            header=('frame_idx,elapsed,progress,'
                    'predicted_rsd,predicted_step,'
                    'predicted_exp,predicted_assistant,'
                    'predicted_senior,'
                    'ground_truth_step'),
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
        help='directory that contains per‑video label CSVs (same layout as in training)'
    )
    args = parser.parse_args()

    main(
        out=args.out,
        input=args.input,
        checkpoint=args.checkpoint,
        labels=args.labels
    )


