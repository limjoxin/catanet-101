import os
import numpy as np
import h5py
from PIL import Image
from tqdm import tqdm

def process_and_save_dataset():
    base_dir = "cataract101"
    splits = ["train", "test", "val"]
    output_file = "processed_cataract101.h5"
    
    print(f"Looking for data in: {os.path.abspath(base_dir)}")
    
    if not os.path.exists(base_dir):
        print(f"Error: Directory {base_dir} not found!")
        print("Current directory contains:", os.listdir("."))
        return
        
    with h5py.File(output_file, 'w') as f:
        for split in splits:
            split_path = os.path.join(base_dir, split)
            print(f"\nProcessing {split} split from: {split_path}")
            
            if not os.path.exists(split_path):
                print(f"Warning: {split_path} not found, skipping...")
                continue
                
            patient_dirs = sorted([d for d in os.listdir(split_path) 
                                 if os.path.isdir(os.path.join(split_path, d))])
            print(f"Found {len(patient_dirs)} patients in {split} split")
            
            # Create groups for this split
            split_group = f.create_group(split)
            
            # Process each patient
            for patient_id in tqdm(patient_dirs, desc=f"Processing {split}"):
                patient_path = os.path.join(split_path, patient_id)
                image_files = sorted([f for f in os.listdir(patient_path) 
                                    if f.endswith('.png')])
                
                if not image_files:
                    print(f"Warning: No PNG files found for patient {patient_id}")
                    continue