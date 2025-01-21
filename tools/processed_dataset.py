#!/usr/bin/env python3

import os
import zipfile
import argparse
from google.colab import drive
from tqdm import tqdm
import sys

def mount_drive():
    """Mount Google Drive and return True if successful."""
    try:
        drive.mount('/content/drive')
        return True
    except Exception as e:
        print(f"Error mounting Google Drive: {e}")
        return False

def get_case_folders(base_path):
    """Get all valid case folders from the base path."""
    try:
        case_folders = [d for d in os.listdir(base_path)
                       if os.path.isdir(os.path.join(base_path, d))
                       and d.startswith('case_')
                       and not d.startswith('_')]
        case_folders.sort()
        return case_folders
    except Exception as e:
        print(f"Error accessing base path: {e}")
        return []

def process_case(case_folder, base_path):
    """Process a single case folder and return its files and total size."""
    case_path = os.path.join(base_path, case_folder)
    case_files = []
    case_size = 0

    try:
        for root, _, files in os.walk(case_path):
            for file in files:
                if not file.startswith('.'):
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, base_path)
                    file_size = os.path.getsize(file_path)
                    case_files.append((file_path, rel_path))
                    case_size += file_size
        return case_files, case_size
    except Exception as e:
        print(f"Error processing case {case_folder}: {e}")
        return [], 0

def create_zip_chunk(chunk_files, zip_path):
    """Create a zip file from the given files."""
    try:
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for fp, rp in chunk_files:
                zipf.write(fp, rp)
        return True
    except Exception as e:
        print(f"Error creating zip file {zip_path}: {e}")
        return False

def download_structured_dataset(base_path, output_drive_folder="cataract101_downloads", chunk_size_mb=1000):
    """
    Process the dataset and save chunks to Google Drive.
    
    Args:
        base_path (str): Path to the dataset
        output_drive_folder (str): Folder name in Google Drive for saving chunks
        chunk_size_mb (int): Size of each chunk in MB
    """
    # Mount Google Drive
    if not mount_drive():
        return False

    # Create output folder
    drive_output_path = f'/content/drive/MyDrive/{output_drive_folder}'
    os.makedirs(drive_output_path, exist_ok=True)

    # Get case folders
    case_folders = get_case_folders(base_path)
    if not case_folders:
        print("No case folders found!")
        return False

    print(f"Found {len(case_folders)} case folders")

    # Initialize chunking variables
    chunk_size = chunk_size_mb * 1024 * 1024  # Convert MB to bytes
    current_chunk_files = []
    current_chunk_size = 0
    chunk_number = 1

    # Process each case
    for case_idx, case_folder in enumerate(tqdm(case_folders), 1):
        case_files, case_size = process_case(case_folder, base_path)
        
        if case_size == 0:
            continue

        # Create new chunk if current would be too large
        if current_chunk_size + case_size > chunk_size and current_chunk_files:
            zip_path = os.path.join(drive_output_path, f'cataract101_chunk_{chunk_number:03d}.zip')
            print(f"\nCreating chunk {chunk_number} ({current_chunk_size/(1024*1024):.2f} MB)...")
            
            if not create_zip_chunk(current_chunk_files, zip_path):
                return False
                
            current_chunk_files = []
            current_chunk_size = 0
            chunk_number += 1

        # Add current case to chunk
        current_chunk_files.extend(case_files)
        current_chunk_size += case_size

    # Handle remaining files
    if current_chunk_files:
        zip_path = os.path.join(drive_output_path, f'cataract101_chunk_{chunk_number:03d}.zip')
        print(f"\nCreating final chunk {chunk_number} ({current_chunk_size/(1024*1024):.2f} MB)...")
        if not create_zip_chunk(current_chunk_files, zip_path):
            return False

    print(f"\nAll chunks have been saved to Google Drive in the folder: {output_drive_folder}")
    return True

def main():
    parser = argparse.ArgumentParser(description='Process and chunk dataset for downloading')
    parser.add_argument('base_path', help='Path to the dataset')
    parser.add_argument('--output-folder', default='cataract101_downloads',
                        help='Output folder name in Google Drive')
    parser.add_argument('--chunk-size', type=int, default=1000,
                        help='Size of each chunk in MB')

    args = parser.parse_args()

    success = download_structured_dataset(
        args.base_path,
        args.output_folder,
        args.chunk_size
    )
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
