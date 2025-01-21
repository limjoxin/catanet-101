
import os
import zipfile
from google.colab import drive
from tqdm import tqdm

def download_structured_dataset(base_path, output_drive_folder="cataract101_downloads"):
    """
    Processes the dataset and saves chunks to Google Drive for reliable downloading.

    Parameters:
        base_path: Path to the dataset
        output_drive_folder: Folder name in Google Drive where chunks will be saved
    """
    # Mount Google Drive
    drive.mount('/content/drive')

    # Create output folder in Google Drive
    drive_output_path = f'/content/drive/MyDrive/{output_drive_folder}'
    os.makedirs(drive_output_path, exist_ok=True)

    # Get all case folders
    case_folders = [d for d in os.listdir(base_path)
                   if os.path.isdir(os.path.join(base_path, d))
                   and d.startswith('case_')
                   and not d.startswith('_')]
    case_folders.sort()

    print(f"Found {len(case_folders)} case folders")

    # Process cases in smaller chunks (about 1GB each)
    chunk_size = 1000 * 1024 * 1024  # 1GB in bytes
    current_chunk_files = []
    current_chunk_size = 0
    chunk_number = 1

    for case_idx, case_folder in enumerate(case_folders, 1):
        case_path = os.path.join(base_path, case_folder)
        case_files = []
        case_size = 0

        # Collect files for this case
        for root, _, files in os.walk(case_path):
            for file in files:
                if not file.startswith('.'):
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, base_path)
                    file_size = os.path.getsize(file_path)
                    case_files.append((file_path, rel_path))
                    case_size += file_size

        print(f"Processing {case_folder}: {len(case_files)} files, {case_size/(1024*1024):.2f} MB")

        # If chunk would be too large, save current chunk
        if current_chunk_size + case_size > chunk_size and current_chunk_files:
            zip_path = os.path.join(drive_output_path, f'cataract101_chunk_{chunk_number:03d}.zip')
            print(f"\nCreating chunk {chunk_number} ({current_chunk_size/(1024*1024):.2f} MB)...")

            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for fp, rp in current_chunk_files:
                    zipf.write(fp, rp)

            print(f"Saved chunk {chunk_number} to Google Drive")
            current_chunk_files = []
            current_chunk_size = 0
            chunk_number += 1

        # Add current case to chunk
        current_chunk_files.extend(case_files)
        current_chunk_size += case_size
        print(f"Progress: {case_idx}/{len(case_folders)} cases processed")

    # Handle remaining files
    if current_chunk_files:
        zip_path = os.path.join(drive_output_path, f'cataract101_chunk_{chunk_number:03d}.zip')
        print(f"\nCreating final chunk {chunk_number} ({current_chunk_size/(1024*1024):.2f} MB)...")

        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for fp, rp in current_chunk_files:
                zipf.write(fp, rp)

        print(f"Saved final chunk to Google Drive")

    print("\nAll chunks have been saved to Google Drive in the folder:", output_drive_folder)
