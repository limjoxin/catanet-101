import torch
import os
import glob
import numpy as np
import cv2


class DatasetNoLabel(torch.utils.data.Dataset):
    """
    Dataset for folders with sampled PNG images from videos.
    This base class handles loading image data without labels and manages temporal information.
    It calculates surgery duration and provides time-based features for each frame.
    """
    def __init__(self, datafolders, img_transform=None, max_len=20, fps=2.5):
        super(DatasetNoLabel, self).__init__()
        
        # Convert single folder to list for consistent handling
        if not isinstance(datafolders, (list, tuple)):
            datafolders = [datafolders]
        
        self.datafolders = datafolders
        self.img_transform = img_transform
        # Calculate maximum length in frames (max_len minutes * fps * 60 seconds)
        self.max_len = max_len * fps * 60.0
        # Calculate conversion factor from frames to minutes
        self.frame2min = 1/(fps * 60.0)
        
        # Initialize storage for surgery durations and image files
        self.surgery_length = {}
        img_files = []
        
        # Process each data folder
        for d in datafolders:
            if not os.path.exists(d):
                raise ValueError(f"Data folder does not exist: {d}")
                
            # Find all PNG files (case-insensitive)
            files = []
            for pattern in ['*.png', '*.PNG']:
                files.extend(glob.glob(os.path.join(d, pattern)))
            
            if not files:
                raise ValueError(f"No PNG images found in folder: {d}")
                
            files = sorted(files)
            img_files.extend(files)
            
            # Calculate surgery length using the last frame
            try:
                patientID, frame = self._name2id(files[-1])
                self.surgery_length[patientID] = float(frame) * self.frame2min
            except (IndexError, ValueError) as e:
                raise ValueError(f"Error processing files in folder {d}: {str(e)}")
        
        self.img_files = sorted(img_files)
        self.nitems = len(self.img_files)
        
        if self.nitems == 0:
            raise ValueError(f"No valid PNG images found in any of the provided folders: {datafolders}")

    def __getitem__(self, index):
        """
        Retrieves an item by index.
        Returns the image and temporal information about the surgery.
        
        Returns:
            tuple: (image tensor, elapsed time, frame number, remaining surgery duration)
        """
        if not 0 <= index < self.nitems:
            raise IndexError(f"Index {index} out of range [0, {self.nitems})")
        
        img_path = self.img_files[index]
        img = cv2.imread(img_path)
        if img is None:
            raise RuntimeError(f"Failed to load image: {img_path}")
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Extract temporal information
        patientID, frame_number = self._name2id(img_path)
        elapsed_time = frame_number * self.frame2min
        rsd = self.surgery_length[patientID] - elapsed_time
        
        # Apply any image transformations
        if self.img_transform is not None:
            img = self.img_transform(img)
            
        # Add temporal information as an additional channel
        time_stamp = torch.ones((1, img.shape[1], img.shape[2])) * frame_number / self.max_len
        img = torch.cat((img, time_stamp.float()), dim=0)
        
        return img, elapsed_time, frame_number, rsd

    def __len__(self):
        return self.nitems

    def _name2id(self, filename):
        """
        Extracts patient ID and frame number from filename.
        Expected format: case_[patientID]_[frameNumber].png
        
        Returns:
            tuple: (patient ID, zero-based frame number)
        """
        try:
            basename = os.path.splitext(os.path.basename(filename))[0]
            parts = basename.split('_')
            if len(parts) < 2:
                raise ValueError(f"Filename does not match expected format: {filename}")
            patientID = parts[-2]
            frame = int(parts[-1])
            return patientID, frame - 1  # Convert to 0-based frame numbering
        except (IndexError, ValueError) as e:
            raise ValueError(f"Error parsing filename {filename}: {str(e)}")


class DatasetCataract101(DatasetNoLabel):
    def __init__(self, datafolders, label_files, img_transform=None, max_len=20, fps=2.5):

        if not isinstance(label_files, (list, tuple)):
            label_files = [label_files]
            
        if not isinstance(datafolders, (list, tuple)):
            datafolders = [datafolders]
            
        if len(label_files) != len(datafolders):
            raise ValueError(f"Number of label files ({len(label_files)}) must match number of data folders ({len(datafolders)})")
        
        # Initialize dictionaries for label storage and valid file tracking
        self.label_files = {}
        self.label_shapes = {}
        folder_to_valid_files = {}
        
        for folder, label_file in zip(datafolders, label_files):
            try:
                if not os.path.exists(label_file):
                    print(f"Warning: Label file does not exist: {label_file}")
                    continue
                    
                patientID = os.path.splitext(os.path.basename(label_file))[0]
                labels = np.genfromtxt(label_file, delimiter=',', skip_header=1)
                
                if labels.size == 0:
                    print(f"Warning: Label file is empty: {label_file}")
                    continue
                
                self.label_files[patientID] = labels[:, -1]
                self.label_shapes[patientID] = labels.shape[0]
                
                # Find and validate image files
                all_files = []
                for pattern in ['*.png', '*.PNG']:
                    matched_files = glob.glob(os.path.join(folder, pattern))
                    all_files.extend(matched_files)
                
                # Filter files based on valid frame numbers
                valid_files = []
                for img_path in sorted(all_files):
                    try:
                        curr_patientID, frame = self._name2id(img_path)
                        if frame < labels.shape[0]:
                            valid_files.append(img_path)
                    except ValueError as e:
                        print(f"Error parsing filename {img_path}: {str(e)}")
                        continue
                
                if valid_files:
                    folder_to_valid_files[folder] = valid_files
                
            except Exception as e:
                print(f"Error processing {label_file}: {str(e)}")
                continue
        
        # Create filtered list of folders with valid files
        valid_datafolders = [
            folder for folder in datafolders 
            if folder in folder_to_valid_files and folder_to_valid_files[folder]
        ]
            
        if not valid_datafolders:
            raise ValueError("No valid folders with both PNG files and label files found")
        
        super().__init__(valid_datafolders, img_transform, max_len, fps)
        
        # Override image files list with filtered version
        self.img_files = []
        for folder in valid_datafolders:
            self.img_files.extend(folder_to_valid_files[folder])
        self.img_files = sorted(self.img_files)
        self.nitems = len(self.img_files)