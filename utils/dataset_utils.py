import torch
import os
import glob
import numpy as np
import cv2


class DatasetNoLabel(torch.utils.data.Dataset):
    """
    Dataset for folders with sampled png images from videos
    """
    def __init__(self, datafolders, img_transform=None, max_len=20, fps=2.5):
        super(DatasetNoLabel, self).__init__()
        
        # Input validation
        if not isinstance(datafolders, (list, tuple)):
            datafolders = [datafolders]
        
        self.datafolders = datafolders
        self.img_transform = img_transform
        self.max_len = max_len * fps * 60.0
        self.frame2min = 1/(fps * 60.0)
        
        # Initialize storage
        self.surgery_length = {}
        img_files = []
        
        # Collect files from all folders
        for d in datafolders:
            # Verify folder exists
            if not os.path.exists(d):
                raise ValueError(f"Data folder does not exist: {d}")
                
            # Try different case patterns for PNG files
            files = []
            for pattern in ['*.png', '*.PNG']:
                files.extend(glob.glob(os.path.join(d, pattern)))
            
            if not files:
                raise ValueError(f"No PNG images found in folder: {d}")
                
            files = sorted(files)
            img_files.extend(files)
            
            # Calculate surgery length for this patient
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
        # Validate index
        if not 0 <= index < self.nitems:
            raise IndexError(f"Index {index} out of range [0, {self.nitems})")
        
        # Load image with error handling
        img_path = self.img_files[index]
        img = cv2.imread(img_path)
        if img is None:
            raise RuntimeError(f"Failed to load image: {img_path}")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        patientID, frame_number = self._name2id(img_path)
        elapsed_time = frame_number * self.frame2min
        rsd = self.surgery_length[patientID] - elapsed_time
        
        if self.img_transform is not None:
            img = self.img_transform(img)
            
        # Create timestamp channel
        time_stamp = torch.ones((1, img.shape[1], img.shape[2])) * frame_number / self.max_len
        img = torch.cat((img, time_stamp.float()), dim=0)
        
        return img, elapsed_time, frame_number, rsd

    def __len__(self):
        return self.nitems

    def _name2id(self, filename):
        """Extract patient ID and frame number from filename."""
        try:
            basename = os.path.splitext(os.path.basename(filename))[0]
            parts = basename.split('_')
            if len(parts) < 2:
                raise ValueError(f"Filename does not match expected format: {filename}")
            patientID = parts[-2]
            frame = int(parts[-1])
            return patientID, frame - 1
        except (IndexError, ValueError) as e:
            raise ValueError(f"Error parsing filename {filename}: {str(e)}")


class DatasetCataract101(DatasetNoLabel):
    """Dataset for Cataract-101 with labels."""
    
    def __init__(self, datafolders, label_files, img_transform=None, max_len=20, fps=2.5):
        # Validate inputs
        if not isinstance(label_files, (list, tuple)):
            label_files = [label_files]
            
        if len(label_files) != len(datafolders):
            raise ValueError(f"Number of label files ({len(label_files)}) must match number of data folders ({len(datafolders)})")
            
        # Initialize parent class
        super().__init__(datafolders, img_transform, max_len, fps)
        
        # Process label files
        self.label_files = {}
        self.label_shapes = {}
        
        for f in label_files:
            if not os.path.exists(f):
                raise ValueError(f"Label file does not exist: {f}")
                
            try:
                patientID = os.path.splitext(os.path.basename(f))[0]
                labels = np.genfromtxt(f, delimiter=',', skip_header=1)
                
                if labels.size == 0:
                    raise ValueError(f"Empty label file: {f}")
                    
                self.label_files[patientID] = labels[:, 1:]  # Skip first column
                self.label_shapes[patientID] = labels.shape[0]
            except Exception as e:
                raise ValueError(f"Error loading label file {f}: {str(e)}")

    def __getitem__(self, index):
        img, elapsed_time, frame_number, rsd = super().__getitem__(index)
        
        # Extract patient ID and frame
        img_path = self.img_files[index]
        patientID, frame = self._name2id(img_path)
        
        # Validate patient ID
        if patientID not in self.label_files:
            raise KeyError(f"PatientID {patientID} not found in label files")
        
        # Handle frame number boundary
        max_frame = self.label_shapes[patientID]
        if frame >= max_frame:
            print(f"Warning: Frame {frame} exceeds label data length {max_frame} for {img_path}")
            frame = max_frame - 1
        
        # Return image and corresponding label
        label = self.label_files[patientID][frame]
        return img, label
