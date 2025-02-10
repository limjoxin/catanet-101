import logging
import torch
import os
import glob
import numpy as np
import cv2


logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True
)
logger = logging.getLogger(__name__)

class DatasetNoLabel(torch.utils.data.Dataset):
    """
    Dataset for folders with sampled PNG images from videos.
    This base class handles loading image data without labels.
    """
    def __init__(self, datafolders, img_transform=None, max_len=20, fps=2.5):

        super(DatasetNoLabel, self).__init__()
        
        # Convert single folder to list for consistent handling
        if not isinstance(datafolders, (list, tuple)):
            datafolders = [datafolders]
        
        self.datafolders = datafolders
        self.img_transform = img_transform
        self.max_len = max_len * fps * 60.0
        self.frame2min = 1/(fps * 60.0)
        
        # Initialize storage for surgery durations and image files
        self.surgery_length = {}
        img_files = []
        
        # Process each data folder
        for d in datafolders:
            if not os.path.exists(d):
                raise ValueError(f"Data folder does not exist: {d}")
                
            # Find all PNG files
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
        """
        if not 0 <= index < self.nitems:
            raise IndexError(f"Index {index} out of range [0, {self.nitems})")
        
        img_path = self.img_files[index]
        img = cv2.imread(img_path)
        if img is None:
            raise RuntimeError(f"Failed to load image: {img_path}")
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Apply any image transformations
        if self.img_transform is not None:
            img = self.img_transform(img)
        else:
            img = torch.from_numpy(img.transpose((2, 0, 1))).float() / 255.0

        # Extract temporal information
        patientID, frame_number = self._name2id(img_path)
        elapsed_time = frame_number * self.frame2min
        rsd = self.surgery_length[patientID] - elapsed_time
        
        # Add temporal information as an additional channel
        time_stamp = torch.ones((1, img.shape[1], img.shape[2])) * frame_number / self.max_len
        img = torch.cat((img, time_stamp.float()), dim=0)
        
        return img, elapsed_time, frame_number, rsd

    def __len__(self):
        return self.nitems

    def _name2id(self, filename):
        try:
            basename = os.path.splitext(os.path.basename(filename))[0]
            parts = basename.split('_')
            if len(parts) < 3:
                raise ValueError(f"Filename does not match expected format 'case_patientID_frameNumber': {filename}")

            patientID = f"{parts[0]}_{parts[1]}"
            frame = int(parts[-1])
            return patientID, frame - 1

        except (IndexError, ValueError) as e:
            raise ValueError(f"Error parsing filename {filename}: {str(e)}")


class DatasetCataract1k(DatasetNoLabel):
    """
    Dataset for Cataract-1K with labels.
    This class extends DatasetNoLabel to include label handling and frame validation.
    """
    def __init__(self, datafolders, label_files, img_transform=None, max_len=20, fps=2.5):
        # Validate input format consistency
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
        
        # Process each folder and its corresponding label file
        for folder, label_file in zip(datafolders, label_files):
            try:              
                # Verify and load label file
                if not os.path.exists(label_file):
                    logger.warning(f"Skipping missing label file: {label_file}")
                    continue
                    
                patientID = os.path.splitext(os.path.basename(label_file))[0]
                labels = np.genfromtxt(label_file, delimiter=',', skip_header=1)
                
                if labels.size == 0:
                    logger.warning(f"Skipping empty label file: {label_file}")
                    continue
                
                # Store both the shape and the actual labels
                self.label_shapes[patientID] = labels.shape[0]
                self.label_files[patientID] = labels
                
                # Find and validate image files
                all_files = []
                for pattern in ['*.png', '*.PNG']:
                    all_files.extend(glob.glob(os.path.join(folder, pattern)))
                
                # Filter files based on valid frame numbers
                valid_files = []
                for img_path in sorted(all_files):
                    try:
                        curr_patientID, frame = self._name2id(img_path)
                        if frame < labels.shape[0]:
                            valid_files.append(img_path)
                    except ValueError:
                        continue
                folder_to_valid_files[folder] = valid_files
                
            except Exception as e:
                logger.error(f"Error processing {label_file}: {str(e)}")
                continue
        
        # Create filtered list of folders with valid files
        valid_datafolders = [
            folder for folder in datafolders 
            if folder in folder_to_valid_files and folder_to_valid_files[folder]
        ]
        
        if not valid_datafolders:
            raise ValueError("No valid folders with both PNG files and label files found")
        
        # Initialize parent class with validated folders
        super().__init__(valid_datafolders, img_transform, max_len, fps)
        
        # Override image files list with filtered version
        self.img_files = []
        for folder in valid_datafolders:
            self.img_files.extend(folder_to_valid_files[folder])
        self.img_files = sorted(self.img_files)
        self.nitems = len(self.img_files)

    def __getitem__(self, index):
      """
      Retrieves an item by index.
      Returns the image and its corresponding label as PyTorch tensors.
      """
      img, _, _, _ = super().__getitem__(index)
      img_path = self.img_files[index]
      patientID, frame = self._name2id(img_path)
      
      # Get corresponding label and convert it to a tensor
      label = self.label_files[patientID][frame]
      label_tensor = torch.from_numpy(label).float()
      
      return img, label_tensor