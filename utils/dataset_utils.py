import logging
import torch
import os
import glob
import numpy as np
import cv2
from torchvision.transforms import Compose, RandomResizedCrop, RandomVerticalFlip, RandomHorizontalFlip, ToPILImage, \
    ToTensor, Resize, ColorJitter, RandomAffine, RandomRotation


logging.basicConfig(
    level=logging.ERROR,
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
        self.fps = fps  # Store fps as an instance variable
        
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
    Extended with support for phase-aware sampling to handle rare phases.
    """
    def __init__(self, datafolders, label_files=None, img_transform=None, max_len=20, fps=2.5,
             min_frames_per_phase=None, balanced_sampling=False, training_mode=False, 
             underrepresented_classes=None, problem_classes=None, problem_class_multiplier=5):

        self.problem_classes = problem_classes if problem_classes is not None else []
        self.problem_class_multiplier = problem_class_multiplier
        
        self.training_mode = training_mode
        self.underrepresented_classes = underrepresented_classes
        self.enhanced_transform = None
        
        if self.training_mode and self.underrepresented_classes:
            self.enhanced_transform = get_enhanced_transform()
        
        self.min_frames_per_phase = min_frames_per_phase
        self.balanced_sampling = balanced_sampling
        
        self.label_files = {}
        self.label_shapes = {}
        self.patient_id_map = {}
        
        super().__init__(datafolders, img_transform, max_len, fps)
        
        self.original_nitems = self.nitems

        if not isinstance(label_files, (list, tuple)):
            label_files = [label_files]
            
        if not isinstance(datafolders, (list, tuple)):
            datafolders = [datafolders]
            
        if len(label_files) != len(datafolders):
            raise ValueError(f"Number of label files ({len(label_files)}) must match number of data folders ({len(datafolders)})")
        
        self._load_labels(datafolders, label_files)
        
        if self.balanced_sampling and self.min_frames_per_phase is not None:
            self.frame_to_phase = {}
            self.phases_to_frames = {}
            # Run analysis
            print(f"Analyzing phase distribution for balanced sampling...")
            self._analyze_phase_distribution()
            self.selected_indices = self._create_balanced_sample()
            self.nitems = len(self.selected_indices)
            
        # Add aggressive oversampling for problem classes
        if self.training_mode and self.problem_classes:
            self._oversample_problem_classes()
            
    def _oversample_problem_classes(self):
        """
        Aggressively oversample frames from problem classes by adding 
        additional indices to the selected indices list.
        """
        if not self.balanced_sampling:
            print("Creating frame_to_phase mapping for problem class oversampling...")
            self.frame_to_phase = {}
            self.phases_to_frames = {}
            self._analyze_phase_distribution()
            
        problem_frames = []
        for problem_class in self.problem_classes:
            if problem_class in self.phases_to_frames:
                class_frames = self.phases_to_frames[problem_class]
                problem_frames.extend(class_frames)
            else:
                print(f"Warning: No frames found for problem class {problem_class}")
        
        if not problem_frames:
            print("Warning: No frames found for any problem classes")
            return
            
        additional_frames = []
        for _ in range(self.problem_class_multiplier - 1):
            additional_frames.extend(problem_frames)
            
        # Add to dataset
        if self.balanced_sampling:
            original_count = len(self.selected_indices)
            self.selected_indices.extend(additional_frames)
            self.nitems = len(self.selected_indices)
        else:
            original_count = self.nitems
            self.balanced_sampling = True
            self.selected_indices = list(range(self.nitems))
            self.selected_indices.extend(additional_frames)
            self.nitems = len(self.selected_indices)
            
        print(f"Aggressively oversampled problem classes {self.problem_classes}. "
              f"Added {len(additional_frames)} frames. "
              f"Dataset size: {original_count} -> {self.nitems}")


    def _load_labels(self, datafolders, label_files):
        """Load and process label files"""
        folder_to_valid_files = {}
        
        filtered_label_files = []
        filtered_datafolders = []
        
        patient_to_labels = {}
        for label_file in label_files:
            basename = os.path.basename(label_file)
            dirname = os.path.dirname(label_file)
            patient_id = os.path.basename(dirname)
            
            if patient_id not in patient_to_labels:
                patient_to_labels[patient_id] = []
            patient_to_labels[patient_id].append(label_file)
        
        for patient_id, files in patient_to_labels.items():
            if len(files) > 1:
                regular_csv = [f for f in files if '_processed_phases' not in f]
                if regular_csv:
                    chosen_file = regular_csv[0]
                else:
                    chosen_file = files[0]
            else:
                chosen_file = files[0]
            
            matching_folder = None
            for folder in datafolders:
                if patient_id in folder:
                    matching_folder = folder
                    break
            
            if matching_folder:
                filtered_label_files.append(chosen_file)
                filtered_datafolders.append(matching_folder)
        
        label_files = filtered_label_files
        datafolders = filtered_datafolders
        
        for folder, label_file in zip(datafolders, label_files):
            try:
                if not os.path.exists(label_file):
                    continue
                
                folder_name = os.path.basename(folder.rstrip('/'))
                patientID = folder_name
                
                try:
                    labels = np.genfromtxt(label_file, delimiter=',', skip_header=1)
                    
                    if labels.size == 0:
                        continue
                    
                    if labels.shape[1] == 3 and '_processed_phases' in label_file:
                        pad_column = np.zeros((labels.shape[0], 1))
                        labels = np.hstack((labels, pad_column))
                        unique_labels = np.unique(labels[:, 0])
                        n_step_classes = 13
                        
                        if np.any(unique_labels >= n_step_classes):
                            labels[:, 0] = 0
                            
                    self.label_shapes[patientID] = labels.shape[0]
                    self.label_files[patientID] = labels
                    
                except Exception as e:
                    continue
                
                all_files = []
                for pattern in ['*.png', '*.PNG']:
                    all_files.extend(glob.glob(os.path.join(folder, pattern)))
                
                if not all_files:
                    continue
                
                all_files = sorted(all_files)
                
                valid_files = []
                for img_path in all_files:
                    try:
                        filename = os.path.basename(img_path)
                        parts = filename.split('_')
                        if len(parts) >= 3 and parts[0] == 'case':
                            img_patientID = f"{parts[0]}_{parts[1]}"
                            self.patient_id_map[img_patientID] = patientID
                        
                        curr_patientID, frame = self._name2id(img_path)
                        if frame < labels.shape[0]:
                            valid_files.append(img_path)
                    except ValueError:
                        continue
                
                folder_to_valid_files[folder] = valid_files
                
            except Exception as e:
                continue
                
        valid_datafolders = [
            folder for folder in datafolders 
            if folder in folder_to_valid_files and folder_to_valid_files[folder]
        ]
        
        if not valid_datafolders:
            raise ValueError("No valid folders with both PNG files and label files found")

        self.img_files = []
        for folder in valid_datafolders:
            self.img_files.extend(folder_to_valid_files[folder])
        self.img_files = sorted(self.img_files)
        self.nitems = len(self.img_files)

    def _analyze_phase_distribution(self):
        """Analyze the distribution of phases across all frames."""
        phase_counts = {}
        total_frames = 0
        invalid_frames = 0
        
        for idx in range(self.original_nitems):
            try:
                total_frames += 1
                img_path = self.img_files[idx]
                patientID, frame = self._name2id(img_path)
                
                if patientID in self.patient_id_map:
                    standardized_id = self.patient_id_map[patientID]
                else:
                    folder_name = os.path.basename(os.path.dirname(img_path))
                    if folder_name in self.label_files:
                        standardized_id = folder_name
                    else:
                        standardized_id = patientID
                
                if standardized_id not in self.label_files:
                    invalid_frames += 1
                    continue
                    
                if frame >= len(self.label_files[standardized_id]):
                    frame = len(self.label_files[standardized_id]) - 1
                
                phase = int(self.label_files[standardized_id][frame][0])
                self.frame_to_phase[idx] = phase
                
                if phase not in self.phases_to_frames:
                    self.phases_to_frames[phase] = []
                
                if idx < self.original_nitems:
                    self.phases_to_frames[phase].append(idx)
                
                if phase not in phase_counts:
                    phase_counts[phase] = 0
                phase_counts[phase] += 1
                
            except Exception as e:
                invalid_frames += 1
                logger.error(f"Error analyzing frame {idx}: {str(e)}")
        
        # Print phase distribution statistics
        print(f"\nPhase Distribution Analysis:")
        print(f"Phase counts:")
        for phase in sorted(phase_counts.keys()):
            count = phase_counts[phase]
            percentage = (count / total_frames) * 100
            print(f"  Phase {phase}: {count} frames ({percentage:.2f}%)")
            
        for phase in sorted(self.phases_to_frames.keys()):
            frames = self.phases_to_frames[phase]
            
            invalid_indices = [idx for idx in frames if idx >= self.original_nitems]
            if invalid_indices:
                self.phases_to_frames[phase] = [idx for idx in frames if idx < self.original_nitems]

    def _create_balanced_sample(self):
        """
        Create a balanced sample of frames ensuring minimum representation per phase.
        Returns list of frame indices.
        """
        original_fps = 60.0
        sample_ratio = original_fps / self.fps

        regular_indices = []
        current_index = 0
        while current_index < self.original_nitems:
            if current_index >= self.original_nitems:
                break
            regular_indices.append(current_index)
            current_index += int(sample_ratio)
            
        regular_indices = [idx for idx in regular_indices if idx < self.original_nitems]
        print(f"Created {len(regular_indices)} regular indices at FPS={self.fps}")
        
        regular_phase_counts = {}
        for idx in regular_indices:
            if idx in self.frame_to_phase:
                phase = self.frame_to_phase[idx]
                if phase not in regular_phase_counts:
                    regular_phase_counts[phase] = 0
                regular_phase_counts[phase] += 1
        
        print("\nPhase distribution in regular sampling:")
        for phase in sorted(regular_phase_counts.keys()):
            print(f"  Phase {phase}: {regular_phase_counts.get(phase, 0)} frames")
        
        additional_indices = []
        for phase, frames in self.phases_to_frames.items():
            current_count = regular_phase_counts.get(phase, 0)
            needed_count = max(0, self.min_frames_per_phase - current_count)
            
            if needed_count > 0:
                available_frames = [f for f in frames if f < self.original_nitems and f not in regular_indices]
                
                if not available_frames:
                    print(f"  No available frames for phase {phase}")
                    continue
                    
                if len(available_frames) <= needed_count:
                    additional_indices.extend(available_frames)
                else:
                    step = len(available_frames) / needed_count
                    samples = []
                    for i in range(needed_count):
                        idx = int(i * step)
                        if idx < len(available_frames):
                            samples.append(available_frames[idx])
                    additional_indices.extend(samples)

        combined_indices = sorted(list(set(regular_indices + additional_indices)))
        combined_indices = [idx for idx in combined_indices if idx < self.original_nitems]
        
        if not combined_indices:
            print("WARNING: No valid indices found! Falling back to regular sampling.")
            combined_indices = list(range(0, min(100, self.original_nitems)))
        
        return combined_indices


    def __getitem__(self, index):
        """Retrieves an item by index with improved error handling and class-specific transformations."""
        try:
            # Input validation
            if index < 0:
                raise IndexError(f"Negative index {index} is invalid")
                
            if self.balanced_sampling:
                if index >= len(self.selected_indices):
                    raise IndexError(f"Index {index} out of range for balanced dataset with {len(self.selected_indices)} indices")
                
                actual_index = self.selected_indices[index]
                if actual_index >= self.original_nitems:
                    actual_index = actual_index % self.original_nitems
                    
                if actual_index >= len(self.img_files):
                    raise IndexError(f"Mapped index {actual_index} is still out of range [0, {len(self.img_files)})")
                
                img_path = self.img_files[actual_index]
            else:
                if index >= self.nitems:
                    raise IndexError(f"Index {index} out of range [0, {self.nitems})")
                    
                img_path = self.img_files[index]
            
            img = cv2.imread(img_path)
            if img is None:
                raise RuntimeError(f"Failed to load image: {img_path}")
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            extracted_patientID, frame = self._name2id(img_path)
            patientID = None
            if extracted_patientID in self.patient_id_map:
                patientID = self.patient_id_map[extracted_patientID]
            
            if patientID is None or patientID not in self.label_files:
                folder_name = os.path.basename(os.path.dirname(img_path))
                if folder_name in self.label_files:
                    patientID = folder_name
            
            if patientID is None or patientID not in self.label_files:
                if extracted_patientID in self.label_files:
                    patientID = extracted_patientID

            if patientID is None or patientID not in self.label_files:
                if self.img_transform is not None:
                    img = self.img_transform(img)
                else:
                    img = torch.from_numpy(img.transpose((2, 0, 1))).float() / 255.0

                frame_number = 0
                time_stamp = torch.ones((1, img.shape[1], img.shape[2])) * frame_number / self.max_len
                img = torch.cat((img, time_stamp.float()), dim=0)
                return img, torch.zeros(4).float()

            if frame >= len(self.label_files[patientID]):
                frame = len(self.label_files[patientID]) - 1
            
            label = self.label_files[patientID][frame]
            class_id = int(label[0])
            
            if self.training_mode and self.enhanced_transform and self.underrepresented_classes and class_id in self.underrepresented_classes:
                img = self.enhanced_transform(img)
            else:
                if self.img_transform is not None:
                    img = self.img_transform(img)
                else:
                    img = torch.from_numpy(img.transpose((2, 0, 1))).float() / 255.0
            
            frame_number = frame
            time_stamp = torch.ones((1, img.shape[1], img.shape[2])) * frame_number / self.max_len
            img = torch.cat((img, time_stamp.float()), dim=0)
            
            label_tensor = torch.from_numpy(label).float()
            
            return img, label_tensor
            
        except Exception as e:
            logger.error(f"Error in __getitem__ for index {index}: {str(e)}")
            if self.balanced_sampling and index < len(self.selected_indices):
                actual_index = self.selected_indices[index]
                logger.error(f"  - Balanced sampling: mapped index {index} to actual_index {actual_index}")
                logger.error(f"  - Total images: {len(self.img_files)}, Total selected indices: {len(self.selected_indices)}")
            
            return torch.zeros((4, 224, 224)), torch.zeros(4).float()
            
    def __len__(self):
        """Return the number of items in the dataset."""
        return self.nitems


def get_enhanced_transform(img_size=224):
    """
    Creates an enhanced transformation pipeline for underrepresented classes
    with more aggressive data augmentation.
    """
    enhanced_transform = Compose([
        ToPILImage(),
        RandomHorizontalFlip(p=0.7),  # Higher flip probability
        RandomVerticalFlip(p=0.7),
        RandomResizedCrop(size=img_size, scale=(0.3, 1.0), ratio=(0.8, 1.2)),
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Add color variations
        RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2)),  # Add rotation and scaling
        ToTensor()
    ])
    return enhanced_transform