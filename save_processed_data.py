import os
import numpy as np
import pandas as pd
from pathlib import Path
import cv2
import logging
from typing import Dict, Any, List, Tuple
import h5py
from tqdm import tqdm  # For progress tracking

class DatasetManager:
    """
    Manages a dataset with multiple subdirectories containing images and CSV files.
    Each subdirectory is treated as a separate category or collection of related data.
    """
    
    def __init__(self, 
                 base_dir: str = 'data/',
                 save_dir: str = 'processed_data/',
                 image_size: Tuple[int, int] = (224, 224)):  # Default size for resizing
        # Set up logging for tracking operations
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize directory paths
        self.base_dir = Path(base_dir)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Set image processing parameters
        self.image_size = image_size
        
        # Define supported file types
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        
        # Dictionary to store metadata about processed data
        self.metadata = {}

    def _process_image(self, image_path: Path) -> np.ndarray:
        """
        Process a single image: read, resize, and normalize it.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Processed image as a numpy array
        """
        try:
            # Read image using OpenCV
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Failed to read image: {image_path}")
            
            # Convert BGR to RGB (OpenCV uses BGR by default)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize image to standard size
            image = cv2.resize(image, self.image_size)
            
            # Normalize pixel values to [0, 1]
            image = image.astype(np.float32) / 255.0
            
            return image
            
        except Exception as e:
            self.logger.error(f"Error processing image {image_path}: {str(e)}")
            return None

    def _process_csv(self, csv_path: Path) -> pd.DataFrame:
        """
        Process a CSV file with error handling and basic cleaning.
        
        Args:
            csv_path: Path to the CSV file
            
        Returns:
            Processed DataFrame
        """
        try:
            # Read CSV file
            df = pd.read_csv(csv_path)
            
            # Basic cleaning steps
            df = df.dropna()  # Remove rows with missing values
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error processing CSV {csv_path}: {str(e)}")
            return None

    def process_and_save_data(self) -> None:
        """
        Process all subdirectories and save their contents in an organized structure.
        Each subdirectory is processed separately to manage memory efficiently.
        """
        self.logger.info("Starting dataset processing...")
        
        # Get all subdirectories
        subdirs = [d for d in self.base_dir.iterdir() if d.is_dir()]
        
        # Create HDF5 file for storing all data
        with h5py.File(self.save_dir / 'processed_dataset.h5', 'w') as hf:
            # Process each subdirectory
            for subdir in tqdm(subdirs, desc="Processing directories"):
                try:
                    # Create a group for this subdirectory in the HDF5 file
                    subdir_group = hf.create_group(subdir.name)
                    
                    # Process images in this subdirectory
                    image_paths = [
                        p for p in subdir.glob('*')
                        if p.suffix.lower() in self.image_extensions
                    ]
                    
                    # Create a dataset for images
                    if image_paths:
                        # Process first image to get dimensions
                        sample_image = self._process_image(image_paths[0])
                        if sample_image is not None:
                            # Create dataset with appropriate shape
                            images_dataset = subdir_group.create_dataset(
                                'images',
                                shape=(len(image_paths), *sample_image.shape),
                                dtype=np.float32,
                                compression='gzip'
                            )
                            
                            # Process and store each image
                            for idx, img_path in enumerate(image_paths):
                                processed_image = self._process_image(img_path)
                                if processed_image is not None:
                                    images_dataset[idx] = processed_image
                                    
                            # Store image filenames as attributes
                            images_dataset.attrs['filenames'] = [
                                str(p.name) for p in image_paths
                            ]
                    
                    # Process CSV file if it exists
                    csv_files = list(subdir.glob('*.csv'))
                    if csv_files:
                        csv_path = csv_files[0]  # Assume one CSV per directory
                        df = self._process_csv(csv_path)
                        if df is not None:
                            # Store DataFrame in HDF5
                            subdir_group.create_dataset(
                                'csv_data',
                                data=df.to_numpy(),
                                compression='gzip'
                            )
                            # Store column names as attributes
                            subdir_group['csv_data'].attrs['columns'] = df.columns.tolist()
                    
                    self.logger.info(f"Successfully processed directory: {subdir.name}")
                    
                except Exception as e:
                    self.logger.error(f"Error processing directory {subdir.name}: {str(e)}")

    def load_data(self, subdirs: List[str] = None) -> Dict[str, Dict[str, Any]]:
        """
        Load processed data for specified subdirectories or all if none specified.
        
        Args:
            subdirs: List of subdirectory names to load, or None for all
            
        Returns:
            Dictionary containing the loaded data organized by subdirectory
        """
        data = {}
        
        try:
            with h5py.File(self.save_dir / 'processed_dataset.h5', 'r') as hf:
                # If no specific subdirs requested, load all
                if subdirs is None:
                    subdirs = list(hf.keys())
                
                # Load requested subdirectories
                for subdir in subdirs:
                    if subdir in hf:
                        data[subdir] = {
                            'images': hf[subdir]['images'][:],
                            'image_names': list(hf[subdir]['images'].attrs['filenames'])
                        }
                        
                        # Load CSV data if it exists
                        if 'csv_data' in hf[subdir]:
                            csv_data = hf[subdir]['csv_data'][:]
                            columns = list(hf[subdir]['csv_data'].attrs['columns'])
                            data[subdir]['csv_data'] = pd.DataFrame(
                                csv_data,
                                columns=columns
                            )
                            
                        self.logger.info(f"Successfully loaded data for {subdir}")
                    else:
                        self.logger.warning(f"Subdirectory {subdir} not found in processed data")
                        
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            
        return data