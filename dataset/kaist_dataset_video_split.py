import os
import json
import random
import csv
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision.transforms import RandomCrop

class KAIST_ThermalEventRGBDatasetVideo(Dataset):
    """
    Dataset loader for KAIST-MS dataset with synchronized Thermal, RGB, and Event Voxel Grids.
    
    Expected Folder Structure for KAIST:
        thermal_root/setXX/V000/lwir/I00000.jpg
        rgb_root/setXX/V000/visible/I00000.jpg
        voxel_root/setXX/V000/setXX_V000_00000.npy (Event Voxel Grid)
    """

    def __init__(self,
                 thermal_root,
                 rgb_root,
                 voxel_root,
                 split_json,                  
                 mode='train',        # 'train' | 'val' | 'test'
                 target_size=(256,256),
                 skip_empty_voxel=True,
                 use_filter_csv=False,
                 filter_csv_path=None,
                 seed=42):
        super().__init__()
        assert mode in ["train", "val", "test"]
        self.thermal_root = Path(thermal_root)
        self.rgb_root     = Path(rgb_root)
        self.voxel_root   = Path(voxel_root)
        self.mode         = mode
        self.target_size  = target_size
        self.skip_empty_voxel = skip_empty_voxel
        self.use_filter_csv   = use_filter_csv
        self.seed = seed
        
        # KAIST Day sets (Sets 01, 02, 06, 07, 08 are captured during the day)
        self.day_sets = {"set01", "set02", "set06", "set07", "set08"}

        self.filtered_triplets_set = set()
        if self.use_filter_csv and filter_csv_path:
            self.filtered_triplets_set = self._load_filtered_triplets(filter_csv_path)

        # Load video split from JSON
        with open(split_json, "r") as f:
            split = json.load(f)
        
        # self.videos is a list of tuples: (set_name, video_name)
        self.videos = [tuple(x) for x in split[mode]]

        self.triplets = self._gather_triplets_from_selected_videos()
        print(f"[{self.mode.upper()}] Using {len(self.triplets)} triplets (from {len(self.videos)} videos)")

    def _load_filtered_triplets(self, csv_path):
        """Loads a list of specific files to ignore from a CSV."""
        filtered = set()
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            next(reader, None) # Skip header
            for row in reader:
                filtered.add(row[0])
        return filtered

    def _gather_triplets_from_selected_videos(self):
        """Scans directories to create valid (Thermal, Event, RGB) triplets."""
        triplets = []
        for set_name, video_name in self.videos:
            # Filter only Day sets
            if set_name not in self.day_sets: 
                continue
            
            video_rgb_path = self.rgb_root / set_name / video_name / "visible"
            video_th_path  = self.thermal_root / set_name / video_name / "lwir"
            video_voxel_path = self.voxel_root / set_name / video_name
            
            if not video_rgb_path.is_dir():
                continue
            
            # Align files based on RGB filenames
            rgb_files = sorted([f for f in os.listdir(video_rgb_path) if f.endswith(".jpg")])
            
            for rgb_file in rgb_files:
                # Extract index (e.g. I01234.jpg -> 01234)
                idx = rgb_file.replace("I", "").replace(".jpg", "")
                
                # Construct corresponding filenames
                thermal_file = f"I{idx}.jpg"
                voxel_file   = f"{set_name}_{video_name}_{idx}.npy"
                
                thermal_path = video_th_path / thermal_file
                rgb_path     = video_rgb_path / rgb_file
                voxel_path   = video_voxel_path / voxel_file
                
                # Check existence
                if not (thermal_path.exists() and voxel_path.exists()):
                    continue
                
                # Option: Skip if voxel grid is completely empty (no events)
                if self.skip_empty_voxel:
                    v = np.load(str(voxel_path))
                    if np.all(v == 0): 
                        continue
                
                # Option: Filter bad samples via CSV
                if self.use_filter_csv and str(voxel_path) in self.filtered_triplets_set:
                    continue
                
                triplets.append((str(thermal_path), str(voxel_path), str(rgb_path)))
        
        return triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        thermal_path, voxel_path, rgb_path = self.triplets[idx]
        
        # Load and Resize Images
        thermal = Image.open(thermal_path).convert("L").resize(self.target_size, Image.BILINEAR)
        rgb     = Image.open(rgb_path).convert("RGB").resize(self.target_size, Image.BILINEAR)

        # Load Voxel Grid [T, H, W]
        voxel = np.load(voxel_path)  
        
        # Resize Voxel Grid if necessary
        # Note: We resize each temporal bin individually as an image
        if voxel.shape[1:] != self.target_size:
            voxel = np.stack([
                np.array(Image.fromarray(voxel[i]).resize(self.target_size, Image.BILINEAR))
                for i in range(voxel.shape[0])
            ])
            
        voxel_tensor = torch.from_numpy(voxel).float()       # [T,H,W]
        thermal_tensor = TF.to_tensor(thermal)  # [1,H,W] in [0,1]
        rgb_tensor     = TF.to_tensor(rgb)      # [3,H,W] in [0,1]

        # --- DATA AUGMENTATION (Optional / Currently Disabled) ---
        # if self.mode == "train": 
        #     if random.random() < 0.5:
        #         thermal_tensor = TF.hflip(thermal_tensor)
        #         rgb_tensor     = TF.hflip(rgb_tensor)
        #         voxel_tensor   = TF.hflip(voxel_tensor)

        #     angle = random.uniform(-10, 10)
        #     thermal_tensor = TF.rotate(thermal_tensor, angle, interpolation=TF.InterpolationMode.BILINEAR)
        #     rgb_tensor     = TF.rotate(rgb_tensor, angle, interpolation=TF.InterpolationMode.BILINEAR)
        #     voxel_tensor   = TF.rotate(voxel_tensor, angle, interpolation=TF.InterpolationMode.BILINEAR)

        #     crop_size = (192, 192)
        #     i, j, h, w = RandomCrop.get_params(thermal_tensor, output_size=crop_size)
        #     thermal_tensor = TF.crop(thermal_tensor, i, j, h, w)
        #     rgb_tensor     = TF.crop(rgb_tensor, i, j, h, w)
        #     voxel_tensor   = TF.crop(voxel_tensor, i, j, h, w)

        #     thermal_tensor = TF.resize(thermal_tensor, self.target_size, interpolation=TF.InterpolationMode.BILINEAR)
        #     rgb_tensor     = TF.resize(rgb_tensor, self.target_size, interpolation=TF.InterpolationMode.BILINEAR)
        #     voxel_tensor   = TF.resize(voxel_tensor, self.target_size, interpolation=TF.InterpolationMode.BILINEAR)

        return {
            "thermal": thermal_tensor,     # [1,H,W]
            "events":  voxel_tensor,       # [T,H,W]
            "rgb_gt":  rgb_tensor,         # [3,H,W]
            "filename": os.path.basename(voxel_path),
            "video_id": "/".join(Path(rgb_path).parts[-5:-3])  # extract set/video
        }