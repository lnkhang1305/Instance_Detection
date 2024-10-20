from pathlib import Path
import torch.utils
from torchvision import transforms 
from typing import Optional, List, Tuple, Any, Dict
import torch
import json
from PIL import Image
import os
import logging

class ImageProcessor:
    """Handles image preprocessing and augmentation"""
    @staticmethod
    def preprocess_image(image: Image.Image, target_size: Optional[int] = None) -> Image.Image:
        """Preprocess image with resizing and normalization"""
        # print(f"[DEBUG] Original image size: {image.size}")
        if target_size:
            w, h = image.size
            if min(w, h) > target_size:
                # print(f"[DEBUG] Resizing image to thumbnail with target size {target_size}")
                image.thumbnail((target_size, target_size), Image.LANCZOS)
            else:
                # print(f"[DEBUG] Resizing image to be divisible by 14 (original size: {w}x{h})")
                new_w = ((w + 13) // 14) * 14
                new_h = ((h + 13) // 14) * 14
                image = image.resize((new_w, new_h), Image.LANCZOS)
                print(f"[DEBUG] New image size: {image.size}")
        return image

class ObjectDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir: str, transform: Optional[transforms.Compose],target_size: Optional[Tuple[int, int]] = None):
        self.data_dir = Path(data_dir)
        self.dataset_type = 'Object'
        self.transform = transform
        self.target_size = target_size
        self.image_info_cfg = self._load_configuration()

    def _load_configuration(self):
        """Create configuration for object dataset"""
        image_config = []
        global_index = 0
        print(f"[DEBUG] Loading configuration for object dataset from {self.data_dir}")  
        for source_dir in sorted(self.data_dir.glob('*')):
            object_name = source_dir.stem
            # print(f"[DEBUG] Processing object: {object_name}")  
            image_paths_list = sorted(source_dir.glob('images/*.*'))
            mask_paths_list = sorted(source_dir.glob('masks/*.*'))

            for img_path, mask_path in zip(image_paths_list, mask_paths_list):
                cfg = {
                    'id': global_index,
                    'object_name': object_name,
                    'data_dir': self.data_dir,
                    'image_path': img_path,
                    'mask_path': mask_path,
                    'dataset_type': 'Object'
                }
                image_config.append(cfg)
                global_index += 1
                # print(f"[DEBUG] Added configuration for image: {img_path}, mask: {mask_path}")
        print(f"[DEBUG] Total configurations loaded: {len(image_config)}")
        return image_config

    def __len__(self) -> int:
        return len(self.image_info_cfg)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        cfg = self.image_info_cfg[idx]
        
        try:
            image = Image.open(cfg['image_path']).convert('RGB')
            # print(f"[DEBUG] Opened image: {cfg['image_path']}")
            
            if self.target_size:
                image = ImageProcessor.preprocess_image(image, self.target_size[0])

            if self.transform:
                image = self.transform(image)

            return_cfg = {
                'id': int(cfg['id']),
                'object_name': str(cfg['object_name']),
                'data_dir': str(cfg['data_dir']),
                'image_path': str(cfg['image_path']),
                'mask_path': str(cfg['mask_path']),
                'dataset_type': str(cfg['dataset_type'])
            }

            return image, return_cfg
            
        except Exception as e:
            print(f"[ERROR] Failed to load image at index {idx}: {str(e)}")
            raise

def is_intable(s):
    try:
        int(s)  
        return True
    except ValueError:
        return False
class SceneDataset(torch.utils.data.Dataset):

    def __init__(self, data_dir: str, cfg_file_path: str, transform: Optional[transforms.Compose] = None) -> None:
        self.data_dir = data_dir
        self.dataset_type = 'Scene'
        self.transform = transform
        print(f"[DEBUG] Loading configuration from file: {cfg_file_path}")
        with open(cfg_file_path, 'r') as f:
            dict_tmp = json.load(f)
            self.cfg_info = dict_tmp['images']
        
        self.image_info = self._load_image_info()

    def _load_image_info(self) -> Dict[str, Any]:
        list_image_info = []
        print("[DEBUG] Loading image info for SceneDataset")
        self.cfg_info = sorted(self.cfg_info, key=lambda x: int(x['id']))

        for cfg in self.cfg_info:
           
            filename = cfg['file_name']
            id_ = cfg['id']
            mode, type_, img_name, extension = filename.split('.')
            image_name = img_name + '.' + extension
            
            if not is_intable(type_.split('_')[-1]):
                if ((type_ == "leisure_zone" or type_=='meeting_room') and mode == 'easy') or ((type_ == "office" or type_=='pantry_room') and mode == 'hard'):
                    type_ = type_ + '_001'
                else:
                    type_ = type_ + '_002'

            image_dir = os.path.join(self.data_dir, mode, type_, image_name)

            new_cfg = {
                'dataset_type': 'Scene',
                'data_dir': self.data_dir,
                'id': id_,
                'mode': mode,
                'type': type_,
                'image_path': image_dir
            }
            
            list_image_info.append(new_cfg)
        print(f"[DEBUG] Total images loaded: {len(list_image_info)}")
        return list_image_info

    def __len__(self) -> int:
        return len(self.image_info)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, Dict[str, Any]]:
        image_cfg = self.image_info[idx]
        image_path = image_cfg['image_path']
        image = Image.open(image_path).convert('RGB')

        if self.transform is not None:
            image,_ = self.transform(image, None)
        
        return image, image_cfg