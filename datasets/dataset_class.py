from pathlib import Path
import torch.utils
from torchvision import transforms 
from typing import Optional, List, Tuple, Any, Dict
import torch
import json
from PIL import Image
import os


class ImageProcessor:
    """Handles image preprocessing and augmentation"""
    @staticmethod
    def preprocess_image(image: Image.Image, target_size: Optional[int] = None) -> Image.Image:
        """Preprocess image with resizing and normalization"""
        if target_size:
            w, h = image.size
            if min(w, h) > target_size:
                image.thumbnail((target_size, target_size), Image.LANCZOS)
            else:
                new_w = ((w + 13) // 14) * 14
                new_h = ((h + 13) // 14) * 14
                image = image.resize((new_w, new_h), Image.LANCZOS)
        return image

class ObjectDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir:str, transform:Optional[transforms.Compose], target_size: Optional[Tuple[int,int]]= None):
        self.data_dir = Path(data_dir)
        self.dataset_type = 'Object'
        self.transform = transform
        self.target_size = target_size
        self.image_info_cfg = self._load_configuration
    
    def _load_configuration(self):
        """Create configuration for object dataset
        
        Include:
            - dataset_type
            - image_dir
            - mask_dir
            - obj_name
            - length
        """
        image_config = []
        global_index = 0
        for source_dir in sorted(self.data_dir.glob('*')):
            object_name = source_dir.stem
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
        return image_config
    
    def __len__(self) -> int:
        return len(self.image_info_cfg)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str,str]]:
        """Get an item from the dataset"""
        path = self.image_info_cfg[idx]['image_path']
        image = Image.open(path).convert('RGB')
        image = ImageProcessor.preprocess_image(image, self.target_size)

        if self.transform:
            image = self.transform(image)
        
        return image, self.image_info_cfg[idx]
    



class SceneDataset(torch.utils.data.Dataset):

    def __init__(self, data_dir:str, cfg_file_path:str, transform: Optional[transforms.Compose] = None) -> None:
        self.data_dir = data_dir
        self.dataset_type = 'Scene'
        self.transform = transform
        with open(cfg_file_path, 'r') as f:
            dict_tmp = json.load(f)
            self.cfg_info = dict_tmp['images']
        
        self.image_info = self._load_image_info()
    
    def _load_image_info(self) -> Dict[str, Any]:
        
        list_image_info = []
        self.cfg_info = sorted(self.cfg_info, key= lambda x: int(x['id']))

        for cfg in self.cfg_info:
            filename = cfg['filename']
            id = cfg['id']
            mode, type, img_name, extension = filename.split('.')
            image_name = img_name + '.' + extension
            image_dir = os.path.join(self.data_dir, mode, type, image_name)

            new_cfg = {
                'dataset_type': 'Scene',
                'data_dir' : self.data_dir,
                'id': id,
                'mode': mode,
                'type': type,
                'image_path': image_dir

            }
            list_image_info.append(new_cfg)
        return list_image_info
        
        
    def __len__(self) -> int:
        return len(self.image_info)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, Dict[str, Any]]:

        image_cfg = self.image_info[idx]
        image_path = image_cfg['image_path']
        image = Image.open(image_path).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)
        return image, image_cfg




