import os
import argparse
import pandas as pd
import torch
from dataclasses import dataclass
from model import SegmentModel
from faisss import FaissIndexStrategy 
from typing import Dict, Any, Optional, Tuple, List
from enum import Enum
import json
from tqdm import tqdm
import logging
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import torch.distributed as dist
from pathlib import Path
from datetime import timedelta, datetime
from extraction import FeatureExtractor 
import numpy as np
import random
import torch.multiprocessing as mp
from PIL import Image, ImageDraw, ImageFont
import torchvision
from torchvision import transforms as T

# ====================== ENUMS & DATACLASSES ======================


@dataclass
class ModelConfig:
    name:Optional[str] = None
    type_model:Optional[str] = None
    pretrained_path:Optional[str] = None
    model_config_path:Optional[str] = None

@dataclass
class FaissConfig:
    index_load_path: str
    index_type:str
    dimension:int
    use_gpu:bool
    device:int
    metric:str
    nlist:Optional[int] = None
    M: Optional[int] = None
    nbits: Optional[int] = None
    nprobe: Optional[int] = None
    k_return:int = 200


@dataclass
class DataConfig:
    image_config_path:str
    output_dir:str
    index2category_path:str


@dataclass
class Config:
    models: Dict[str, ModelConfig]
    faiss: FaissConfig
    data: DataConfig
    output_dir: str
    sam_threshold:int
    distributed:bool
    world_size:int 
    seed:int=42



# ========================= HELPER FUNCTIONS =========================

def perform_nms(boxes: List[torch.Tensor], scores: List[float], iou_threshold: float = 0.5) -> List[int]:
    if not boxes:
        return []
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if isinstance(boxes[0], torch.Tensor):
        boxes_tensor = torch.stack(boxes).to(device)
    else:
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32).to(device)

    scores_tensor = torch.tensor(scores, dtype=torch.float32).to(device)

    keep_indices = torchvision.ops.nms(boxes_tensor, scores_tensor, iou_threshold)
    return keep_indices

def optimized_search_and_match(
faiss_index_strategy: FaissIndexStrategy,
features_np: np.ndarray,
faiss_config: FaissConfig,
logger: logging.Logger,
initial_k: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    # logger.info(f"Starting optimized search and match with initial_k={initial_k}")
    # logger.info(f"Input features shape: {features_np.shape}")

    metric = faiss_config.metric.lower()
    # logger.info(f"Using metric: {metric}")

    total_vectors = faiss_index_strategy.index.ntotal
    # logger.info(f"Total vectors in FAISS index: {total_vectors}")
    if total_vectors == 0:
        logger.error("FAISS index is empty!")
        raise ValueError("FAISS index contains no vectors")
    
    try:
        # logger.info(f"Performing initial search with k={min(initial_k, total_vectors)}")
        initial_distances, initial_indices = faiss_index_strategy.search(
            features_np, k=min(initial_k, total_vectors)
        )
        # logger.info(f"Initial distances shape: {initial_distances.shape}")
        # logger.info(f"Initial indices shape: {initial_indices.shape}")
        logger.debug(f"Distance range: [{initial_distances.min():.4f}, {initial_distances.max():.4f}]")

    except Exception as e:
        logger.error(f"FAISS search failed: {str(e)}")
        raise e
    
    logger.debug("Preparing preference matrix")
    try:
        preference_mat = np.full(
            (features_np.shape[0], total_vectors), -np.inf
        )
        # logger.info(f"Preference matrix shape: {preference_mat.shape}")
    except Exception as e:
        logger.error(f"Failed to create preference matrix: {str(e)}")
        raise

    # Fill preference matrix
    # logger.info("Filling preference matrix")
    for i in range(features_np.shape[0]):
        try:
            if metric == 'cosine':
                preference_mat[i, initial_indices[i]] = initial_distances[i]
            elif metric == 'l2':
                preference_mat[i, initial_indices[i]] = 1 / (initial_distances[i] + 1e-6)
            else:
                logger.error(f"Unsupported metric: {metric}")
                raise ValueError(f"Unsupported metric: {metric}")
            if i < 5:  
                logger.debug(f"Sample preference values for row {i}: {preference_mat[i, initial_indices[i]][:5]}")
        except Exception as e:
            logger.error(f"Error filling preference matrix at row {i}: {str(e)}")
            raise
    

    # logger.info("Applying initial stable matching")
    try:
        engagement_mat = stable_matching(preference_mat, logger=logger)
        # logger.info(f"Initial engagement matrix shape: {engagement_mat.shape}")
        # logger.info(f"Initial engagement matrix sum: {engagement_mat.sum()}")
    except Exception as e:
        logger.error(f"Stable matching failed: {str(e)}")
        raise 

    unmatched_rois = np.where(engagement_mat.sum(axis=1) == 0)[0]
    logger.debug(f"Number of unmatched ROIs: {len(unmatched_rois)}")
    if len(unmatched_rois) > 0:
        logger.debug(f"Unmatched ROI indices: {unmatched_rois}")

    if len(unmatched_rois) > 0:
        logger.debug(f"Performing full search for {len(unmatched_rois)} unmatched ROIs")
        try:
            full_distances, full_indices = faiss_index_strategy.search(
                features_np[unmatched_rois], k=total_vectors
            )
            logger.debug(f"Full distances shape: {full_distances.shape}")
            logger.debug(f"Full indices shape: {full_indices.shape}")
        except Exception as e:
            logger.error(f"Full FAISS search failed: {str(e)}")
            raise

        for i, roi_idx in enumerate(unmatched_rois):
            if metric == 'cosine':
                preference_mat[roi_idx] = full_distances[i]
            elif metric == 'l2':
                preference_mat[roi_idx] = 1 / (full_distances[i] + 1e-6)
            
            if i < 5:  
                logger.debug(f"Updated preference values for unmatched ROI {roi_idx}: {preference_mat[roi_idx][:5]}")

        logger.debug("Applying final stable matching")
        try:
            engagement_mat = stable_matching(preference_mat)
            logger.debug(f"Final engagement matrix shape: {engagement_mat.shape}")
            logger.debug(f"Final engagement matrix sum: {engagement_mat.sum()}")
        except Exception as e:
            logger.error(f"Final stable matching failed: {str(e)}")
            raise
        

    matched_rois = np.where(engagement_mat.sum(axis=1) > 0)[0]
    # logger.info(f"Final number of matched ROIs: {len(matched_rois)}")
    # logger.info(f"Final number of unmatched ROIs: {features_np.shape[0] - len(matched_rois)}")
    
    if len(matched_rois) == 0:
        logger.warning("No ROIs were matched in the final result!")

    return engagement_mat, initial_indices, preference_mat


def load_json(file_path: str) -> Any:
    import json
    with open(file_path, 'r') as f:
        return json.load(f)
    
def save_dataframe(df: pd.DataFrame, path:str) -> None:
    df.to_csv(path, index=False)
    
def load_config(config_path: str)  -> Config:
    
    config_dict = load_json(config_path)

    config = Config(
        models= {k: ModelConfig(**v) for k, v in config_dict['models'].items()},
        faiss=FaissConfig(**config_dict['faiss']),
        data = DataConfig(**config_dict['data']),
        output_dir= config_dict['output_dir'],
        distributed=config_dict['distributed'],
        world_size=config_dict['world_size'],
        seed = config_dict.get('seed',42),
        sam_threshold= config_dict.get('sam_threshold', 0.5)
    )
    return config

def stable_matching(preference_mat: np.ndarray, logger: logging.Logger):
    """Compute stable matching 

    Args:
        preference_mat (np.ndarray): Preference matrix where each row represents an ROI's preference over object vectors
    Returns:
        np.ndarray: engagement matrix with matched pairs

    """
    m, n = preference_mat.shape
    logger.debug(f"Input preference matrix shape: ({m}, {n})")
    logger.debug("Initialized engagement matrix with zeros")
    engagement_matrix = np.zeros_like(preference_mat, dtype=int)
    logger.debug("Initialized engagement matrix with zeros")
    logger.debug("Creating sorted preference lists for each ROI")
    preferences = [list(np.argsort(-preference_mat[i])) for i in range(m)]
    logger.debug(f"Sample preference list (ROI 0): {preferences[0][:5]}...")
    
    free_rois = list(range(m))
    logger.debug(f"Initialized {len(free_rois)} free ROIs")
    
    current_matches = {}
    logger.debug("Starting matching iterations")

    iteration = 0
    while free_rois:
        iteration += 1
        if iteration % 1000 == 0:
            logger.debug(f"Iteration {iteration}: {len(free_rois)} ROIs remaining to match")
            
        roi = free_rois.pop(0)
        logger.debug(f"Processing ROI {roi}")
        
        if not preferences[roi]:
            logger.warning(f"ROI {roi} has exhausted all preferences")
            continue
            
        obj = preferences[roi].pop(0)
        logger.debug(f"ROI {roi} proposing to object {obj}")
        
        if obj not in current_matches:
            logger.debug(f"Object {obj} is free, matching with ROI {roi}")
            current_matches[obj] = roi
            engagement_matrix[roi, obj] = 1
        else:
            current_roi = current_matches[obj]
            logger.debug(f"Object {obj} is currently matched to ROI {current_roi}")
            
            if preference_mat[roi, obj] > preference_mat[current_roi, obj]:
                logger.debug(f"New ROI {roi} preferred over current ROI {current_roi}")
                engagement_matrix[current_roi, obj] = 0
                current_matches[obj] = roi
                engagement_matrix[roi, obj] = 1
                free_rois.append(current_roi)
            else:
                logger.debug(f"Current ROI {current_roi} preferred over new ROI {roi}")
                free_rois.append(roi)
    # print("Engagement matrix: ", engagement_matrix)
    final_matches = np.sum(engagement_matrix)
    logger.debug(f"Stable matching completed with {final_matches} total matches")
    logger.debug(f"Matches per ROI: {np.sum(engagement_matrix, axis=1)[:5]}... (showing first 5)")
    logger.debug(f"Matches per object: {np.sum(engagement_matrix, axis=0)[:5]}... (showing first 5)")
    
    unmatched_rois = np.where(np.sum(engagement_matrix, axis=1) == 0)[0]
    if len(unmatched_rois) > 0:
        logger.warning(f"Found {len(unmatched_rois)} unmatched ROIs")
        if len(unmatched_rois) < 10:
            logger.warning(f"Unmatched ROI indices: {unmatched_rois}")
    
    return engagement_matrix
        

def setup_logging(output_dir: str, rank: Optional[int] = None) -> Tuple[logging.Logger, Path]:
    """
    Setup logging configuration with both file and console handlers.

    Args:
        output_dir (str): Directory for log files.
        rank (Optional[int]): Process rank for distributed training.

    Returns:
        Tuple[logging.Logger, Path]: Configured logger and path to the log file.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    rank_suffix = f"_rank{rank}" if rank is not None else ""
    log_file = log_dir / f"process_log_{timestamp}{rank_suffix}.txt"
    
    logger = logging.getLogger(f"Rank{rank}" if rank is not None else "Main")
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(levelname)s: %(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    return logger, log_file


def save_tensor_images(images_tensor: torch.Tensor, 
                      masks_tensor: torch.Tensor, 
                      save_dir: str = "tensor_visualization"):
    
    from pathlib import Path
    
    base_dir = Path(save_dir)
    image_dir = base_dir / "images"
    mask_dir = base_dir / "masks"
    
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    
    images_tensor = images_tensor.cpu()
    masks_tensor = masks_tensor.cpu()
    
    to_pil = T.ToPILImage()
    
    for idx in range(len(images_tensor)):
        img = to_pil(images_tensor[idx])
        img.save(image_dir / f"image_{idx:03d}.png")
        
        mask = masks_tensor[idx]
        if mask.dim() > 2:
            mask = mask.squeeze()
        mask = mask.numpy()
        mask = (mask*255).astype(np.uint8)
        mask_img = Image.fromarray(mask, mode='L')
        mask_img.save(mask_dir / f"mask_{idx:03d}.png")
        
        if idx % 10 == 0:
            print(f"Saved {idx}/{len(images_tensor)} images")
    
    print(f"Saved {len(images_tensor)} images and masks to {save_dir}")

def setup_distributed(rank:int, world_size:int, logger:logging.Logger, timeout:int=3600) -> None:
    """Initialize distributed training

    Args:
        rank (int): Rank of the current process
        world_size (int): Total number of processes
        logger (logging.logger): Logging instances
    """
    try:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group(
            backend="nccl", 
            rank=rank, 
            world_size=world_size,
            timeout=timedelta(seconds=timeout)
        )
        logger.info(f"Distributed process group initialized with rank {rank}/{world_size}")
        torch.cuda.set_device(rank)
        
    except Exception as e:
        logger.error(f"Failed to initialize distributed process group: {e}")
        raise e

def cleanup(logger: logging.Logger)->None:
    """Clean up the distributed training

    Args:
        logger (logging.logger): Logger instance
    """
    dist.destroy_process_group()
    logger.info("Destroyed distributed process group")

def parse_args():
    parser = argparse.ArgumentParser(description="Image ROI Matching System")
    parser.add_argument('--config', type=str, required=True, help="Path to the main configuration JSON file")
    return parser.parse_args()



def plot_boxes_to_image(
    image_pil: Image.Image,
    boxes: List[int],
    classes: List[str],
    output_path: str
) -> None:
    """
    Draw bounding boxes and phrases on the image and save it.

    Args:
        image_pil (Image.Image): The original PIL image, before resizing and put into the model.
        boxes (torch.Tensor): Boxes under tensor type, in xyxy format and normalized between (0,1).
        classes (List[str]): List of classes.
        output_path (str): Path to save the annotated image.
    """
    draw = ImageDraw.Draw(image_pil)
    
    try:
        font = ImageFont.truetype("/kaggle/input/fontttt/Arial.ttf", size=20)
    except IOError:
        font = ImageFont.load_default()
    
    for box, class_ in zip(boxes, classes):
        x0, y0, x1, y1 = map(int, box)
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        draw.rectangle([x0, y0, x1, y1], outline=color, width=10)
        text = class_
        _, _, text_width, text_height = draw.textbbox((0, 0), text=text, font=font)
        draw.rectangle([x0, y0 - text_height, x0 + text_width, y0], fill=color)
        draw.text((x0, y0 - text_height), text, fill="white", font=font)
    image_pil.save(output_path)
    print(f"Annotated image saved to {output_path}")


# ============================ DATASET ==============================
class ImageDataset(Dataset):
    def __init__(self, image_configs: List[Dict[str,Any]], logger: logging.Logger):
        self.image_configs = image_configs
        self.logger = logger
        # self.logger.info(f"Finding {len(self.image_configs)} number of image(s)")
        self.transform = torchvision.transforms.Compose([

            torchvision.transforms.ToTensor(),
        ])
    def __len__(self):
        return len(self.image_configs)
    
    def __getitem__(self, idx: int):
        config = self.image_configs[idx]
        image_path = config['image_path']
        
        try:
            with Image.open(image_path).convert('RGB') as img:
                image_tensor = self.transform(img)
            
            bounding_boxes = torch.tensor(config.get('bounding_boxes', []), 
                                       dtype=torch.float32) if config.get('bounding_boxes') else []
            
            scores = torch.tensor(config.get('scores', []), 
                               dtype=torch.float32) if config.get('scores') else []
            
            return {
                'id': config['id'],
                'image': image_tensor,  
                'bounding_boxes': bounding_boxes,
                'scores': scores,
                'phrases': config.get('phrases', []),
                'annotated_image_path': config.get('annotated_image_path', ""),
                'image_path': config.get('image_path', [])
            }
            
        except Exception as e:
            self.logger.error(f"Failed to load/process image {image_path}: {e}")
            raise e
    
    
    @staticmethod
    def collate_fn(batch):
        ids = [item['id'] for item in batch]
        images = torch.stack([item['image'] for item in batch])
        phrases = [item['phrases'] for item in batch]
        annotated_paths = [item['annotated_image_path'] for item in batch]
        image_path = [item['annotated_image_path'] for item in batch]

        bounding_boxes = [item['bounding_boxes'] for item in batch]
        scores = [item['scores'] for item in batch]

        return {
            'id': ids,
            'image': images,
            'bounding_boxes': bounding_boxes,  
            'scores': scores,  
            'phrases': phrases,
            'annotated_image_path': annotated_paths,
            'image_path': image_path
        }
# ========================= ROI MATCHING ============================
class ROIMatching:
    def __init__(
        self,
        model_checkpoint: str,
        model_config: str,
        device: torch.device,
        logger: logging.Logger
    ):
        """
        Initialize the ROI Matcher with the SAM2.1 model.

        Args:
            model_checkpoint (str): Path to the SAM2.1 model checkpoint.
            model_config (str): Path to the SAM2.1 model configuration.
            device (torch.device): Device to run the model on.
            logger (logging.Logger): Logger instance.
        """
        self.logger = logger
        self.device = device
        # self.logger.info("Initializing SAM2.1 SegmentModel")
        if not os.path.exists(model_checkpoint):
            self.logger.error(f"Model checkpoint not found at: {model_checkpoint}")
            raise FileNotFoundError(f"Model checkpoint not found: {model_checkpoint}")
            
        if not os.path.exists(model_config):
            self.logger.error(f"Model config not found at: {model_config}")
            raise FileNotFoundError(f"Model config not found: {model_config}")
            
        try:
            self.segment_model = SegmentModel(
                checkpoint= model_checkpoint,
                model_cfg= model_config
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize SAM2.1 model: {str(e)}")
            raise

        # self.logger.info("SAM2.1 Segment Model Initialized sucessfully")
    
    def masks_to_roi_images(
        self,
        image_id,
        original_image: Image.Image,
        masks: List[Dict[str, Any]]
    ) -> List[Image.Image]:
        """Convert masks to ROI Images

        Args:
            original_image (Image.Image): The original PIl image
            masks (List[Dict[str, Any]]): List of mask dictionaries

        Returns:
            List[Image.Image]: List of ROI Images
        """
        # self.logger.info(f"Converting {len(masks)} masks to ROI images")
        # self.logger.info(f"Original image size: {original_image.size}")
        roi_images = []
        roi_images_masks = []
        roi_bounding_boxes = []
        
        for idx, mask in enumerate(masks):
            try:
                bbox = mask['bbox']
                x, y, width, height = map(int, bbox)
                self.logger.debug(f"Processing mask {idx} with bbox: x={x}, y={y}, w={width}, h={height}")
                if width <= 0 or height <= 0:
                    self.logger.warning(f"Invalid bbox dimensions for mask {idx}: width={width}, height={height}")
                    continue
                cropped_image = original_image.crop(
                    (x, y, x + width, y + height)
                )
                self.logger.debug(f"Cropped image size for mask {idx}: {cropped_image.size}")
                full_mask = mask['segmentation'].astype(np.uint8)
                # print("FULL_MAKS segmentation shape:", full_mask.shape)
                if full_mask.shape[0] < y + height or full_mask.shape[1] < x + width:
                    self.logger.error(f"Mask dimensions mismatch for mask {idx}")
                    continue
                cropped_mask = full_mask[y:y+height, x:x+width]
                cropped_image_np = np.array(cropped_image)
                self.logger.debug(f"Cropped image array shape: {cropped_image_np.shape}")

                if len(cropped_image_np.shape) == 2:
                    self.logger.debug("Converting grayscale image to RGB")
                    cropped_image_np = np.stack([cropped_image_np]*3, axis=-1)
                elif cropped_image_np.shape[2] == 4:
                    self.logger.debug("Converting RGBA image to RGB")
                    cropped_image_np = cropped_image_np[:,:,:3]
                
                roi_images.append(cropped_image_np)
                roi_images_masks.append(cropped_mask)
                roi_bounding_boxes.append(bbox)
                self.logger.debug(f"Successfully processed mask {idx}")
            except Exception as e:
                self.logger.error(f"Error processing mask {idx}: {str(e)}")
                continue
        # self.logger.info(f"Successfully converted {len(roi_images)} masks to ROI images")
        return roi_images, roi_images_masks, roi_bounding_boxes

    def _convert_images_to_tensor(self, images: List) -> torch.Tensor:
        """Convert list of PIL images or numpy arrays to tensor and resize to target size"""
        tensors = []
        image_transforms = T.Compose([
            T.Resize((224,224)),
            T.ToTensor(),
        ])
        for img in images:
            try:
                if isinstance(img, np.ndarray):
                    img = Image.fromarray(img)

                img_tensor = image_transforms(img)
                tensors.append(img_tensor)
            except Exception as e:
                print("Inside images to tensor: ", e)
                raise e
        
        return torch.stack(tensors).to(self.device)
    
    def _convert_masks_to_tensor(self, masks: List) -> torch.Tensor:
        """Convert list of segmentation masks to tensor and resize to target size"""
        tensors = []
        mask_transforms = T.Compose([
            T.Resize((224,224), interpolation=T.InterpolationMode.NEAREST)
        ])
        for mask in masks:
            try:
                if isinstance(mask, np.ndarray):
                    mask_img = Image.fromarray(mask.astype(np.uint8))
                else:
                    mask_img = Image.fromarray(np.array(mask).astype(np.uint8))

                mask_resized = mask_transforms(mask_img)

                mask_tensor = torch.from_numpy(np.array(mask_resized)).float()
                tensors.append(mask_tensor)
            except Exception as e:
                print("Inside mask to tensor: ", e)
                raise e
        
        return torch.stack(tensors).to(self.device)
    def extract_rois(
        self,
        image_tensor: torch.Tensor,
        bounding_boxes: List[List[int]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract rois from an image using SAM2.1 model and convert them to PIL images   

        Args:
            image_path (str): Path to the Image 

        Returns:
            Tuple[List[Dict[str, Any]], List[Image.Image]]: 
            - List of mask dictionaries
            - List of ROI images
        """

        try:
            image = torchvision.transforms.ToPILImage()(image_tensor)
            image_np = np.array(image)
            # self.logger.info(f"Converted tensor to image array of shape: {image_np.shape}")
        except Exception as e:
            self.logger.error(f"Failed to convert tensor to image: {e}")
            raise e
        
        all_roi_masks = []
        all_roi_images = []
        all_roi_boxes = []

        rois_idx_2_image = {}
        
        rois_id = 0
        for idx, bbox in enumerate(bounding_boxes):
            try:
                self.logger.debug(f"Processing bounding box {idx}: {bbox}")
                x1, y1, x2, y2 = map(int, bbox)
                
                cropped_image = image.crop((x1, y1, x2, y2))
                cropped_image.save(f'/kaggle/working/debug/image_cropped/image_{idx}.png')
                cropped_np = np.array(cropped_image)
                
                self.logger.debug(f"Extracting ROIs from bbox {idx}")
                roi_masks = self.segment_model.predict(cropped_np)
                self.logger.debug(f"Extracted {len(roi_masks)} ROIs from bbox {idx}")

                # for mask in roi_masks:
                #     mask_bbox = mask['bbox']
                #     mask_bbox[0] += x1  
                #     mask_bbox[1] += y1  
                #     all_roi_masks.append(mask)

                roi_images, roi_image_masks, roi_bounding_boxes = self.masks_to_roi_images(idx,cropped_image, roi_masks)
                
                roi_images_tensor = self._convert_images_to_tensor(roi_images)
                all_roi_images.append(roi_images_tensor)

                roi_masks_tensor = self._convert_masks_to_tensor(roi_image_masks)
                all_roi_masks.append(roi_masks_tensor)

                roi_boxes_tensor = torch.tensor(roi_bounding_boxes, dtype=torch.float32, device=self.device)
                all_roi_boxes.append(roi_boxes_tensor)

                for _ in range(len(roi_images)):
                    rois_idx_2_image[rois_id] = idx
                    rois_id += 1


                self.logger.info(f"Processed {len(roi_images)} ROI images from bbox {idx}")

            except Exception as e:
                self.logger.error(f"Error processing bbox {idx}: {e}")
                continue
        
        # if not all_roi_masks:
        #     self.logger.warning("No ROIs detected in any of the bounding boxes")
        # else:
        #     self.logger.info(f"Total ROIs extracted: {len(all_roi_masks)}")

        # print("INDEXX: ", rois_idx_2_image)    
        if all_roi_images and all_roi_masks:
            all_roi_images_tensor = torch.cat(all_roi_images, dim=0)
            all_roi_masks_tensor = torch.cat(all_roi_masks, dim=0)
            all_roi_boxes_tensor = torch.cat(all_roi_boxes, dim=0)
            # print("All roi images tensor shape:", all_roi_images_tensor.shape)
            # print("All roi masks tensor shape: ", all_roi_masks_tensor.shape)
            # print("All roi boxes tensor shape: ", all_roi_boxes_tensor.shape)
            return all_roi_images_tensor, all_roi_masks_tensor, all_roi_boxes_tensor, rois_idx_2_image
        else:
            return torch.empty(0), torch.empty(0), torch.empty(0), {}

    
def process_worker(rank:int, world_size:int, config: Config, return_list:List[Dict[str,Any]]) -> List[Dict[str, Any]]:
    """Worker function for each process

    Args:
        rank (int): rank
        world_size (int): how many processes
        config (Config): Configuration
        return_list (List[Dict[str,Any]]): _description_
    """
    logger, _ = setup_logging(config.output_dir, rank)
    logger.info(f"Process {rank}/{world_size-1} starting")
    logger.info(f"Using config: {config}")
    args = parse_args()
    config = load_config(args.config)
    try:
        image_configs = load_json(config.data.image_config_path)
        logger.info(f"Loaded {len(image_configs)} image configurations")
        # logger.info(f"First image config: {image_configs[0] if image_configs else 'No configs'}")
    except Exception as e:
        logger.error(f"Failed to load image configs: {str(e)}")
        return
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    logger.info(f"Set random seed to {config.seed}")
    try:
        if config.distributed:
            setup_distributed(rank, world_size, logger)
            torch.cuda.set_device(rank)
            logger.info(f"Rank {rank}: Set CUDA device to {rank}")
            device = torch.device(f'cuda:{rank}')
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
    except Exception as e:
        logger.error(f"Failed to setup device: {str(e)}")
        return

    try:
        logger.info("Initialzing ROI extracting from SAM2.1")
        
        roi_matcher = ROIMatching(
            model_checkpoint = config.models['SAM2'].pretrained_path,
            model_config = config.models['SAM2'].model_config_path,
            device=device,
            logger=logger
        )
    except KeyError as e:
        logger.error(f"Missing SAM2 configuration in models config: {str(e)}")
        return
    except Exception as e:
        logger.error(f"Failed to initialize ROI matcher: {str(e)}")
        return
    if 'feature_extractor' not in config.models:
        logger.error("Feature extractor configuration 'feature_extractor' not found in 'models'")
        return
    feature_model_config = config.models['feature_extractor']
    model_type_str = feature_model_config.type_model.upper()
    try:
        model_config = ModelConfig(model_type_str.upper())
    except Exception as e:
        logger.error(f"Unsupported feature extractor type: {model_type_str}")
        return
    
    try:
        logger.info(f"Initializing the Feature Extractor with type: {model_type_str}")
        logger.info(f"With Feature Model Config: {feature_model_config}")
        feature_extractor = FeatureExtractor(
            model_type=model_type_str,
            model_config=feature_model_config,
            rank=rank,
            config=config,
            logger=logger
        )
    except Exception as e:
        logger.error(f"Failed to initialize FeatureExtractor: {e}")
        raise e
    faiss_config = config.faiss
    try:
        logger.info("Initialzing Faiss Index")
        faiss_index_strategy  = FaissIndexStrategy()
        faiss_index_strategy.load(faiss_config.index_load_path)
        logger.info(f"FAISS index configuration - Type: {config.faiss.index_type}, "
                    f"Dimension: {config.faiss.dimension}, GPU: {config.faiss.use_gpu}")
    except Exception as e:
        logger.error(f"Failed to initialize FAISS index strategy: {e}")
        return
    
    try:
        logger.info(f"Loaded index2category from path{config.data.index2category_path}")
        index2category = load_json(config.data.index2category_path)
        index2cat = index2category['object_name']
    except Exception as e:
        logger.error(f"Failed to load index2category mapping: {e}")
        return
    try:
        logger.info("Initializing the ImageDataset")
        dataset = ImageDataset(
            image_configs=image_configs,
            logger=logger
        )
        logger.info(f"Complete initializing the dataset, len = {len(dataset)}")
    except Exception as e:
        logger.error(f"Failed to create ImageDataset: {e}")
        return
    if config.distributed:
        sampler = DistributedSampler(
            dataset=dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False
        )
        logger.info(f"Sampler Distributed has been set up with hyperparameters: {world_size, rank}")
    else:
        sampler=None
    logger.info("Setting up DataLoader")
    data_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=sampler,
        num_workers=2,
        pin_memory=True,
        collate_fn=dataset.collate_fn
    )
    results = []
    logger.info(f"Starting to process {len(dataset)} images")
    total_processed = 0
    total_successful = 0
    total_failed = 0
    save_image_path = os.path.join(config.data.output_dir, "annotated_images")
    os.makedirs(save_image_path, exist_ok=True)
    to_pil = T.ToPILImage()
    for batch in tqdm(data_loader, desc="Processing batch of images..."):
        image_id = batch['id'][0]
        image_path = batch['image_path'][0]
        image = batch['image'][0]
        image_PIL = to_pil(image)
        bboxes = batch['bounding_boxes'][0]
        image_name =  os.path.basename(os.path.split(image_path)[0]) + os.path.basename(image_path)
        # logger.info(f"Processing image ID: {image_id} from path: {image_path}")
        # logger.info(f"Image shape: {image.size() if hasattr(image, 'size') else 'unknown'}")
        # logger.info(f"Number of bounding boxes: {len(bboxes)}")

        try:
            logger.info(f"Extracting ROIs for image {image_id}")
            roi_extraction_start = datetime.now()
            roi_images, roi_masks, roi_bboxes,rois_idx_2_image = roi_matcher.extract_rois(
                image_tensor=image,
                bounding_boxes=bboxes
            )
            roi_extraction_time = (datetime.now() - roi_extraction_start).total_seconds()
            # logger.info(f"ROI extraction completed in {roi_extraction_time:.2f} seconds. Found {len(roi_images)} ROIs")
            
            save_tensor_images(roi_images, roi_masks, '/kaggle/working/debug')
            if roi_images is None or len(roi_images) == 0:
                logger.warning(f"No ROIs found for image ID: {image_id}")
                total_failed += 1
                continue
                
            # logger.info(f"Extracting features for {len(roi_images)} ROIs")
            feature_extraction_start = datetime.now()
            feature_matrix = torch.from_numpy(feature_extractor.extract_features(roi_images, roi_masks))
            print("Feature matrix shape: ", feature_matrix.shape)          
            feature_extraction_time = (datetime.now() - feature_extraction_start).total_seconds()
            # logger.info(f"Feature extraction completed in {feature_extraction_time:.2f} seconds")
            if not isinstance(feature_matrix, torch.Tensor):
                logger.error(f"Feature extraction returned non-tensor type {type(feature_extraction_start)} for image ID: {image_id}")
                total_failed += 1
                continue
            
            features_np = feature_matrix.cpu().detach().numpy()
            # logger.info(f"Feature shape: {features_np.shape}")
            
            if features_np.size == 0:
                logger.warning(f"No features extracted for image ID: {image_id}")
                total_failed += 1
                continue

            # logger.info("Starting optimized search and match")
            
            matching_start = datetime.now()
            engagement_matrix, _, preference_mat = optimized_search_and_match(
                faiss_index_strategy=faiss_index_strategy,
                features_np=features_np,
                faiss_config=faiss_config,
                logger=logger,
                initial_k=faiss_config.k_return
            )
            
            matching_time = (datetime.now() - matching_start).total_seconds()
            # logger.info(f"Search and match completed in {matching_time:.2f} seconds")

            # logger.info("\n====Detail matching  =======\n")
            roi_to_vec = {}
            for roi_idx in range(preference_mat.shape[0]):
                match_indices = np.where(engagement_matrix[roi_idx] == 1)[0]
                if len(match_indices) > 0:
                    roi_to_vec[roi_idx] = match_indices[0]
            # logger.info("\nROI to Vector Mapping:")
            # logger.info("Format: ROI_ID -> Matched_Vector_Index (Similarity_Score)")
            
            sorted_rois = sorted(roi_to_vec.keys())

            boxes = []
            scores = []
            categories = []
            classes_name = []


            for roi_idx in sorted_rois:
                vector_idx = roi_to_vec[roi_idx]
                score = preference_mat[roi_idx, vector_idx]
                # print("ROI index: ", roi_idx)
                # print("Vector index: ", vector_idx)
                # print("Category: ", index2cat[vector_idx])
                # print("Score index: ", score)
                if score < config.sam_threshold:
                    logger.debug("Confidence score to low, skip")
                    continue
                category_id = index2cat[vector_idx].split('_')[0]
                category_class = ' '.join(index2cat[vector_idx].split('_')[1:])
                old_bounding_box = bboxes[rois_idx_2_image[roi_idx]]
                logger.debug(f"ROI {roi_idx} matched to category: {category_id}")
               
                bounding_box = roi_bboxes[roi_idx]
                logger.debug(f"Original Bounding box: {old_bounding_box}")
                logger.debug(f"Bounding box, relatives to the image {bounding_box} ")
                
                
                x, y= bounding_box[0], bounding_box[1]
                x0, y0 = old_bounding_box[0], old_bounding_box[1]
                x = x + x0
                y = y + y0
                bounding_box[0] = x
                bounding_box[1] = y


                bounding_box[2] = bounding_box[2] + bounding_box[0]
                bounding_box[3] = bounding_box[3] + bounding_box[1]
                # logger.info(f"Transformed bbox: {bounding_box}")

                boxes.append(bounding_box)
                scores.append(score)
                categories.append(category_id)
                classes_name.append(category_class)

            if boxes:
                # logger.info(f"Number of boxes before nms: {len(boxes)}")
                keep_indices = perform_nms(boxes, scores, 0.5)
                # logger.info(f"Number of boxes after nms: {len(keep_indices)}")
                kept_boxes = []
                kept_classes = []
                for idx in keep_indices:
                    results.append({
                        'image_id': image_id,
                        'category_id': categories[idx],
                        'bounding_box': boxes[idx],
                        'score': scores[idx],
                        'scale': 1
                    })
                    kept_boxes.append(boxes[idx])
                    kept_classes.append(classes_name[idx])
                try:
                    plot_boxes_to_image(
                        image_PIL,
                        kept_boxes,
                        kept_classes,
                        os.path.join(save_image_path,image_name)
                    )
                    logger.debug(f"Successfully saved annotated image for ROI {roi_idx}")
                except Exception as e:
                    logger.error(f"Failed to save annotated image for ROI {roi_idx}: {str(e)}")
                
               
                matches_found = True
                if matches_found > 0:
                    total_successful += 1
                else:
                    total_failed += 1
        except Exception as e:
            logger.error(f"Error processing image ID {image_id}: {str(e)}", exc_info=True)
            total_failed += 1
            continue
        total_processed += 1
        if total_processed % 10 == 0:
                logger.info(f"Progress Update - Processed: {total_processed}, "
                          f"Successful: {total_successful}, Failed: {total_failed}, ")
    if total_processed > 0:
        logger.info(f"Processing Complete - Total Images: {total_processed}, "
                   f"Successful: {total_successful}, Failed: {total_failed}")
    if config.distributed:
        all_results = [None] * world_size
        logger.info("Gathering results from all processes")
        try:
            cpu_results = []
            for result in results:
                cpu_result = {}
                for k,v in result.items():
                    if isinstance(v, torch.Tensor):
                        cpu_result[k] = v.cpu()
                    else:
                        cpu_result[k] = v
                cpu_results.append(cpu_result)
            dist.all_gather_object(all_results,cpu_results)
           
        except Exception as e:
            logger.error(f"Failed to gather results from all processes due to e: {e}")
            raise e
        
        if rank == 0:
            logger.info("Process 0: Flattening results from all processes")
            flattened_results = [item for sublist in all_results for item in sublist]

            logger.info(f"Total results gathered: {len(flattened_results)}")
            simplified_results = []
            for result in flattened_results:
                simplified_result = {}
                for k, v in result.items():
                    if isinstance(v, (torch.Tensor, np.ndarray)):
                        simplified_result[k] = v.tolist()
                    else:
                        simplified_result[k] = v
                simplified_results.append(simplified_result)

            return_list.extend(simplified_results)
            return simplified_results
    else:
        simplified_results = []
        for result in results:
            simplified_result = {}
            for k, v in result.items():
                if isinstance(v, (torch.Tensor, np.ndarray)):
                    simplified_result[k] = v.tolist()
                else:
                    simplified_result[k] = v
            simplified_results.append(simplified_result)
        return_list.extend(simplified_results)
        return simplified_results

    logger.info(f"Process {rank}: Returning empty results list")
    return []

def main_worker(rank: int, world_size: int, config: Config, return_list: List[Dict[str, Any]]):
    results = process_worker(rank, world_size, config, return_list)
def main():
    args = parse_args()
    config = load_config(args.config)
    logger, _ = setup_logging(config.output_dir)
    logger.info("Starting Image ROI Matching system")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    manager = mp.Manager()
    return_list = manager.list()
    try:
        mp.spawn(
            fn=main_worker,
            args=(config.world_size, config, return_list),
            nprocs=config.world_size,
            join=True
        )
    except Exception as e:
        logger.error(f"Failed during multiprocessing spawn: {e}")
        raise e
    
    logger.info("All processes have finished. Aggregating results.")
    return_list = list(return_list)
    results = list(return_list)
    if results and any(results):  
        df = pd.DataFrame(results)
        output_path = os.path.join(config.output_dir, "roi_matching_results.csv")
        os.makedirs(config.output_dir, exist_ok=True)
        save_dataframe(df, output_path)
        logger.info(f"ROI Matching results saved to {output_path}")
    else:
        logger.warning("No results to save")
if __name__ == "__main__":
    main()