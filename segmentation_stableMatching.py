import os
import argparse
import pandas as pd
import torch
from dataclasses import dataclass
from model import SegmentModel
from faisss import FaissIndexStrategy  # Ensure this is correctly named as 'faiss.py'
from typing import Dict, Any, Optional, Tuple, List
from enum import Enum
import json
import logging
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import torch.distributed as dist
from pathlib import Path
from datetime import timedelta, datetime
from extraction import FeatureExtractor  # Ensure this is correctly implemented
import numpy as np
import random
import torch.multiprocessing as mp
from PIL import Image, ImageDraw, ImageFont


# ====================== ENUMS & DATACLASSES ======================
class ModelType(Enum):
    CLIP = "CLIP"
    DINOV2 = "DINOV2"

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
    k_return:int


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
    distributed:bool
    world_size:int 
    seed:int=42



# ========================= HELPER FUNCTIONS =========================

def optimized_search_and_match(
  faiss_index_strategy: FaissIndexStrategy,
  features_np: np.ndarray,
  faiss_config: FaissConfig,
  logger: logging.Logger,
  initial_k: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    logger.info(f"Starting optimized search and match with initial_k={initial_k}")
    logger.debug(f"Input features shape: {features_np.shape}")

    metric = faiss_config.metric.lower()
    logger.info(f"Using metric: {metric}")

    total_vectors = faiss_index_strategy.index.ntotal
    logger.info(f"Total vectors in FAISS index: {total_vectors}")

    logger.info(f"Performing initial search with k={min(initial_k, total_vectors)}")
    initial_distances, initial_indices = faiss_index_strategy.search(
        features_np, k=min(initial_k, total_vectors)
    )
    logger.debug(f"Initial distances shape: {initial_distances.shape}")
    logger.debug(f"Initial indices shape: {initial_indices.shape}")


    logger.info("Preparing preference matrix")
    preference_mat = np.full(
        (features_np.shape[0], total_vectors), -np.inf
    )
    logger.debug(f"Preference matrix shape: {preference_mat.shape}")

    # Fill preference matrix
    for i in range(features_np.shape[0]):
        if metric == 'cosine':
            preference_mat[i, initial_indices[i]] = initial_distances[i]
        elif metric == 'l2':
            preference_mat[i, initial_indices[i]] = 1 / (initial_distances[i] + 1e-6)
        else:
            logger.error(f"Unsupported metric: {metric}")
            raise ValueError(f"Unsupported metric: {metric}")
        
        if i < 5:  #
            logger.debug(f"Sample preference values for row {i}: {preference_mat[i, initial_indices[i]][:5]}")

    logger.info("Applying initial stable matching")
    engagement_mat = stable_matching(preference_mat)
    logger.debug(f"Initial engagement matrix shape: {engagement_mat.shape}")
    logger.debug(f"Initial engagement matrix sum: {engagement_mat.sum()}")


    unmatched_rois = np.where(engagement_mat.sum(axis=1) == 0)[0]
    logger.info(f"Number of unmatched ROIs: {len(unmatched_rois)}")
    if len(unmatched_rois) > 0:
        logger.debug(f"Unmatched ROI indices: {unmatched_rois}")

    if len(unmatched_rois) > 0:
        logger.info(f"Performing full search for {len(unmatched_rois)} unmatched ROIs")
        full_distances, full_indices = faiss_index_strategy.search(
            features_np[unmatched_rois], k=total_vectors
        )
        logger.debug(f"Full distances shape: {full_distances.shape}")
        logger.debug(f"Full indices shape: {full_indices.shape}")

        for i, roi_idx in enumerate(unmatched_rois):
            if metric == 'cosine':
                preference_mat[roi_idx] = full_distances[i]
            elif metric == 'l2':
                preference_mat[roi_idx] = 1 / (full_distances[i] + 1e-6)
            
            if i < 5:  
                logger.debug(f"Updated preference values for unmatched ROI {roi_idx}: {preference_mat[roi_idx][:5]}")

        logger.info("Applying final stable matching")
        engagement_mat = stable_matching(preference_mat)
        logger.debug(f"Final engagement matrix shape: {engagement_mat.shape}")
        logger.debug(f"Final engagement matrix sum: {engagement_mat.sum()}")

    matched_rois = np.where(engagement_mat.sum(axis=1) > 0)[0]
    logger.info(f"Final number of matched ROIs: {len(matched_rois)}")
    logger.info(f"Final number of unmatched ROIs: {features_np.shape[0] - len(matched_rois)}")

    return engagement_mat, initial_indices






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
        seed = config_dict.get('seed',42)
    )
    return config

def stable_matching(preference_mat: np.ndarray):
    """Compute stable matching 

    Args:
        preference_mat (np.ndarray): Preference matrix where each row represents an ROI's preference over object vectors
    Returns:
        np.ndarray: engagement matrix with matched pairs

    """
    m, _ = preference_mat.shape
    engagement_matrix  = np.zeros_like(preference_mat, dtype=int)
    preferences = [
        list(np.argsort(-preference_mat[i]) for i in range(m))
    ]
    free_rois = list(range(m))
    current_matches = {}

    while free_rois:
        roi = free_rois.pop(0)
        if not preferences[roi]:
            continue # no more preferences to propose
            
        obj = preferences[roi].pop
        if obj not in current_matches:
            current_matches[obj] = roi
            engagement_matrix[roi, obj] = 1
        else:
            current_roi = current_matches[obj]
            if preference_mat[roi, obj] > preference_mat[current_roi, obj]:
                engagement_matrix[current_roi, obj] = 0
                current_matches[obj] = roi
                engagement_matrix[roi, obj] = 1
                free_rois.append(current_roi)
            else:
                free_rois.append(roi)
        
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
        ### DEBUG, only taking 1 image to test
        self.image_configs = image_configs[0]
        self.logger = logger

    def __len__(self):
        return len(self.image_configs)
    
    def __getitem__(self, idx: int):
        config = self.image_configs[idx]
        image_path = config['image_path']
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            self.logger.error(f"Failed to load image {image_path}: {e}")
            raise e
        return {
            'id': config['id'],
            'image': image,
            'bounding_boxes': config.get('bounding_boxes', []),
            'scores': config.get('scores', []),
            'phrases': config.get('phrases', []),
            'annotated_image_path': config.get('annotated_image_path', "")
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
        self.logger.info("Initializing SAM2.1 SegmentModel")
        self.segment_model = SegmentModel(
            checkpoint= model_checkpoint,
            model_cfg= model_config
        )
        self.logger.info("SAM2.1 Segment Model Initialized sucessfully")
    
    def masks_to_roi_images(
        self,
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
        roi_images = []
        for mask in masks:
            bbox = mask['bbox']
            x, y, width, height = map(int, bbox)
            
            cropped_image = original_image.crop(
                (x, y, x + width, y + height)
            )
            full_mask = mask['segmentation'].astype(np.uint8)
            cropped_mask = full_mask[y:y+height, x:x+width]
            mask_image = Image.fromarray(cropped_mask * 255).convert('L').resize(cropped_image.size)
            cropped_image_np = np.array(cropped_image)
            mask_np = np.array(mask_image) / 255.0

            if len(cropped_image_np.shape) == 2:
                cropped_image_np = np.stack([cropped_image_np]*3, axis=-1)
            elif cropped_image_np.shape[2] == 4:
                cropped_image_np = cropped_image_np[:,:,:3]

            masked_image_np = cropped_image_np * mask_np[:, :, np.newaxis]
            masked_image = Image.fromarray(masked_image_np.astype(np.uint8))

            roi_images.append(masked_image)
        return roi_images


    def extract_rois(
        self,
        image_path: str
    ) -> Tuple[List[Dict[str, Any]], List[Image.Image]]:
        """Extract rois from an image using SAM2.1 model and convert them to PIL images   

        Args:
            image_path (str): Path to the Image 

        Returns:
            Tuple[List[Dict[str, Any]], List[Image.Image]]: 
              - List of mask dictionaries
              - List of ROI images
        """
        self.logger.info(f"Loading image from {image_path}")
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            self.logger.error(f"Failed to load image {image_path}: {e}")
            raise e
        
        image_np = np.array(image)
        self.logger.debug(f"Image shape: {image_np.shape}")
        try:
            roi_masks = self.segment_model.predict(image_np)
            self.logger.info(f"Extracted {len(roi_masks)} ROIs from image {image_path}")
            roi_images = self.masks_to_roi_images(image, roi_masks)
            self.logger.info(f"Converted {len(roi_images)} masks to ROI images")
            return roi_masks, roi_images
        except Exception as e:
            self.logger.error(f"Failed to extract ROIs from {image_path}: {e}")
            raise e
    
def process_worker(rank:int, world_size:int, config: Config, return_list:List[Dict[str,Any]]) -> List[Dict[str, Any]]:
    """Worker function for each process

    Args:
        rank (int): rank
        world_size (int): how many processes
        config (Config): Configuration
        return_list (List[Dict[str,Any]]): _description_
    """
    args = parse_args()
    config = load_config(args.config)

    logger, _ = setup_logging(config.output_dir)
    logger.info("Starting Image ROI matching system")


    image_configs = load_json(config.data.image_config_path)
    logger.info(f"Loaded {len(image_configs)} image configurations")

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    if config.distributed:
        setup_distributed(rank, world_size, logger)
        torch.cuda.set_device(rank)
        logger.info(f"Rank {rank}: Set CUDA device to {rank}")
        device = torch.device(f'cuda:{rank}')
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

    try:
        roi_matcher = ROIMatching(
            model_checkpoint = config.models['SAM2'].pretrained_path,
            model_config = config.models['SAM2'].model_config_path,
            device=device,
            logger=logger
        )
    except Exception as e:
        logger.error("SAM2 model configuration 'sam2' not found in 'models'")
        raise e

    if 'feature_extractor' not in config.models:
        logger.error("Feature extractor configuration 'feature_extractor' not found in 'models'")
        return

    feature_model_config = config.models['feature_extractor']
    model_type_str = feature_model_config.type_model.lower()
    try:
        model_type = ModelConfig(model_type_str.lower())
    except Exception as e:
        logger.error(f"Unsupported feature extractor type: {model_type_str}")
        return
    
    try:
        feature_extractor = FeatureExtractor(
            model_type=model_type,
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
        faiss_index_strategy  = FaissIndexStrategy(
            index_type=faiss_config.index_type,
            dimension=faiss_config.dimension,
            use_gpu=faiss_config.use_gpu,
            device= faiss_config.device,
            metric=faiss_config.metric,
            nlist=faiss_config.nlist,
            M=faiss_config.M,
            nbits=faiss_config.nbits,
            nprobe=faiss_config.nprobe
        )
        faiss_index_strategy.load(faiss_config.index_load_path)
        logger.info("FAISS index loaded sucessfully!")
    except Exception as e:
        logger.error(f"Failed to initialize FAISS index strategy: {e}")
        return
    
    try:
        index2category = load_json(config.data.index2category_path)
        index2cat = index2category['object_name']
    except Exception as e:
        logger.error(f"Failed to load index2category mapping: {e}")
        return

    try:
        dataset = ImageDataset(
            image_configs=config.data.image_config_path,
            logger=logger
        )
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
    
    data_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=sampler,
        num_workers=2,
        pin_memory=True
    )

    results = []
    
    for batch in data_loader:
        image_id = batch['id'][0]
        image_path = batch['image_path'][0]
        image = batch['image'][0]
        bboxes = batch['bounding_boxes'][0]

        try:
            roi_masks, roi_images = roi_matcher.extract_rois(
                image_path=image_path
            )
            if not roi_images:
                logging.warning(f"No ROIs found! for image ID: {image_id}")
                continue
                
            features: torch.Tensor = feature_extractor.extract_features(roi_images)
            if not isinstance(features, torch.Tensor):
                logger.error(f"Feature extraction returned non-tensor for image ID: {image_id}")
                continue
            features_np = features.cpu().detach().numpy()
            
            if features_np.size == 0:
                logger.warning(f"No features extracted for image ID: {image_id}")
                continue

            engagement_matrix, indices = optimized_search_and_match(
                faiss_index_strategy=faiss_index_strategy,
                features_np=features_np,
                faiss_config=faiss_config,
                logger=logger,
                initial_k= faiss_config.k_return
            )

            for roi_idx, mask in enumerate(roi_masks):
                matched_obj_indices = np.where(engagement_matrix[roi_idx] == 1)[0]
                if len(matched_obj_indices) == 0:
                    logger.info(f"No match found for ROI {roi_idx} in image {image_id}")
                    continue
            
                obj_idx = matched_obj_indices[0]
                faiss_index = indices[roi_idx][obj_idx]
                category_id = index2cat[faiss_index].split('_')[0]
                old_bounding_box = bboxes[roi_idx]

                bounding_box = mask['bbox']
                for box in bounding_box:
                    x, y= box[0], box[1]
                    x0, y0 = old_bounding_box[0], old_bounding_box[1]
                    x = x + x0
                    y = y + y0

                    box[0] = x
                    box[1] = y
                

                plot_boxes_to_image(
                    image,
                    [bounding_box],
                    [category_id],
                    os.path.join(config.data.output_dir, "annotated_images", f"{image_id}.png")
                )
                results.append({
                    'image_id': image_id,
                    'category_id': category_id,
                    'bounding_box': bounding_box,
                    'scale': 1  
                })
        except Exception as e:
            logger.error(f"Error processing image ID: {image_id}: {e}")
    
    all_results = [None] * world_size 
    dist.all_gather_object(all_results, results)
    if rank == 0:
        # Flatten the list of results from all processes
        flattened_results = [item for sublist in all_results for item in sublist]
        return flattened_results
    else:
        return []
def main_worker(rank: int, world_size: int, config: Config, return_list: List[Dict[str, Any]]):
    """
    Worker function to be spawned via torch.multiprocessing.

    Args:
        rank (int): Rank of the process.
        world_size (int): Total number of processes.
        config (Config): Configuration.
        return_list (List[Dict[str, Any]]): Shared list to collect results.
    """
    process_worker(rank, world_size, config, return_list)

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

    if return_list:
        df = pd.DataFrame(return_list)
        output_path = os.path.join(config.output_dir, "roi_matching_results.csv")
        os.makedirs(config.output_dir, exist_ok=True)
        save_dataframe(df, output_path)
        logger.info(f"ROI Matching results saved to {output_path}")
    
    else:
        logger.warning("No results to save")


if __name__ == "__main__":
    main()
        


    

    



      

    


    
         



