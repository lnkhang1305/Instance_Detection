import os
import json
import torch
from typing import Optional, List, Tuple, Dict, Any
import torch.distributed as dist
from datetime import timedelta
from dataclasses import dataclass
import logging
from pathlib import Path
from datetime import datetime
import torch.distributed
from torchvision import transforms
from datasets import SceneDataset
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from utils import plot_boxes_to_image
from PIL import Image
from GroundingDINO.groundingdino.datasets.transforms import Compose, RandomResize, ToTensor, Normalize
import argparse

from model import GroundingDinoClass


@dataclass
class Config:
    config_file: str
    checkpoint_path: str
    data_dir: str
    data_config_file: str
    text_prompt: str
    output_dir: str
    box_threshold: float
    text_threshold: float
    cpu_only: float
    batch_size: int
    num_workers: int 
    nms_threshold: float = 0.5
    world_size: int = torch.cuda.device_count()


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

def cleanup(logger: logging.Logger)->None:
    """Clean up the distributed training

    Args:
        logger (logging.logger): Logger instance
    """
    dist.destroy_process_group()
    logger.info("Destroyed distributed process group")


def custom_collate_fn(
    batch: List[Tuple[torch.Tensor, Dict[str, Any]]]
) -> Dict[str, Any]:
    """Custom collate fn to handle PIl image

    Args:
        batch (List[Tuple[torch.Tensor, Dict[str, Any]]]): List of samples

    Returns:
        Dict[str, Any]: Batched data
    """
    images = torch.stack([item[0]for item in batch])
    metadata = [item[1] for item in batch]

    return {
        'images': images,
        'metadata': metadata
    }

def load_config(config_path: str) -> Config:

    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    config = Config(
        config_file=config_dict.get('config_file'),
        checkpoint_path=config_dict.get('checkpoint_path'),
        data_dir=config_dict.get('data_dir'),
        data_config_file= config_dict.get('data_config_file'),
        text_prompt=config_dict.get('text_prompt'),
        output_dir=config_dict.get('output_dir'),
        box_threshold=config_dict.get('box_threshold'),
        text_threshold=config_dict.get('text_threshold'),
        cpu_only=config_dict.get('cpu_only', False),
        batch_size=config_dict.get('batch_size', 4),
        num_workers=config_dict.get('num_workers', 2),
        nms_threshold=config_dict.get('nms_threshold', 0.5),
        world_size=config_dict.get('world_size', torch.cuda.device_count())
    )
    return config
def main_worker(rank: int, world_size: int, config: Config) -> None:
    """Main worker

    Args:
        rank (int): Rank of the current process
        world_size (int): Total number of processes
        config (Config): Configuration settings
    """ 
    logger, _ = setup_logging(config.output_dir, rank)
    logger.info(f"Rank {rank}: Starting extraction process")

       
    setup_distributed(
        rank,
        world_size,
        logger
    )
    torch.cuda.set_device(rank)
    device = torch.device(
        'cpu' if config.cpu_only else torch.device(f'cuda:{rank}')
    )
    output_dir = config.output_dir
    annotated_image_dir = os.path.join(
        output_dir,
        'annotated_images'
    )
    os.makedirs(annotated_image_dir, exist_ok=True)

    transformer = Compose(
        [
            RandomResize([1500], max_size=1500),
            ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    dataset = SceneDataset(data_dir=config.data_dir, cfg_file_path=config.data_config_file, transform=transformer)
    sampler = DistributedSampler(dataset=dataset, shuffle=False)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        sampler =sampler,
        num_workers= config.num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    
    grounding_dino = GroundingDinoClass(
        model_checkpoint_path= config.checkpoint_path,
        model_config_path= config.config_file,
        device= device.type
    )
    model = grounding_dino.model
    model.to(device)
    if not config.cpu_only:
        model = DDP(model, device_ids=[rank])
    
    all_annotations = []

    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Rank {rank} Processing batches")):
        images = batch['images'].to(device)
        metadata_list = batch['metadata']

        current_batch_size = images.size(0)

        batch_captions = [config.text_prompt] * current_batch_size
        boxes_list, scores_list, phrases_list = grounding_dino.predict(
            images,
            captions=batch_captions,
            box_threshold=config.box_threshold,
            text_threshold=config.text_threshold,
            device=device.type,
            nms_threshold=config.nms_threshold
        )
        for idx in range(current_batch_size):
            metadata = metadata_list[idx]
            image_path = metadata['image_path']
            image_name = os.path.basename(image_path)
            image_folder_belong = os.path.basename(os.path.dirname(image_path))
            image_name = f"{image_folder_belong}_{image_name}"
            original_image = Image.open(image_path).convert('RGB')


            boxes = boxes_list[idx]
            phrases = scores_list[idx]
            scores = phrases_list[idx]

            if len(boxes) > 0:
                annotated_image = plot_boxes_to_image(original_image.copy(), {
                  "size": original_image.size,
                  "boxes": boxes,
                  "labels": phrases,
                })
            else:
                annotated_image = original_image
            
            annotated_image_path = os.path.join(annotated_image_dir, image_name)
            annotated_image.save(annotated_image_path)

            annotation = metadata.copy()
            annotation["bounding_boxes"] = boxes.tolist()
            annotation["scores"] = scores.tolist()
            annotation["phrases"] = phrases
            annotation["annotated_image_path"] = annotated_image_path
            all_annotations.append(annotation)
    

    gathered_annotations = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_annotations, all_annotations)

    if rank == 0:
        combined_annotations = []
        for annotations in gathered_annotations:
            combined_annotations.extend(annotations)
        
        combined_annotations.sort(key=lambda x: x['id'])
        annotations_file = os.path.join(output_dir, "all_annotations.json")
        
        with open(annotations_file, 'w') as f:
            json.dump(combined_annotations, f, indent=4)
    cleanup(logger)

def main():
    parser = argparse.ArgumentParser(description="GroundingDINO Inference")
    parser.add_argument('--config', type=str, required=True, help="Path to the config.json file")
    args = parser.parse_args()

    config = load_config(args.config)
    world_size = config.world_size
    if world_size < 1:
        raise ValueError("No CUDA devices available for DDP.")
    torch.multiprocessing.spawn(
        main_worker,
        args=(world_size, config),
        nprocs=world_size,
        join=True
    )
if __name__ == "__main__":
    main()