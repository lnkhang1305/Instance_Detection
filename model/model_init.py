"""
Initialization for model initialization, including
- Zero-shot Feature extraction: CLIP, DinoV2
- Zero-shot GroundingDino
- Zero-shot segmentation: SAM2
"""
import torch
from PIL import Image
from typing import List, Tuple,Dict,Any,Optional,Union
from abc import ABC, abstractmethod
import torchvision
import sys
sys.path.append('.')
import numpy as np
import open_clip
from transformers import AutoImageProcessor, AutoModel


from ..GroundingDINO.groundingdino.models import build_model
from ..GroundingDINO.groundingdino.util.box_ops import box_cxcywh_to_xyxy
from ..GroundingDINO.groundingdino.util.slconfig import SLConfig
from ..GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from ..sam2_repo.sam2.build_sam import build_sam2
from ..sam2_repo.sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

NMS_THRESHOLD=0.5


class FeatExtractInterace(ABC):
    @abstractmethod
    def extract_features(self, images: List[Union[torch.Tensor, Image.Image]]) -> torch.Tensor:
        """
        Extract features from a batch of image
        Args:
          images (List[Union[torch.Tensor, Image.Image]]): A list of image tensors or PIL Images.

          Returns:
              torch.Tensor: A batch of feature vectors, shape (N, D) where N is the number of images
                            and D is the dimension of the feature vector.
        """
        pass

class DinoV2Model(FeatExtractInterace):
    """Initialize the DinoV2 model"""
    def __init__(self, model_name:str = 'facebook/dinov2-large'):
        """
        Initialize the DinoV2 model.

        Args:
            model_name (str): The name of the DinoV2 model to use.
        """
        self.model = AutoModel.from_pretrained(model_name)
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model.eval()
    
    def extract_features(self, images: List[Union[torch.Tensor, Image.Image]]) -> torch.Tensor:
        inputs = self.procesor(images = images, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :] ## getting the CLS token only

class CLIPModel(FeatExtractInterace):
    """Initialize the CLIP model"""
    def __init__(self, model_name:str) -> None:
        """
        Initialize the CLIP model

        Args:
            model_name (str): Model name
        """
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name
        )
        self.model.eval()
    
    def extract_features(self, images: List[Union[torch.Tensor, Image.Image]]) -> torch.Tensor:
        """
        Extract features using the DinoV2 model.

        Args:
            images (List[Union[torch.Tensor, Image.Image]]): A list of image tensors or PIL Images.

        Returns:
            torch.Tensor: A batch of feature vectors, shape (N, D).
        """
        if isinstance(images[0], Image.Image):
            images = [self.preprocess(image) for image in images]
        
        with torch.no_grad():
            features = self.model.encode_image(torch.stack(images))
        return features


class GroundingDino:
    """Initialization of GroundingDino"""
    def __init__(
        self,
        model_checkpoint_path:str,
        model_config_path = '../GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py',
        device:str = 'cuda'
    ):
        self.model = self._load_model_from_config(
            config_file = model_config_path,
            checkpoint_path = model_checkpoint_path,
            cpu_only=False,
            device=device
        )
        
    
    def _load_model_from_config(
        self,
        config_file:str,
        checkpoint_path: str,
        cpu_only:bool =False,
        device: Optional[str] = None
    ) -> torch.nn.Module:
        """
        Load the GroundingDINO model from a configuration file and checkpoint.

        Args:
            config_file (str): Path to the model configuration file.
            checkpoint_path (str): Path to the model checkpoint.
            cpu_only (bool, optional): If True, load the model on CPU. Defaults to False.
            device (Optional[str], optional): Device to load the model on. Defaults to None.

        Returns:
            torch.nn.Module: Loaded model.
        """
        print(f"Loading model from config: {config_file} and checkpoint: {checkpoint_path}")
        args = SLConfig.fromfile(config_file)
        args.device = "cpu" if cpu_only else ("cuda" if torch.cuda.is_available() else "cpu")
        model = build_model(args)
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
        model.eval()
        print("Model loaded and set to evaluation mode.")
        return model

    def predict(
        self,
        images: torch.Tensor,
        captions: List[str],
        box_threshold: float,
        text_threshold: float,
        device: str = "cuda"
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[List[str]]]:
        """
        Get predictions from the model for a batch of images.

        Args:
            model (torch.nn.Module): The GroundingDINO model.
            images (torch.Tensor): Batch of image tensors.
            captions (List[str]): List of text prompts.
            box_threshold (float): Threshold for box scores.
            text_threshold (float): Threshold for text scores.
            device (str, optional): Device to run the model on. Defaults to "cuda".

        Returns:
            Tuple[List[torch.Tensor], List[torch.Tensor], List[List[str]]]: List of boxes, scores, and phrases for each image.
        """
        captions = [cap.lower().strip() + "." if not cap.endswith(".") else cap.lower().strip() for cap in captions]

        self.model = self.model.to(device)
        images = images.to(device)

        with torch.no_grad():
            outputs = self.model(images, captions=captions)

        logits = outputs["pred_logits"].cpu().sigmoid()  # (bs, nq, 256)
        boxes = outputs["pred_boxes"].cpu()  # (bs, nq, 4)

        all_boxes = []
        all_scores = []
        all_phrases = []

        for i in range(images.size(0)):
            logits_i = logits[i]  # (nq, 256)
            boxes_i = boxes[i]  # (nq, 4)
            mask = logits_i.max(dim=1)[0] > box_threshold
            logits_i = logits_i[mask]
            boxes_i = boxes_i[mask]
            if logits_i.shape[0] == 0:
                all_boxes.append(torch.empty((0, 4)))
                all_scores.append(torch.empty((0,)))
                all_phrases.append([])
                continue

            tokenizer = self.model.tokenizer
            tokenized = tokenizer(captions[i])
            phrases_i = [
                get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer).replace('.', '')
                for logit in logits_i
            ]
            scores_i = logits_i.max(dim=1)[0]
            boxes_xyxy = box_cxcywh_to_xyxy(boxes_i)
            
            boxes_i, scores_i, phrases_i = self.apply_nms(boxes_xyxy, scores_i, phrases_i, NMS_THRESHOLD)
            
            all_boxes.append(boxes_i)
            all_scores.append(scores_i)
            all_phrases.append(phrases_i)

        return all_boxes, all_scores, all_phrases

    @staticmethod
    def apply_nms(
        boxes: torch.Tensor,
        scores: torch.Tensor,
        phrases: List[str],
        iou_threshold:float=0.5    
    ) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """
        Apply Non-Maximum Suppression to boxes and scores.
        
        Args:
            boxes (torch.Tensor): Bounding boxes in (x1, y1, x2, y2) format
            scores (torch.Tensor): Confidence scores for each box
            phrases (List[str]): Corresponding phrases for each box
            iou_threshold (float): IoU threshold for NMS
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor, List[str]]: Filtered boxes, scores, and phrases
        """
        print("Len before nms: ", boxes.size(0))
        
        keep_indices = torchvision.ops.nms(boxes, scores, iou_threshold)
        print("Len after: ", len(keep_indices))
        return boxes[keep_indices], scores[keep_indices], [phrases[i] for i in keep_indices]

class SegmentModel:
    def __init__(
        self,
        checkpoint:str,
        model_cfg: str
    ):
        """Initialize the segment model: SAM2

        Args:
            checkpoint (str): The checkpoint path
            model_cfg (str): The model config
        """
        self.model = build_sam2(
            model_cfg,
            checkpoint,
            torch.device('cuda'),
            apply_postprocessing=False
        )
        self.model.to(torch.device('cuda'))
        self.mask_generator = SAM2AutomaticMaskGenerator(self.model)

    @torch.no_grad()
    def predict(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Generate masks for input image

        Args:
            image (np.ndarray): The input image as a numpy array
        """

        """Mask output format
        
        segmentation : the mask
        area : the area of the mask in pixels
        bbox : the boundary box of the mask in XYWH format
        predicted_iou : the model's own prediction for the quality of the mask
        point_coords : the sampled input point that generated this mask
        stability_score : an additional measure of mask quality
        crop_box : the crop of the image used to generate this mask in XYWH format
        """
        masks = self.mask_generator.generate(image)
        return masks




