import os
import json 
from PIL import Image, ImageDraw, ImageFont
import torch
from typing import Any, Dict, List, Optional
from GroundingDINO.groundingdino.util.box_ops import box_cxcywh_to_xyxy
import numpy as np

def plot_boxes_to_image(image_pil: Image.Image, boxes: torch.Tensor, phrases: List[str]) -> Image.Image:
    """
    Drawing bounding boxes and phrases on the image
    Args:
        image_pil: The original PIL image, before resizing and put into the model
        boxes(torch.Tensor): boxes under tensor type, in xyxy format and normalized between (0,1)
        phrases(List[str]): List of phrases
    Return:
        annotated image
    """
    print(type(image_pil))
    draw = ImageDraw.Draw(image_pil)
    W, H = image_pil.size
    boxes = boxes * torch.tensor([W, H, W, H])
    
    try:
        font = ImageFont.truetype("/kaggle/input/fontttt/Arial.ttf", size=20)
    except IOError:
        font = ImageFont.load_default()
    for box, phrase in zip(boxes, phrases):
        box = box.cpu().detach().numpy()
        x0, y0, x1, y1 = box.tolist()
        x0, y0, x1, y1 = map(int, [x0,y0,x1,y1])

        color = tuple(np.random.randint(0, 255, size=3).tolist())
        draw.rectangle([x0, y0, x1, y1], outline=color, width=5)
        
        text = phrase
        _, _, text_width, text_height = draw.textbbox((0, 0), text=text, font=font)
        draw.rectangle([x0, y0 - text_height, x0 + text_width, y0], fill=color)
        draw.text((x0, y0 - text_height), text, fill="white", font=font)
    return image_pil

