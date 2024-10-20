import os
import json 
from PIL import Image, ImageDraw, ImageFont
import torch
from typing import Any, Dict, List, Optional
from GroundingDINO.groundingdino.util.box_ops import box_cxcywh_to_xyxy
import numpy as np

def plot_boxes_to_image(
    image_pil: Image.Image,
    tgt: Dict[str, Any]
) -> Image.Image:
    w, h = image_pil.size
    boxes = tgt['boxes']
    labels = tgt['labels']

    draw = ImageDraw.Draw(image_pil)

    for box, label in zip(boxes, labels):
        box = box.cpu()
        box = box * torch.tensor([w, h, w, h])
        x0, y0, x1, y1 = box.tolist() 
        x0, y0, x1, y1 = map(int, [x0, y0, x1, y1])
        color = tuple(np.random.randint(0, 255, size=3).tolist())

        draw.rectangle([x0, y0, x1, y1], outline=color, width=15)
        font = ImageFont.load_default()
        text_size = draw.textsize(label, font)
        draw.rectangle([x0, y0 - text_size[1], x0 + text_size[0], y0], fill=color)
        draw.text((x0, y0 - text_size[1]), label, fill='white', font=font)
    return image_pil

