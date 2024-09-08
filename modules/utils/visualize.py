from copy import deepcopy
from pathlib import Path

import numpy as np
import cv2

from modules.utils import imwrite_jpn


__all__ = ["visualize_bbox"]

# 定数
COLORS = [
    (163, 8, 154), (31, 241, 13), (147, 39, 121), (246, 231, 135), (100, 87, 247), 
    (139, 122, 15), (81, 185, 168), (96, 11, 110), (49, 118, 245), (14, 118, 68), 
    (144, 168, 104), (58, 38, 203), (197, 203, 5), (137, 48, 203), (106, 140, 32), 
    (27, 240, 225), (63, 165, 168), (108, 26, 117), (29, 146, 246), (244, 186, 126), 
    (2, 233, 72), (85, 150, 219), (229, 24, 54), (254, 139, 135), (226, 138, 195)
]

def visualize_bbox(
    img: np.ndarray,
    bboxes: np.ndarray | list[list[int]],
    labels: np.ndarray | list[int],
    dataset_type: str = "coco",
    output_path: str | Path = "image_vis.png",
    save: bool= True
) -> np.ndarray:
    """BBoxを可視化した画像を出力
    
    Args:
        img (np.ndarray): 画像データ
        bboxes (np.ndarray,list): BBoxが格納された配列
    """
    img_vis = deepcopy(img)
    
    # BBoxを可視化
    for bbox, label in zip(bboxes, labels):
        if dataset_type == "coco":
            pt1 = (int(bbox[0]), int(bbox[1]))
            pt2 = (int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3]))
        elif dataset_type == "voc":
            pt1 = (int(bbox[0]), int(bbox[1]))
            pt2 = (int(bbox[2]), int(bbox[3]))
        img_vis = cv2.rectangle(img_vis, pt1, pt2, COLORS[label])
        
    # 可視化画像を出力
    if save:
        imwrite_jpn(output_path, img_vis)
        
    return img_vis
    
