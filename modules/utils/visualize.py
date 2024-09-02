from copy import deepcopy
from pathlib import Path

import numpy as np
import cv2

from modules.utils import imwrite_jpn


__all__ = ["visualize_bbox"]

# 定数
COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255), (255, 255, 0)
]

def visualize_bbox(
    img: np.ndarray,
    bboxes: np.ndarray | list[list[int]],
    labels: np.ndarray | list[int],
    output_path: str | Path
) -> None:
    """BBoxを可視化した画像を出力
    
    Args:
        img (np.ndarray): 画像データ
        bboxes (np.ndarray,list): BBoxが格納された配列
    """
    img_vis = deepcopy(img)
    
    # BBoxを可視化
    for bbox, label in zip(bboxes, labels):
        pt1 = (int(bbox[0]), int(bbox[1]))
        pt2 = (int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3]))
        img_vis = cv2.rectangle(img_vis, pt1, pt2, COLORS[label])
        
    # 可視化画像を出力
    imwrite_jpn(output_path, img_vis)
    
