from pathlib import Path

import torch
from torch.nn.functional import softmax 
from transformers import DetrForObjectDetection
from transformers.image_transforms import center_to_corners_format
from albumentations.core.bbox_utils import denormalize_bboxes
import numpy as np

from .loader.augmentation import get_transform
from .utils import imread_jpn, visualize_bbox


class Inference:
    """推論を実行するクラス"""
    
    def __init__(
        self,
        model: DetrForObjectDetection,
        threshold: float,
        input_size: list[int],
        device: torch.device | str,
        output_path: str | Path
    ) -> None:
        """コンストラクタ
        """
        self.model = model
        self.threshold = threshold
        self.input_size = input_size
        self.device = device
        self.output_path = Path(output_path)
        
        # 学習の準備
        self.model.to(device)
        self.model.eval()
        
        # DataAugmentation
        self.transform = get_transform("coco", self.input_size, "test")
        
    def __call__(
        self,
        img_path: Path | str
    ) -> float:
        """1画像で推論の処理を実行
        """
        # 画像読み込み
        img = imread_jpn(img_path)
        h, w = img.shape[:2]
        
        # 画像の前処理を適用
        pixel_values = self.transform(image=img)['image']
        pixel_values = pixel_values.to(self.device)
        pixel_values = pixel_values.unsqueeze(0)
        
        with torch.no_grad():
            outputs = self.model(pixel_values=pixel_values)
                
        # 予測BBoxと予測ラベルを取得
        logits, pred_bboxes = outputs.logits, outputs.pred_boxes
        
        # 予測BBoxを変換
        pred_bboxes = center_to_corners_format(pred_bboxes)
        
        # スコアと予測ラベルを計算
        prob = softmax(logits, -1)
        scores, pred_labels = prob[..., :-1].max(-1)
        
        # torch.Tensor -> numpy.ndarray
        scores = scores.cpu().numpy()[0]
        pred_labels = pred_labels.cpu().numpy()[0]
        pred_bboxes = pred_bboxes.cpu().numpy()[0]
        
        # 閾値以上のスコアの予測のみを抽出
        vis_idx = np.where(scores > self.threshold)
        labels = pred_labels[vis_idx]
        bboxes = denormalize_bboxes(pred_bboxes[vis_idx], rows=h, cols=w)
        
        visualize_bbox(img, bboxes, labels, "xyxy", self.output_path.joinpath(img_path.name))
    
