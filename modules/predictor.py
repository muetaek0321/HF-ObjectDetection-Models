import numpy as np
import torch
from albumentations.core.bbox_utils import denormalize_bboxes
from torch.nn.functional import softmax
from transformers.image_transforms import center_to_corners_format
from transformers.modeling_utils import PreTrainedModel

from .loader.dataset import get_transform


class Predictor:
    """推論を実行するクラス"""

    def __init__(
        self,
        model: PreTrainedModel,
        threshold: float,
        input_size: list[int],
        device: torch.device | str,
    ) -> None:
        """コンストラクタ"""
        self.model = model
        self.threshold = threshold
        self.input_size = input_size
        self.device = device

        # 学習の準備
        self.model.to(device)
        self.model.eval()

        # DataAugmentation
        self.transform = get_transform("coco", self.input_size, "test")

    def __call__(self, img: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """1画像で推論の処理を実行"""

        # 画像の前処理を適用
        pixel_values = self.transform(image=img)["image"]
        pixel_values = pixel_values.to(self.device)
        pixel_values = pixel_values.unsqueeze(0)

        with torch.no_grad():
            outputs = self.model(pixel_values=pixel_values)

        # 予測BBoxと予測ラベルを取得
        logits, pred_bboxes = outputs.logits, outputs.pred_boxes

        # 予測BBoxを変換
        pred_bboxes = center_to_corners_format(pred_bboxes)

        # モデルの種類に応じてスコアと予測ラベルを計算
        # DETR (初代): CrossEntropyLoss (最終次元が背景) -> Softmax
        # Deformable-DETR / RF-DETR: Focal Loss (全クラス独立) -> Sigmoid
        model_type = getattr(self.model.config, "model_type", "").lower()
        if model_type == "detr":
            prob = softmax(logits, -1)
            scores, pred_labels = prob[..., :-1].max(-1)
        else:
            prob = logits.sigmoid()
            scores, pred_labels = prob.max(-1)

        # torch.Tensor -> numpy.ndarray
        scores = scores.cpu().numpy()[0]
        pred_labels = pred_labels.cpu().numpy()[0]
        pred_bboxes = pred_bboxes.cpu().numpy()[0]

        # 閾値以上のスコアの予測のみを抽出
        vis_idx = np.where(scores > self.threshold)
        scores = scores[vis_idx]
        labels = pred_labels[vis_idx]
        bboxes = denormalize_bboxes(pred_bboxes[vis_idx], shape=img.shape)

        return bboxes, labels, scores
