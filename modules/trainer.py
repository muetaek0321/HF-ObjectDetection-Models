from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import supervision as sv
import torch
from schedulefree import RAdamScheduleFree
from supervision.metrics import MeanAveragePrecision
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.image_transforms import center_to_corners_format
from transformers.modeling_utils import PreTrainedModel

from modules.schema import ModelType
from modules.utils import get_logger

# エラー対処
matplotlib.use("Agg")

# ロガーの取得
logger = get_logger(__name__)


@dataclass(frozen=True)
class ValidationResult:
    """検証結果を格納するデータクラス。"""

    val_loss: float
    map50: float
    map75: float
    map50_95: float
    is_early_stopping: bool


class Trainer:
    """訓練を実行するクラス"""

    def __init__(
        self,
        model: PreTrainedModel,
        optimizer: Optimizer | RAdamScheduleFree,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        patience: int,
        device: torch.device | str,
        model_name: ModelType,
        output_path: str | Path,
    ) -> None:
        """初期化

        Args:
            model (PreTrainedModel): 訓練するモデル
            optimizer (Optimizer | RAdamScheduleFree): 最適化手法
            train_dataloader (DataLoader): 訓練データのDataLoader
            val_dataloader (DataLoader): 検証データのDataLoader
            patience (int): Early Stoppingの更新無しエポック数
            device (torch.device | str): 使用するデバイス
            model_name (ModelType): モデル名
            output_path (str | Path): 出力先のパス
        """
        self.model = model
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.patience = patience
        self.device = device
        self.model_name = model_name
        self.output_path = Path(output_path)

        # 学習の準備
        self.model.to(device)
        self.best_model = None
        self.best_epoch = 0
        self.best_score = 0.0

        # ログ保存の準備
        self.log = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "val_map50": [],
            "val_map75": [],
            "val_map50_95": [],
        }

    def train(self, epoch: int) -> float:
        """訓練のループを実行

        Args:
            epoch (int): 現在のエポック数

        Returns:
            float: 訓練の平均loss
        """
        self.model.train()
        self.optimizer.train()
        iter_train_loss = []

        for batch in tqdm(self.train_dataloader, desc="train"):
            pixel_values = batch["pixel_values"].to(self.device)
            labels = [
                {key: value.to(self.device) for key, value in targets.items()}
                for targets in batch["labels"]
            ]

            self.optimizer.zero_grad()

            output = self.model(pixel_values=pixel_values, labels=labels)

            output.loss.backward()
            self.optimizer.step()

            iter_train_loss.append(output.loss.item())

        # 1epochの平均lossを計算
        epoch_train_loss = np.mean(iter_train_loss)
        self.log["train_loss"].append(epoch_train_loss)
        self.log["epoch"].append(epoch)

        return epoch_train_loss

    def validation(self, epoch: int) -> ValidationResult:
        """検証のループを実行

        Args:
            epoch (int): 現在のエポック数

        Returns:
            ValidationResult: 検証の実行結果
        """
        self.model.eval()
        self.optimizer.eval()
        iter_val_loss = []
        map_metric = MeanAveragePrecision()
        is_early_stopping = False

        for batch in tqdm(self.val_dataloader, desc="val"):
            pixel_values = batch["pixel_values"].to(self.device)
            labels = [
                {key: value.to(self.device) for key, value in targets.items()}
                for targets in batch["labels"]
            ]

            with torch.no_grad():
                output = self.model(pixel_values=pixel_values, labels=labels)

            iter_val_loss.append(output.loss.item())
            self._update_map_metric(
                map_metric, output.logits, output.pred_boxes, pixel_values, labels
            )

        # 1epochの平均lossを計算
        epoch_val_loss = np.mean(iter_val_loss)
        self.log["val_loss"].append(epoch_val_loss)

        # mAP指標を計算
        map_result = map_metric.compute()
        self.log["val_map50"].append(map_result.map50)
        self.log["val_map75"].append(map_result.map75)
        self.log["val_map50_95"].append(map_result.map50_95)

        # 最良のScoreを判定
        if self.best_score < map_result.map50:
            self.best_model = deepcopy(self.model)
            self.best_epoch = epoch
            self.best_score = map_result.map50
        else:
            # EarlyStoppingの判定
            if epoch - self.best_epoch >= self.patience:
                is_early_stopping = True

        return ValidationResult(
            val_loss=epoch_val_loss,
            map50=map_result.map50,
            map75=map_result.map75,
            map50_95=map_result.map50_95,
            is_early_stopping=is_early_stopping,
        )

    def _update_map_metric(
        self,
        map_metric: MeanAveragePrecision,
        logits: torch.Tensor,
        pred_boxes: torch.Tensor,
        pixel_values: torch.Tensor,
        labels: list[dict[str, torch.Tensor]],
    ) -> None:
        """検証バッチごとの検出結果をmAP指標に追加

        Args:
            map_metric (MeanAveragePrecision): mAP指標のインスタンス
            logits (torch.Tensor): モデルの出力ロジット
            pred_boxes (torch.Tensor): モデルの出力予測ボックス
            pixel_values (torch.Tensor): 入力画像
            labels (list[dict[str, torch.Tensor]]): 正解ラベルのリスト
        """
        height, width = pixel_values.shape[-2:]
        scale = pred_boxes.new_tensor([width, height, width, height])
        pred_boxes = center_to_corners_format(pred_boxes) * scale

        if self.model_name == "detr":
            probabilities = logits.softmax(dim=-1)[..., :-1]
        else:
            probabilities = logits.sigmoid()
        scores, pred_class_ids = probabilities.max(dim=-1)

        predictions = [
            sv.Detections(
                xyxy=pred_boxes[index].detach().cpu().numpy(),
                confidence=scores[index].detach().cpu().numpy(),
                class_id=pred_class_ids[index].detach().cpu().numpy(),
            )
            for index in range(pixel_values.shape[0])
        ]
        targets = [
            sv.Detections(
                xyxy=(center_to_corners_format(target["boxes"]) * scale).detach().cpu().numpy(),
                class_id=target["class_labels"].detach().cpu().numpy(),
            )
            for target in labels
        ]

        # mAP指標を更新
        map_metric.update(predictions=predictions, targets=targets)

    def save_weight(self) -> None:
        """モデルの重みを保存"""
        # 最終epochのモデル
        epoch = self.log["epoch"][-1]
        model_name = f"{epoch}_latest.pth"
        torch.save(self.model.state_dict(), self.output_path.joinpath(model_name))
        logger.info(f"model saved: {model_name}")

        # 最良のepochのモデル
        best_model_name = f"{self.best_epoch}_best.pth"
        torch.save(self.best_model.state_dict(), self.output_path.joinpath(best_model_name))
        logger.info(f"best model saved: {best_model_name} (best score: {self.best_score:.4f})")

    def output_learning_curve(self) -> None:
        """学習曲線の出力"""
        epoch = len(self.log["epoch"])  # 現在までのエポック数を取得

        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle(f"Learning Curve (Epoch: {epoch})")
        ax[0].set_title("Loss")
        ax[0].plot(self.log["epoch"], self.log["train_loss"], c="red", label="train")
        ax[0].plot(self.log["epoch"], self.log["val_loss"], c="blue", label="val")
        ax[0].set_xlabel("Epoch")
        ax[0].set_ylabel("Loss")
        ax[0].legend()

        ax[1].set_title("mAP")
        ax[1].plot(self.log["epoch"], self.log["val_map50"], label="mAP@50")
        ax[1].plot(self.log["epoch"], self.log["val_map75"], label="mAP@75")
        ax[1].plot(self.log["epoch"], self.log["val_map50_95"], label="mAP@50:95")
        ax[1].set_xlabel("Epoch")
        ax[1].set_ylabel("mAP")
        ax[1].legend()

        plt.tight_layout()
        plt.savefig(self.output_path.joinpath("learning_curve.png"))

        plt.close()

    def output_log(self) -> None:
        """ログファイルの出力"""
        # DataFrameに変換してcsvで出力
        log_df = pd.DataFrame(self.log)
        log_df.to_csv(
            self.output_path.joinpath("training_log.csv"), encoding="utf-8-sig", index=False
        )
