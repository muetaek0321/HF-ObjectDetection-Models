from pathlib import Path
from copy import deepcopy

import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from transformers import DetrForObjectDetection
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from schedulefree import RAdamScheduleFree


# エラー対処
matplotlib.use('Agg') 


class Trainer:
    """訓練を実行するクラス"""
    
    def __init__(
        self,
        model: DetrForObjectDetection,
        optimizer: Optimizer | RAdamScheduleFree,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        device: torch.device | str,
        output_path: str | Path
    ) -> None:
        """コンストラクタ
        """
        self.model = model
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.output_path = Path(output_path)
        
        # 学習の準備
        self.model.to(device)
        self.best_model = None
        self.best_epoch = 0
        self.best_loss = np.inf
        
        # ログ保存の準備
        self.log = {"epoch": [], "train_loss": [], "val_loss": []}
        
    def train(
        self,
        epoch: int
    ) -> float:
        """訓練のループを実行
        """
        self.model.train()
        self.optimizer.train()
        iter_train_loss = []
        
        for batch in tqdm(self.train_dataloader, desc="train"):
            pixel_values = batch["pixel_values"].to(self.device)
            labels = [{key: value.to(self.device) for key, value in targets.items()} 
                      for targets in batch["labels"]]
            
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
    
    def validation(
        self,
        epoch: int
    ) -> float:
        """検証のループを実行
        """
        self.model.eval()
        self.optimizer.eval()
        iter_val_loss = []
        
        for batch in tqdm(self.val_dataloader, desc="val"):
            pixel_values = batch["pixel_values"].to(self.device)
            labels = [{key: value.to(self.device) for key, value in targets.items()} 
                      for targets in batch["labels"]]

            with torch.no_grad():
                output = self.model(pixel_values=pixel_values, labels=labels)
            
            iter_val_loss.append(output.loss.item())
            
        # 1epochの平均lossを計算
        epoch_val_loss = np.mean(iter_val_loss)
        self.log["val_loss"].append(epoch_val_loss)
        
        # 最良のLossを判定
        if self.best_loss > epoch_val_loss:
            self.best_model = deepcopy(self.model)
            self.best_epoch = epoch
            self.best_loss = epoch_val_loss
            
        return epoch_val_loss
        
    def save_weight(
        self
    ) -> None:
        """モデルの重みを保存
        """        
        # 最終epochのモデル
        epoch = self.log["epoch"][-1]
        model_name = f"{epoch}_latest.pth"
        torch.save(self.model.state_dict(), self.output_path.joinpath(model_name))
        print(f"model saved: {model_name}")
        
        # 最良のepochのモデル
        best_model_name = f"{self.best_epoch}_best.pth"
        torch.save(self.best_model.state_dict(), self.output_path.joinpath(best_model_name))
        print(f"best model saved: {best_model_name} (best loss: {self.best_loss})")
        
    def output_learning_curve(
        self,
    ) -> None:
        """学習曲線の出力
        """
        epoch = len(self.log["epoch"]) # 現在までのエポック数を取得
        
        fig = plt.figure()
        ax = fig.add_subplot(title=f"Loss (Epoch:{epoch})")
        ax.plot(self.log["epoch"], self.log["train_loss"], c='red', label='train')
        ax.plot(self.log["epoch"], self.log["val_loss"], c='blue', label='val')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_path.joinpath("loss_curve.png"))
        
        plt.close()
        
    def output_log(
        self,
        
    ) -> None:
        """ログファイルの出力
        """
        # DataFrameに変換してcsvで出力
        log_df = pd.DataFrame(self.log)
        log_df.to_csv(self.output_path.joinpath("training_log.csv"),
                      encoding='utf-8-sig', index=False)
    
