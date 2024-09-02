import os
os.environ["HF_HOME"] = "weights" # 事前学習モデルの保存先を指定
from pathlib import Path
import shutil

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
import toml
from transformers import DetrConfig, DetrForObjectDetection

from modules.trainer import Trainer
from modules.custom_collate_fn import collate_fn
from modules.utils import fix_seeds, now_date_str
from modules.loader import make_pathlist_voc, DETRDataset


# 定数
CONFIG_PATH = "./config/train_config.toml"
    
def main():
    # 乱数の固定
    fix_seeds()
    
    # 設定ファイルの読み込み
    with open(CONFIG_PATH, mode="r", encoding="utf-8") as f:
        cfg = toml.load(f)
    
    ## 入出力パス
    input_path = Path(cfg["input_path"])
    output_path = Path(cfg["output_path"]).joinpath(now_date_str())
    output_path.mkdir(parents=True, exist_ok=True)
    
    ## 各種パラメータ
    num_epoches = cfg["parameters"]["num_epoches"]
    batch_size = cfg["parameters"]["batch_size"]
    classes = cfg["parameters"]["classes"]
    input_size = cfg["parameters"]["input_size"]
    lr = cfg["optimizer"]["lr"]
    weight_decay = cfg["optimizer"]["weight_decay"]
    
    #デバイスの設定
    gpu = cfg["gpu"]
    if torch.cuda.is_available() and (gpu >= 0):
        device = torch.device(f"cuda:{gpu}")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    else:
        device = torch.device("cpu")
    print(f"使用デバイス {device}")
    
    # データの読み込み
    train_img_list, train_anno_list, val_img_list, val_anno_list = make_pathlist_voc(input_path)
    print(f"データ分割 train:val = {len(train_img_list)}:{len(val_img_list)}")
    
    # Datasetの作成
    train_dataset = DETRDataset(train_img_list, train_anno_list, classes, input_size,
                                dataset_type="pascal_voc", phase="train")
    val_dataset = DETRDataset(val_img_list, val_anno_list, classes, input_size,
                              dataset_type="pascal_voc", phase="val")
    
    # DataLoaderの作成
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=0,
                                  pin_memory=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size, shuffle=False, num_workers=0,
                                pin_memory=True, collate_fn=collate_fn)
    
    # モデルの定義
    id2label = {str(i): class_name for i, class_name in enumerate(classes)}
    label2id = {class_name: i for i, class_name in enumerate(classes)}
    config = DetrConfig(id2label=id2label, label2id=label2id)
    model = DetrForObjectDetection(config)
    
    # optimizerの定義
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Trainerの定義
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        device=device,
        output_path=output_path
    )
    
    # configを保存
    shutil.copy2(CONFIG_PATH, output_path)
    model.config.to_json_file(output_path.joinpath("config.json"))
    
    # 学習ループを実行
    for i in range(num_epoches):
        epoch = i + 1
        
        # 訓練
        train_loss = trainer.train(epoch)
        # 検証
        val_loss = trainer.validation(epoch)
        
        # ログの標準出力
        print(f"Epoch:{epoch}  train_loss:{train_loss:.4f}  val_loss:{val_loss:.4f}")
        
        # 学習の進捗を出力
        trainer.output_learning_curve()
    
    # モデルとログの出力
    trainer.save_weight()
    trainer.output_log()


if __name__ == "__main__":
    main()