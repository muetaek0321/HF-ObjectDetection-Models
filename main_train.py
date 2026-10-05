import os

# 事前学習モデルの保存先を指定
os.environ["HF_HOME"] = "pretrained"

import shutil
from pathlib import Path

import toml
import torch
from schedulefree import RAdamScheduleFree
from torch.utils.data import DataLoader
from tqdm import tqdm

from modules.custom_collate_fn import collate_fn
from modules.loader import DETRDataset, make_pathlist_voc
from modules.models import get_model_train
from modules.predictor import Predictor
from modules.schema import TrainConfig
from modules.trainer import Trainer
from modules.utils import ProcessTimeManager, fix_seeds, imread_jpn, now_date_str
from modules.utils.visualize import visualize_bbox

# 定数
CONFIG_PATH = "./config/train_config.toml"


def main():
    # 乱数の固定
    fix_seeds()

    # 設定ファイルの読み込み
    with open(CONFIG_PATH, mode="r", encoding="utf-8") as f:
        cfg = TrainConfig.model_validate(toml.load(f))

    ## モデル名称
    model_name = cfg.model_name
    use_pretrained = cfg.use_pretrained

    ## 入出力パス
    input_path = Path(cfg.input_path)
    output_path = Path(cfg.output_path) / f"{model_name}_{now_date_str()}"
    output_path.mkdir(parents=True, exist_ok=True)

    ## 各種パラメータ
    num_epoches = cfg.parameters.num_epoches
    batch_size = cfg.parameters.batch_size
    classes = cfg.parameters.classes
    input_size = cfg.parameters.input_size
    dataset_type = cfg.parameters.dataset_type
    patience = cfg.parameters.patience
    lr = cfg.optimizer.lr
    lr_backbone = cfg.optimizer.lr_backbone

    # デバイスの設定
    gpu = cfg.gpu
    if torch.cuda.is_available() and (gpu >= 0):
        device = torch.device(f"cuda:{gpu}")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    else:
        device = torch.device("cpu")
    print(f"使用デバイス {device}")

    # データのパスリストを作成
    train_df, val_df, test_df = make_pathlist_voc(input_path, is_split=True, test_data_ratio=0.01)
    print(f"データ分割 train:val = {len(train_df)}:{len(val_df)}")

    # Datasetの作成
    train_dataset = DETRDataset(
        train_df["image"],
        train_df["annotation"],
        classes,
        input_size,
        dataset_type=dataset_type,
        phase="train",
    )
    val_dataset = DETRDataset(
        val_df["image"],
        val_df["annotation"],
        classes,
        input_size,
        dataset_type=dataset_type,
        phase="val",
    )

    # DataLoaderの作成
    train_dataloader = DataLoader(
        train_dataset,
        batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # モデルの定義
    model, params = get_model_train(model_name, classes, lr_backbone, use_pretrained)

    # optimizerの定義
    optimizer = RAdamScheduleFree(params, lr=lr)

    # Trainerの定義
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        patience=patience,
        device=device,
        model_name=model_name,
        output_path=output_path,
    )

    # configを保存
    shutil.copy2(CONFIG_PATH, output_path)
    model.config.to_json_file(json_file_path=output_path / "config.json")

    # 学習ループを実行
    for epoch in range(1, num_epoches + 1):
        # 訓練
        train_loss = trainer.train(epoch)
        # 検証
        val_result = trainer.validation(epoch)

        # ログの標準出力
        print(
            f"Epoch:{epoch}\n"
            f"  train_loss:{train_loss:.4f}  val_loss:{val_result.val_loss:.4f}\n"
            f"  val_map50:{val_result.map50:.4f}  val_map75:{val_result.map75:.4f}  val_map50_95:{val_result.map50_95:.4f}"
        )

        # 学習の進捗を出力
        trainer.output_learning_curve()

        if val_result.is_early_stopping:
            print("EarlyStoppingで学習を終了します。")
            break

    # モデルとログの出力
    trainer.save_weight()
    trainer.output_log()

    # 入力データの一覧をファイル出力
    data_log_path = output_path / "input_data"
    data_log_path.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(data_log_path / "train.csv", encoding="utf-8-sig", index=False)
    val_df.to_csv(data_log_path / "val.csv", encoding="utf-8-sig", index=False)
    test_df.to_csv(data_log_path / "test.csv", encoding="utf-8-sig", index=False)

    # テスト結果の保存フォルダを作成
    test_output_path = output_path / "test"
    test_output_path.mkdir(parents=True, exist_ok=True)

    # 推論クラスの定義
    infer = Predictor(model=model, threshold=0.5, input_size=input_size, device=device)

    # 画像を1枚ずつ推論
    for img_path in tqdm(test_df["image"].tolist(), desc="inference"):
        # 画像読み込み
        img = imread_jpn(img_path)

        # 推論を実行
        bboxes, labels, scores = infer(img)

        # 推論結果を可視化して保存
        visualize_bbox(
            img, bboxes, labels, scores, classes, output_path=test_output_path / Path(img_path).name
        )


if __name__ == "__main__":
    with ProcessTimeManager(is_print=True) as pt:
        main()
