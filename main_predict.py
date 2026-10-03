import os
from pathlib import Path

import toml
import torch
from tqdm import tqdm

from modules.models import get_model_inference
from modules.predictor import Predictor
from modules.schema import InferenceConfig, TrainConfig
from modules.utils import fix_seeds, imread_jpn
from modules.utils.visualize import visualize_bbox

# 定数
CONFIG_PATH = "./config/inference_config.toml"


def main():
    # 乱数の固定
    fix_seeds()

    # 設定ファイルの読み込み
    with open(CONFIG_PATH, mode="r", encoding="utf-8") as f:
        cfg = InferenceConfig.model_validate(toml.load(f))

    ## 入出力パス
    result_path = Path(cfg.train_result_path)
    input_path = Path(cfg.input_path)
    if cfg.output_path == "":
        # 未入力なら学習結果のフォルダに推論結果フォルダを作成
        output_path = result_path.joinpath("inference")
    else:
        output_path = cfg.output_path
    output_path.mkdir(parents=True, exist_ok=True)

    # デバイスの設定
    gpu = cfg.gpu
    if torch.cuda.is_available() and (gpu >= 0):
        device = torch.device(f"cuda:{gpu}")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    else:
        device = torch.device("cpu")
    print(f"使用デバイス {device}")

    # 推論用のパラメータを取得
    threshold = cfg.parameter.threshold

    # 学習時のconfigから必要なパラメータを取得
    with open(result_path.joinpath("train_config.toml"), mode="r", encoding="utf-8") as f:
        cfg_t = TrainConfig.model_validate(toml.load(f))
    model_name = cfg_t.model_name
    input_size = cfg_t.parameters.input_size
    classes = cfg_t.parameters.classes

    # 推論する画像データのパスリストを作成
    img_path_list = input_path.glob("*")

    # 自作モデルを使用
    model = get_model_inference(model_name, result_path, device)

    # 推論クラスの定義
    infer = Predictor(model=model, threshold=threshold, input_size=input_size, device=device)

    # 画像を1枚ずつ推論
    for img_path in tqdm(list(img_path_list), desc="inference"):
        # 画像読み込み
        img = imread_jpn(img_path)

        # 推論を実行
        bboxes, labels, scores = infer(img)

        # 推論結果を可視化して保存
        visualize_bbox(
            img, bboxes, labels, scores, classes, output_path=output_path / Path(img_path).name
        )


if __name__ == "__main__":
    main()
