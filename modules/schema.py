"""コンフィグファイルのバリデーション用Pydanticスキーマ定義モジュール。

train_config.toml および inference_config.toml の構造を
Pydantic BaseModel で定義し、型安全な設定値の読み込みを実現する。
"""

from enum import StrEnum

from pydantic import BaseModel, Field, field_validator


class ModelType(StrEnum):
    """サポートされるモデル形式の列挙型。"""

    DETR = "DETR"
    DEFORMABLE_DETR = "Deformable-DETR"
    RF_DETR = "RF-DETR"
    RT_DETR = "RT-DETR"


class DatasetType(StrEnum):
    """サポートされるデータセット形式の列挙型。"""

    COCO = "coco"
    PASCAL_VOC = "pascal_voc"


# ==============================================================
# 訓練コンフィグ (train_config.toml)
# ==============================================================


class TrainParameters(BaseModel):
    """訓練パラメータのスキーマ。

    train_config.toml の [parameters] セクションに対応する。
    """

    num_epoches: int = Field(gt=0, description="エポック数")
    batch_size: int = Field(gt=0, description="バッチサイズ")
    classes: list[str] = Field(min_length=1, description="検出対象クラス名のリスト")
    input_size: list[int] = Field(
        min_length=2, max_length=2, description="入力画像サイズ [height, width]"
    )
    dataset_type: DatasetType = Field(description="データセット形式")

    @field_validator("input_size")
    @classmethod
    def _validate_input_size_positive(cls, v: list[int]) -> list[int]:
        """入力画像サイズの各要素が正の整数であることを検証する。"""
        if any(dim <= 0 for dim in v):
            msg = f"input_size の各要素は正の整数である必要があります: {v}"
            raise ValueError(msg)
        return v


class OptimizerConfig(BaseModel):
    """オプティマイザ設定のスキーマ。

    train_config.toml の [optimizer] セクションに対応する。
    """

    lr: float = Field(gt=0, description="学習率")
    lr_backbone: float = Field(gt=0, description="バックボーンの学習率")
    weight_decay: float = Field(ge=0, description="重み減衰")


class TrainConfig(BaseModel):
    """訓練コンフィグ全体のスキーマ。

    train_config.toml のトップレベル構造に対応する。
    """

    input_path: str = Field(description="入力データセットのパス")
    output_path: str = Field(description="出力先ディレクトリのパス")
    gpu: int = Field(ge=-1, description="使用するGPU番号 (-1でCPU使用)")
    use_pretrained: bool = Field(description="事前学習済みモデルを使用するか")
    model_name: ModelType = Field(description="使用するモデルの名前")
    parameters: TrainParameters = Field(description="訓練パラメータ")
    optimizer: OptimizerConfig = Field(description="オプティマイザ設定")


# ==============================================================
# 推論コンフィグ (inference_config.toml)
# ==============================================================


class InferenceParameter(BaseModel):
    """推論パラメータのスキーマ。

    inference_config.toml の [parameter] セクションに対応する。
    """

    threshold: float = Field(gt=0, le=1, description="検出閾値 (0〜1)")


class InferenceConfig(BaseModel):
    """推論コンフィグ全体のスキーマ。

    inference_config.toml のトップレベル構造に対応する。
    """

    train_result_path: str = Field(description="学習結果ディレクトリのパス")
    input_path: str = Field(description="推論対象の画像ディレクトリのパス")
    output_path: str = Field(description="推論結果の出力先パス (空文字で学習結果フォルダ内に出力)")
    gpu: int = Field(ge=-1, description="使用するGPU番号 (-1でCPU使用)")
    parameter: InferenceParameter = Field(description="推論パラメータ")
