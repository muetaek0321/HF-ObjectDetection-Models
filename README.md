# DETR Object Detection (DETR-template)

Hugging Face Transformers を活用した DETR（DEtection TRansformer）およびその派生モデル（Deformable-DETR, RF-DETR, RT-DETR）による物体検出の学習・推論パイプライン＆テンプレートリポジトリです。

---

## 🚀 特徴

- **最新・多様な DETR 派生モデルのサポート**:
  - **DETR** (`facebook/detr-resnet-50`)
  - **Deformable-DETR** (`SenseTime/deformable-detr`)
  - **RF-DETR** (`Roboflow/rf-detr-base`)
  - **RT-DETR** (`PekingU/rtdetr_v2_r50vd`)
- **Pydantic & TOML による型安全な設定管理**:
  - 学習設定（[`config/train_config.toml`](config/train_config.toml)）および推論設定（[`config/inference_config.toml`](config/inference_config.toml)）を Pydantic スキーマ（[`modules/schema.py`](modules/schema.py)）で厳格にバリデーション。
- **Pascal VOC 形式データセット対応**:
  - VOC XML アノテーションの読み込み、Train / Val / Test への自動分割、データセット一覧 CSV の自動保存。
- **最新の最適化・データ拡張パイプライン**:
  - スケジューラ設定不要で高速収束する **Schedule-Free Optimizer**（`RAdamScheduleFree`）の採用。
  - **Albumentations** を用いたバウンディングボックス追従のデータ拡張。
- **supervision による高精度な評価 & 可視化**:
  - `supervision.metrics.MeanAveragePrecision` による mAP50 / mAP75 / mAP50:95 の自動計算。
  - mAP50 に基づく **Early Stopping** 機能（`patience` パラメータ対応）。
  - `BoxAnnotator` / `LabelAnnotator` によるスマートなバウンディングボックス描画。
- **充実したログと可視化機能**:
  - 学習曲線（損失推移）の自動グラフ描画・保存（`learning_curve.png`）。
  - 日本語パスや文字描画に対応した画像ユーティリティ（`imread_jpn`, `imwrite_jpn`）。
  - 学習完了時のテスト推論結果の自動保存および推論時間計測（`ProcessTimeManager`）。
- **モダンな Python 開発環境**:
  - 高速パッケージマネージャー **uv** に完全対応（PyTorch CUDA 12.8 wheel 自動設定）。
  - **Ruff** によるコードフォーマット・静的解析の自動化。

---

## 🛠️ 技術スタック

- **Python**: `>= 3.13`
- **パッケージ管理**: [uv](https://docs.astral-sh/uv/)
- **ディープラーニング**:
  - [PyTorch](https://pytorch.org/) & [torchvision](https://pytorch.org/vision/)（CUDA 12.8 対応）
  - [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
  - [timm](https://github.com/huggingface/pytorch-image-models)
  - [schedulefree](https://github.com/facebookresearch/schedule_free) (`RAdamScheduleFree`)
- **物体検出評価 & 可視化**:
  - [supervision](https://github.com/roboflow/supervision)
  - [Matplotlib](https://matplotlib.org/)
- **画像処理 & 拡張**:
  - [Albumentations](https://albumentations.ai/)
  - [OpenCV](https://opencv.org/)
- **データ管理 & バリデーション**:
  - [Pydantic](https://docs.pydantic.dev/) (v2)
  - [Pandas](https://pandas.pydata.org/)
  - [toml](https://github.com/uiri/toml)
- **コード品質**:
  - [Ruff](https://github.com/astral-sh/ruff)
  - [pytest](https://docs.pytest.org/)

---

## 📋 前提条件

- Python 3.13 以上
- [uv](https://docs.astral-sh/uv/)
- NVIDIA GPU & CUDA 環境（GPU 推奨。CPU でも動作可能）

---

## 🏁 使い方 (Getting Started)

### 1. 環境構築

本プロジェクトは **uv** を使用してパッケージと仮想環境を一元管理します。`pyproject.toml` に PyTorch (CUDA 12.8) の専用ソースが設定されているため、以下のコマンドを実行するだけで仮想環境の作成と依存関係の同期が完了します。

```bash
# 仮想環境を作成し依存パッケージを自動同期
uv sync
```

---

### 2. モデルの学習

#### 2.1 設定ファイルの編集

[`config/train_config.toml`](config/train_config.toml) を編集して、学習パラメータやデータセットのパス、対象モデルを指定します。設定値は起動時に Pydantic スキーマによって型検証されます。

| パラメータ                | 説明                                                              | 設定例                   |
| :------------------------ | :---------------------------------------------------------------- | :----------------------- |
| `input_path`              | VOC 形式データセットのルートパス                                  | `"../_dataset/face_voc/"` |
| `output_path`             | 学習結果を保存するディレクトリ                                    | `"./results"`            |
| `gpu`                     | 使用する GPU 番号（`-1` で CPU）                                  | `0`                      |
| `use_pretrained`          | 事前学習済み重みを使用するか                                      | `true`                   |
| `model_name`              | 使用するモデル名（`DETR`, `Deformable-DETR`, `RF-DETR`, `RT-DETR`） | `"RF-DETR"`              |
| `parameters.num_epoches`  | 総学習エポック数                                                  | `1000`                   |
| `parameters.batch_size`   | バッチサイズ                                                      | `8`                      |
| `parameters.classes`      | 検出対象のクラス名リスト                                          | `["face"]`               |
| `parameters.input_size`   | 入力画像サイズ `[height, width]`                                  | `[512, 512]`             |
| `parameters.dataset_type` | データセット種別（`"pascal_voc"` または `"coco"`）                | `"pascal_voc"`           |
| `parameters.patience`     | Early Stopping の許容エポック数（mAP50 の更新停止判定）           | `50`                     |
| `optimizer.lr`            | 最適化アルゴリズムの学習率                                        | `1e-4`                   |
| `optimizer.lr_backbone`   | バックボーンの学習率                                              | `1e-5`                   |
| `optimizer.weight_decay`  | 重み減衰（Weight Decay）                                          | `1e-4`                   |

#### 2.2 学習スクリプトの実行

```bash
uv run python main_train.py
```

> [!NOTE]
> 学習が開始されると、`results/<モデル名>_<YYYY-MM-DD_HH-MM-SS>/` ディレクトリが生成され、以下の成果物が自動保存されます。
>
> - 最良・最終モデルの重み（`<best_epoch>_best.pth`, `<epoch>_latest.pth`）
> - 学習設定のバックアップ（`train_config.toml`, `config.json`）
> - 損失および mAP 推移ログ CSV & 学習曲線グラフ（`log.csv`, `learning_curve.png`）
> - データ分割ログ（`input_data/train.csv`, `val.csv`, `test.csv`）
> - テストデータに対するサンプル推論結果画像（`test/`）

---

### 3. 推論

#### 3.1 設定ファイルの編集

[`config/inference_config.toml`](config/inference_config.toml) を編集して、学習済みモデルへのパスと入出力ディレクトリを指定します。

| パラメータ            | 説明                                                                             | 設定例                                            |
| :-------------------- | :------------------------------------------------------------------------------- | :------------------------------------------------ |
| `train_result_path`   | 保存された学習結果ディレクトリへのパス                                           | `"./results/RF-DETR_2026-10-05_23-16-00"`         |
| `input_path`          | 推論対象の画像ディレクトリへのパス                                               | `"../_dataset/face_voc/JPEGImages/"`              |
| `output_path`         | 推論結果の保存ディレクトリ（空文字 `""` の場合は `train_result_path/inference`） | `""`                                              |
| `gpu`                 | 使用する GPU 番号（`-1` で CPU）                                                 | `0`                                               |
| `parameter.threshold` | バウンディングボックス検出の信頼度しきい値                                       | `0.5`                                             |

#### 3.2 一括推論の実行

対象ディレクトリ内の全画像に対して一括推論を行い、バウンディングボックスとクラス名、スコアを描画した画像を保存します。

```bash
uv run python main_predict.py
```

#### 3.3 推論テスト & グリッド表示

最初の9枚に対して推論速度を計測（ミリ秒単位でログ表示）し、3×3 のグリッド画像としてウィンドウ表示・確認します。

```bash
uv run python main_predict_test.py
```

---

## 💻 開発用コマンド

### パッケージの追加・削除

```bash
# 通常の依存関係を追加
uv add <package_name>

# 開発用依存関係を追加
uv add --dev <package_name>

# パッケージの削除
uv remove <package_name>
```

### リンター & フォーマッター (Ruff)

```bash
# コードの自動フォーマット
uv run ruff format .

# 静的解析チェック
uv run ruff check .

# 自動修正可能なエラーを修正
uv run ruff check --fix .
```

### テスト実行 (pytest)

```bash
uv run pytest
```

---

## 📂 ディレクトリ構成

```text
.
├── .agents/                 # AI エージェント支援用設定・スキル
├── .vscode/                 # VS Code ワークスペース推奨設定 (Ruff 連携等)
├── config/
│   ├── inference_config.toml # 推論用設定ファイル
│   └── train_config.toml     # 学習用設定ファイル
├── modules/
│   ├── loader/              # データ読み込み・前処理
│   │   ├── dataset.py       # VOC XML / COCO データセットローダー & データ拡張
│   │   └── load_data.py     # データセット分割ユーティリティ (Train / Val / Test)
│   ├── utils/               # 汎用ユーティリティ
│   │   ├── cv2_japanese.py  # 日本語パス対応画像入出力 (imread_jpn, imwrite_jpn)
│   │   ├── date_str.py      # 日時文字列生成
│   │   ├── fix_seed.py      # 乱数シード固定
│   │   ├── logger.py        # 統一ロガー設定
│   │   ├── process_time.py  # 処理時間計測コンテキストマネージャ
│   │   └── visualize.py     # supervision による BBox / ラベル可視化
│   ├── custom_collate_fn.py # DataLoader 用カスタムバッチ処理
│   ├── models.py            # 各種 DETR モデル定義 (DETR, Deformable-DETR, RF-DETR, RT-DETR)
│   ├── predictor.py         # 単一画像推論エンジン
│   ├── schema.py            # Pydantic による設定ファイルバリデーション
│   └── trainer.py           # 学習・検証ループ、Early Stopping & mAP 評価
├── AGENTS.md                # 開発規約・コーディングガイドライン
├── LICENSE                  # Apache-2.0 ライセンス
├── main_predict.py          # 画像一括推論エントリポイント
├── main_predict_test.py     # サンプル画像推論 & グリッド表示テスト
├── main_train.py            # モデル学習エントリポイント
├── pyproject.toml           # プロジェクト定義 & ツール設定 (Ruff, pytest, uv sources)
├── README.md                # 本ドキュメント
└── uv.lock                  # 依存パッケージロックファイル
```

---

## 👤 Author

- **プロジェクト作成者**: muetaek0321
- **README 作成**: Gemini 3.8 Flash

---

## 📄 ライセンス

本プロジェクトは [Apache License 2.0](LICENSE) の下で公開されています。
