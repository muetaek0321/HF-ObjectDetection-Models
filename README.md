# DETR Object Detection (DETR-template)

Hugging Face Transformers を活用した DETR（DEtection TRansformer）およびその派生モデルによる物体検出の学習・推論パイプライン＆テンプレートリポジトリです。

---

## 🚀 特徴

- **多様な DETR 派生モデルのサポート**:
  - **DETR** (`facebook/detr-resnet-50`)
  - **Deformable-DETR** (`SenseTime/deformable-detr`)
  - **DETA** (`jozhang97/deta-resnet-50`)
  - **ConditionalDETR** (`microsoft/conditional-detr-resnet-50`)
- **TOML による直感的な設定管理**:
  - 学習設定（[`config/train_config.toml`](file:///c:/Users/masaki/Desktop/Python/DETR/config/train_config.toml)）および推論設定（[`config/inference_config.toml`](file:///c:/Users/masaki/Desktop/Python/DETR/config/inference_config.toml)）で、ハイパーパラメータやパスを一元管理。
- **Pascal VOC 形式データセット対応**:
  - VOC XML アノテーションの読み込み、Train / Val / Test への自動分割、データセット一覧 CSV の自動記録。
- **最新の最適化・データ拡張パイプライン**:
  - スケジューラ設定不要で高速収束する **Schedule-Free Optimizer**（`RAdamScheduleFree`）の採用。
  - **Albumentations** を用いたバウンディングボックス追従のデータ拡張。
- **充実したログと可視化機能**:
  - 学習曲線（損失推移）の自動グラフ描画・保存。
  - 日本語パスや文字描画に対応した画像ユーティリティ（`imread_jpn`, `visualize_bbox`）。
  - 学習完了時のテスト推論結果の自動保存。
- **モダンな Python 開発環境**:
  - 高速パッケージマネージャー **uv** に対応。
  - **Ruff** によるコードフォーマット・静的解析の自動化。

---

## 🛠️ 技術スタック

- **Python**: `>= 3.13`
- **パッケージ管理**: [uv](https://docs.astral-sh/uv/)（推奨） / pip
- **ディープラーニング**:
  - [PyTorch](https://pytorch.org/) & [torchvision](https://pytorch.org/vision/)
  - [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
  - [timm](https://github.com/huggingface/pytorch-image-models)
  - [schedulefree](https://github.com/facebookresearch/schedule_free)
- **画像処理 & 拡張**: [Albumentations](https://albumentations.ai/), [OpenCV](https://opencv.org/), [Pillow](https://python-pillow.org/)
- **可視化 & データ操作**: [Matplotlib](https://matplotlib.org/), [Pandas](https://pandas.pydata.org/), [tqdm](https://github.com/tqdm/tqdm)
- **設定管理**: [tomli](https://github.com/hukkin/tomli) / [toml](https://github.com/uiri/toml)
- **コード品質**: [Ruff](https://github.com/astral-sh/ruff), [pytest](https://docs.pytest.org/)

---

## 📋 前提条件

- Python 3.13 以上
- [uv](https://docs.astral-sh/uv/)（推奨）
- NVIDIA GPU & CUDA 環境（GPU 推奨。CPU でも動作可能）

---

## 🏁 使い方 (Getting Started)

### 1. 環境構築

#### uv を使用する場合（推奨）

```bash
# 仮想環境を作成して依存パッケージをインストール
uv venv
uv pip install -r requirements.txt
```

#### pip を使用する場合

```bash
python -m venv .venv
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

---

### 2. モデルの学習

#### 2.1 設定ファイルの編集

[`config/train_config.toml`](file:///c:/Users/masaki/Desktop/Python/DETR/config/train_config.toml) を編集して、学習パラメータやデータセットのパス、対象モデルを指定します。

| パラメータ                | 説明                                                                     | 設定例                   |
| :------------------------ | :----------------------------------------------------------------------- | :----------------------- |
| `input_path`              | VOC 形式データセットのルートパス                                         | `"../_dataset/VOC2012/"` |
| `output_path`             | 学習結果を保存するディレクトリ                                           | `"./results"`            |
| `gpu`                     | 使用する GPU 番号（`-1` で CPU）                                         | `0`                      |
| `use_pretrained`          | 事前学習済み重みを使用するか                                             | `true`                   |
| `model_name`              | 使用するモデル名（`DETR`, `Deformable-DETR`, `DETA`, `ConditionalDETR`） | `"DETR"`                 |
| `parameters.num_epoches`  | 総学習エポック数                                                         | `100`                    |
| `parameters.batch_size`   | バッチサイズ                                                             | `32`                     |
| `parameters.classes`      | 検出対象のクラス名リスト                                                 | `["person", "cat", ...]` |
| `parameters.input_size`   | 入力画像サイズ `[height, width]`                                         | `[512, 512]`             |
| `parameters.dataset_type` | データセット種別                                                         | `"pascal_voc"`           |
| `optimizer.lr`            | 最適化アルゴリズムの学習率                                               | `1e-4`                   |
| `optimizer.lr_backbone`   | バックボーンの学習率                                                     | `1e-5`                   |

#### 2.2 学習スクリプトの実行

```bash
uv run python main_train.py
```

> [!NOTE]
> 学習が開始されると、`results/<モデル名>_<YYYY-MM-DD_HH-MM-SS>/` ディレクトリが生成され、以下の成果物が自動保存されます。
>
> - 最良・最終モデルの重み（`best_model.pth`, `final_model.pth`）
> - 学習設定のバックアップ（`train_config.toml`, `config.json`）
> - 損失ログ CSV & 学習曲線グラフ（`log.csv`, `learning_curve.png`）
> - データ分割ログ（`input_data/train.csv`, `val.csv`, `test.csv`）
> - テストデータに対するサンプル推論結果（`test/`）

---

### 3. 推論

#### 3.1 設定ファイルの編集

[`config/inference_config.toml`](file:///c:/Users/masaki/Desktop/Python/DETR/config/inference_config.toml) を編集して、学習済みモデルへのパスと入出力ディレクトリを指定します。

| パラメータ            | 説明                                                                             | 設定例                                            |
| :-------------------- | :------------------------------------------------------------------------------- | :------------------------------------------------ |
| `train_result_path`   | 保存された学習結果ディレクトリへのパス                                           | `"./results/Deformable-DETR_2024-10-30_20-03-06"` |
| `input_path`          | 推論対象の画像ディレクトリへのパス                                               | `"../face_detection/imgs/"`                       |
| `output_path`         | 推論結果の保存ディレクトリ（空文字 `""` の場合は `train_result_path/inference`） | `""`                                              |
| `gpu`                 | 使用する GPU 番号（`-1` で CPU）                                                 | `0`                                               |
| `parameter.threshold` | バウンディングボックス検出の信頼度しきい値                                       | `0.9`                                             |

#### 3.2 推論スクリプトの実行

```bash
uv run python main_inference.py
```

検出結果（バウンディングボックスとラベル、スコアが描画された画像）が指定された出力先に保存されます。

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
│   │   ├── augmentation.py  # Albumentations によるデータ拡張
│   │   ├── dataset.py       # VOC XML / COCO データセットローダー
│   │   └── load_data.py     # データセット分割ユーティリティ
│   ├── utils/               # 汎用ユーティリティ
│   │   ├── cv2_japanese.py  # 日本語パス対応画像入出力 & テキスト描画
│   │   ├── date_str.py      # 日時文字列生成
│   │   ├── fix_seed.py      # 乱数シード固定
│   │   ├── process_time.py  # 処理時間計測コンテキストマネージャ
│   │   └── visualize.py     # バウンディングボックス描画
│   ├── custom_collate_fn.py # DataLoader 用カスタムバッチ処理
│   ├── Inference.py         # 単一画像推論エンジン
│   ├── models.py            # DETR 各種モデル初期化・バックボーン設定
│   └── trainer.py           # 学習・検証ループ & ログ記録
├── AGENTS.md                # 開発規約・コーディングガイドライン
├── LICENSE                  # Apache-2.0 ライセンス
├── main_inference.py        # 推論実行エントリポイント
├── main_train.py            # 学習実行エントリポイント
├── pyproject.toml           # プロジェクト定義 & ツール設定 (Ruff, pytest)
├── README.md                # 本ドキュメント
└── requirements.txt         # 依存パッケージ定義
```

---

## 👤 Author

- **プロジェクト作成者**: muetaek0321
- **README 作成**: Gemini 3.8 Flash

---

## 📄 ライセンス

本プロジェクトは [Apache License 2.0](LICENSE) の下で公開されています。
