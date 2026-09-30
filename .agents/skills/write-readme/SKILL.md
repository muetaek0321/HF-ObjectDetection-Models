---
name: write-readme
description: "Pythonプロジェクトの構成・設定・ソースコードを解析し、高品質で分かりやすいREADME.mdを自動生成・更新するスキル。「READMEを作成して」「READMEを更新して」「README.mdを書いて」「ドキュメントを整備して」などの指示に対して使用します。Webアプリ（FastAPI/Django等）、CLIツール、ライブラリ/パッケージ、テンプレート、データ分析スクリプトなど、様々なPythonプロジェクトに柔軟に対応します。"
user-invocable: true
---

# Pythonプロジェクト用 README.md 自動生成・更新スキル

Pythonリポジトリの設定ファイル、ディレクトリ構造、ソースコード、開発ツール、既存ドキュメントを解析し、プロジェクトの性格（Webアプリ、API、CLI、ライブラリ、スターターテンプレート等）やツールチェーン（`uv`, `poetry`, `pip` 等）に最適化されたモダンで読みやすい `README.md` を作成・更新します。

---

## 🎯 基本原則

1. **事実に基づく正確性 (Fact-based)**:
   - `pyproject.toml` や環境設定に実在するパッケージ、定義済みスクリプト、導入済みツール（Ruff, pytest, mypy 等）のみを記述します。
   - 導入されていない架空のツールや未設定のコマンドを推測で書かないようにします。
2. **既存ドメイン知識の尊重 (Respect Existing Content)**:
   - 既存の `README.md` がある場合、全面破棄せず、固有の背景説明・アーキテクチャの動機・業務ロジックなどのドメイン知識を保持します。
   - 古くなったコマンド、ディレクトリツリー、依存関係情報を最新の状態へ更新・統合します。
3. **プロジェクト種別（Archetype）への最適化**:
   - Webアプリ/API、CLIツール、公開パッケージ、スターターテンプレートなど、対象読者が求める情報（起動コマンド、引数仕様、インストール方法、初期化手順など）にフォーカスした構成にします。
4. **Pythonモダンツールチェインの正確な反映**:
   - `uv` が使われている場合は `uv sync`, `uv run` を基本とし、`poetry` や `pip` が使われている場合はそれぞれの標準コマンドを選択します。

---

## 🔍 事前調査ステップ

コードやドキュメントを出力する前に、リポジトリ全体を調査して以下の情報を収集します。

1. **既存ドキュメントの確認**:
   - 既存の `README.md`（または `docs/`）の有無と内容を確認し、保持すべきプロジェクト概要や背景を把握します。
2. **設定ファイル・マニフェストの解析**:
   - `pyproject.toml`:
     - プロジェクト名（`project.name`）、説明（`project.description`）、必要Pythonバージョン（`requires-python`）
     - 依存関係（`dependencies`, `optional-dependencies`, `dependency-groups`）
     - エントリポイント・スクリプト（`project.scripts`, `project.gui-scripts`）
     - ツール設定（`[tool.ruff]`, `[tool.pytest.ini_options]`, `[tool.mypy]`, `[tool.uv]` 等）
   - ロックファイル・環境設定:
     - `uv.lock`（`uv` プロジェクト）
     - `poetry.lock`（`poetry` プロジェクト）
     - `requirements.txt`, `Pipfile`, `setup.py`
     - `.python-version`（指定Pythonバージョン）
   - 環境変数定義:
     - `.env.example`, `.env.sample`, `.env.template`
   - エディタ・開発支援・CI:
     - `.vscode/`（推奨拡張機能、フォーマッター設定）
     - `.github/workflows/`（CI/CDテスト・リント自動化）
     - `.agents/`, `copilot-instructions.md`, `AGENTS.md`
   - ライセンス:
     - `LICENSE`, `LICENSE.md`
3. **ソースコード・エントリポイントの確認**:
   - メインコードの配置（`src/<package_name>/`, `app/`, `main.py` 等）
   - テストコードの配置（`tests/`）
   - アプリケーションの実行方法（FastAPI/uvicorn, Click/Typer CLI, バッチスクリプトなど）
4. **プロジェクト作成者（Author）情報の取得**:
   - `git config user.name` を実行して作成者名を取得。
   - `pyproject.toml` の `authors` 欄も確認。

---

## 🏷️ Pythonプロジェクト種別の判定

収集した情報からプロジェクトの種別を特定し、強調すべきセクションを決定します。

| 種別 | 判定の目安 | READMEで強調すべき内容 |
| :--- | :--- | :--- |
| **スターター / テンプレート** | プレースホルダー（`{{...}}`）やテンプレート表記がある | テンプレート適用後の初期設定（`name`, `description` 書換手順）、同梱スタック |
| **Web アプリ / API** | `fastapi`, `uvicorn`, `django`, `flask` 等の依存 | サーバー起動コマンド、APIドキュメント（Swagger/OpenAPI）のURL、環境変数設定 |
| **CLI ツール** | `project.scripts` 定義、`click`, `typer`, `argparse` | コマンドの構文・オプション一覧、実行例、ヘルプ出力 |
| **ライブラリ / パッケージ** | `[build-system]` 定義、再利用モジュール構成 | インストール方法（`pip install` 等）、インポート例、クイックスタートコード |
| **データ分析 / スクリプト** | `pandas`, `numpy`, `jupyter`, 単一目的のスクリプト | データの配置場所、スクリプト実行順序、出力成果物の説明 |

---

## 📝 `README.md` の構成案

プロジェクト種別に応じて必要なセクションを選択・調整して構成します。

```markdown
# [プロジェクト名]

[プロジェクトの一行要約・キャッチコピー]

## 🚀 特徴
- [特徴1: 例: uv による高速な環境管理と依存関係解決]
- [特徴2: 例: FastAPI による非同期ハイパフォーマンス API]
- [特徴3: 例: Ruff & pytest による堅牢なコード品質維持]

## 🛠️ 技術スタック
- **Python**: `>= 3.13`
- **パッケージ管理**: [uv](https://github.com/astral-sh/uv) （または Poetry / pip）
- **主要フレームワーク / ライブラリ**: [例: FastAPI, Pydantic, SQLAlchemy]
- **品質・テストツール**: [Ruff](https://github.com/astral-sh/ruff), [pytest](https://docs.pytest.org/)

## 📋 前提条件
- Python [指定バージョン] 以上
- [uv](https://docs.astral.sh/uv/)（推奨）などのツール

## 🏁 使い方 (Getting Started)

<!-- テンプレートプロジェクトの場合のみ記載 -->
### 1. プロジェクトの初期設定
テンプレートからプロジェクトを作成した場合、`pyproject.toml` を開き、以下の項目をお使いのプロジェクト情報に書き換えてください。
- `name = "your-project-name"`
- `description = "Your project description"`

### 2. 環境構築
依存関係をインストールし、仮想環境を自動セットアップします。
```bash
uv sync
```

<!-- .env.example がある場合 -->
### 3. 環境変数の設定
設定サンプルをコピーして環境変数ファイルを作成します。
```bash
cp .env.example .env
```
| 環境変数 | 必須 | 説明 | デフォルト値 |
| :--- | :---: | :--- | :--- |
| `DATABASE_URL` | ○ | 接続先データベースURL | `sqlite:///./app.db` |

### 4. 実行 / 起動
```bash
# Webアプリの場合（例）
uv run uvicorn app.main:app --reload

# CLI / スクリプトの場合（例）
uv run python -m src.main
```

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

<!-- mypy / pyright が導入されている場合 -->
### 型チェック
```bash
uv run mypy .
```

<!-- pytest が導入されている場合 -->
### テスト実行 (pytest)
```bash
uv run pytest
```

## 📂 ディレクトリ構成
```text
.
├── src/                  # アプリケーションソースコード
│   └── app/
│       ├── api/          # エンドポイント定義
│       └── main.py       # アプリケーションエントリポイント
├── tests/                # テストコード
├── .env.example          # 環境変数サンプル
├── .python-version       # Pythonバージョン指定
├── pyproject.toml        # プロジェクト定義 & ツール設定
└── README.md             # 本ドキュメント
```

## 👤 Author
- **プロジェクト作成者**: [git config user.name で取得したユーザー名]
- **README 作成**: [実行中のAIモデル名（例: Gemini 3.8 Flash）]

<!-- LICENSEファイルがある場合 -->
## 📄 ライセンス
[LICENSEファイルに基づくライセンス名（例: MIT License）]
```

---

## 📁 ディレクトリ構成ツリー作成の指針

1. **ノイズの徹底排除**:
   - 以下のディレクトリ・ファイルは**必ず除外**してください。
     - `.venv/`, `venv/`, `env/`（仮想環境）
     - `__pycache__/`, `*.pyc`（バイトコードキャッシュ）
     - `.pytest_cache/`, `.ruff_cache/`, `.mypy_cache/`（ツールキャッシュ）
     - `.git/`（Git管理ディレクトリ）
     - `dist/`, `build/`, `*.egg-info/`（ビルド成果物）
2. **インライン注釈の付与**:
   - 各主要ディレクトリ・ファイルに `# 説明` を添えて、役割が一目でわかるようにします。
3. **階層の深さ**:
   - 原則 2〜3 階層にとどめ、本質的な構造が伝わるようにします。

---

## 🎨 出力スタイル要件

- **見出しと絵文字**: 各セクション見出しに適度な絵文字（🚀, 🛠️, 📋, 🏁, 💻, 📂, 👤 等）を配置し、視認性を高めます。
- **シンタックスハイライト**: コードブロックには必ず適切な言語タグ（`bash`, `toml`, `text`, `python` 等）を指定します。
- **パッケージマネージャーの整合性**:
   - `uv` が導入されている場合は `uv` を優先。
   - `poetry` が導入されている場合は `poetry install`, `poetry run pytest`, `poetry add` を使用。
   - 標準 `pip` のみの場合は `python -m venv .venv`, `pip install -r requirements.txt` を使用。

---

## 💬 呼び出し例

ユーザーが以下のように要求した際に本スキルを起動します：

- `READMEを作成して`
- `このプロジェクトのREADME.mdを書いて`
- `READMEを最新の構成・コマンドに合わせて更新して`
- `開発環境や使い方がわかるようにREADME.mdを整備して`
