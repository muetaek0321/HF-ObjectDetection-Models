import logging
from pathlib import Path


def logging_config(debug: bool, log_file: str | Path | None = None) -> None:
    """ロギングの設定を行う関数（標準出力とファイル出力に対応）

    Args:
        debug (bool): デバッグモードかどうか
        log_file (str | Path | None, optional): ログ出力先のファイルパス。指定した場合はファイルにも出力する。Defaults to None.
    """
    handlers: list[logging.Handler] = [logging.StreamHandler()]

    # ファイルパスが設定されている場合はファイル出力用のハンドラを追加
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))

    # ロギングの基本設定
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
        force=True,
    )

    # ロガーレベルの個別設定
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def get_logger(
    name: str, debug: bool = False, log_file: str | Path | None = None
) -> logging.Logger:
    """ロガーを取得する関数

    Args:
        name (str): ロガーの名前
        debug (bool): デバッグモードかどうか
        log_file (str | Path | None, optional): ログ出力先のファイルパス。指定した場合はファイルにも出力する。Defaults to None.

    Returns:
        logging.Logger: ロガーのインスタンス
    """
    # ロギングの設定
    logging_config(debug, log_file=log_file)

    # ロガー取得
    logger = logging.getLogger(name)

    return logger
