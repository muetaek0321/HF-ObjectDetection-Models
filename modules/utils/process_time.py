import logging
import time


class ProcessTimeManager:
    """実行時間計測をするコンテキストマネージャー"""

    def __init__(
        self, is_print: bool = False, desc: str = "Process Time", logger: logging.Logger = None
    ) -> None:
        """

        Args:
            is_print (bool): 計測後に標準出力をするか指定
            desc (str): 標準出力の際に一緒に表示する文字列
            logger (logging.Logger): ロガーのインスタンス。Noneの場合は標準出力
        """
        self.is_print = is_print
        self.desc = desc
        self.logger = logger

    def __enter__(self):
        """with句に入るときの処理"""
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """with句を出るときの処理"""
        self.proc_time = time.perf_counter() - self.start_time

        if self.is_print:
            if self.logger:
                self.logger.info(f"{self.desc}: {self.proc_time:.2f}")
            else:
                print(f"{self.desc}: {self.proc_time:.2f}")
