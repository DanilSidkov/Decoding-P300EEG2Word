# lsl_file_logger.py
import threading
import queue
import time
from pathlib import Path

class LSLFileLogger:
    def __init__(self, log_dir="logs", filename="lsl_markers.log"):
        self.log_dir = Path(__file__).parent / log_dir
        self.log_dir.mkdir(exist_ok=True)
        self.log_file = self.log_dir / filename
        self.queue = queue.Queue()
        self.running = True
        self.thread = threading.Thread(target=self._writer, daemon=True)
        self.thread.start()

    def log(self, marker_string):
        """Добавить маркер в очередь на запись"""
        self.queue.put((time.time(), marker_string))

    def _writer(self):
        with open(self.log_file, "a", encoding="utf-8") as f:
            while self.running:
                try:
                    timestamp, msg = self.queue.get(timeout=1)
                    f.write(f"{timestamp:.6f} - {msg}\n")
                    f.flush()  # можно убрать, если не критично
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"Ошибка записи LSL: {e}")

    def stop(self):
        self.running = False
        self.thread.join(timeout=2)