"""CLI для запуска real-time P300 инференса.

Пример:
    python -m src.inference.run_inference \
        --model models/p300_model.pth \
        --marker-stream annotations \
        --log reports/online_trials.json
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

from src.inference.realtime_inference import (
    InferenceConfig,
    RealtimeInference,
)

try:
    import yaml as _yaml
except ImportError:
    _yaml = None  # type: ignore[assignment]


def _load_yaml_config(path: str) -> dict:
    if _yaml is None:
        raise ImportError("pyyaml не установлен. Установите: pip install pyyaml")
    with open(path, encoding="utf-8") as f:
        return _yaml.safe_load(f) or {}


def main() -> None:
    p = argparse.ArgumentParser(description="P300 online inference")
    p.add_argument(
        "--config", type=str, default=None,
        help="Путь к YAML-конфигу (например configs/inference.yaml). "
             "CLI-аргументы перекрывают значения из файла.",
    )
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--marker-stream", type=str, default=None)
    p.add_argument("--feedback-stream", type=str, default=None)
    p.add_argument("--buffer-sec", type=float, default=None)
    p.add_argument("--filter-pad", type=float, default=None)
    p.add_argument(
        "--average-last", type=int, default=None,
        help="Усреднять только последние N репетиций на букву.",
    )
    p.add_argument(
        "--eeg-unit-scale", type=float, default=None,
        help="Множитель перевода сэмплов потока в вольты "
             "(1e-6 если поток в µV; 1.0 если уже в V).",
    )
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--log", type=str, default=None)
    p.add_argument(
        "--lsl-channel-names", type=str, default=None,
        help="Имена каналов через запятую (перекрывает конфиг).",
    )
    args = p.parse_args()

    # --- базовые дефолты ---
    defaults: dict = {
        "model_path": "models/p300_model.pth",
        "marker_stream_name": "annotations",
        "feedback_stream_name": "p300_feedback",
        "buffer_capacity_sec": 10.0,
        "filter_pad_sec": 1.0,
        "average_last": None,
        "device": None,
        "eeg_unit_scale": 1e-6,
        "log_path": None,
        "lsl_channel_names": None,
    }

    # --- значения из YAML (перекрывают дефолты) ---
    config_path = args.config or (
        "configs/inference.yaml"
        if Path("configs/inference.yaml").exists()
        else None
    )
    if config_path is not None:
        if not Path(config_path).exists():
            print(f"[cfg] Конфиг не найден: {config_path}", file=sys.stderr)
            sys.exit(1)
        file_cfg = _load_yaml_config(config_path)
        # yaml ключи: model_path, marker_stream_name и т.д.
        key_map = {
            "model_path": "model_path",
            "marker_stream_name": "marker_stream_name",
            "feedback_stream_name": "feedback_stream_name",
            "buffer_capacity_sec": "buffer_capacity_sec",
            "filter_pad_sec": "filter_pad_sec",
            "average_last": "average_last",
            "device": "device",
            "eeg_unit_scale": "eeg_unit_scale",
            "log_path": "log_path",
            "lsl_channel_names": "lsl_channel_names",
        }
        for yaml_key, cfg_key in key_map.items():
            if yaml_key in file_cfg:
                defaults[cfg_key] = file_cfg[yaml_key]
        print(f"[cfg] Загружен конфиг: {config_path}")

    # --- CLI перекрывает файл (только если явно передан) ---
    cli_map = {
        "model": "model_path",
        "marker_stream": "marker_stream_name",
        "feedback_stream": "feedback_stream_name",
        "buffer_sec": "buffer_capacity_sec",
        "filter_pad": "filter_pad_sec",
        "average_last": "average_last",
        "device": "device",
        "eeg_unit_scale": "eeg_unit_scale",
        "log": "log_path",
    }
    for arg_attr, cfg_key in cli_map.items():
        val = getattr(args, arg_attr.replace("-", "_"), None)
        if val is not None:
            defaults[cfg_key] = val

    if args.lsl_channel_names:
        defaults["lsl_channel_names"] = [
            c.strip() for c in args.lsl_channel_names.split(",")
        ]

    cfg = InferenceConfig(**defaults)

    rt = RealtimeInference(cfg)

    def _handler(sig, frame):  # noqa: ARG001
        print("\n[RT] Останавливаем...")
        rt.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _handler)
    try:
        signal.signal(signal.SIGTERM, _handler)
    except Exception:
        pass

    rt.start()
    print("[RT] Инференс запущен. Ctrl+C для остановки.")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        _handler(None, None)


if __name__ == "__main__":
    main()
