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

from src.inference.realtime_inference import (
    InferenceConfig,
    RealtimeInference,
)


def main() -> None:
    p = argparse.ArgumentParser(description="P300 online inference")
    p.add_argument("--model", type=str, default="models/p300_model.pth")
    p.add_argument("--marker-stream", type=str, default="annotations")
    p.add_argument("--feedback-stream", type=str, default="p300_feedback")
    p.add_argument("--buffer-sec", type=float, default=10.0)
    p.add_argument("--filter-pad", type=float, default=1.0)
    p.add_argument(
        "--average-last", type=int, default=None,
        help="Усреднять только последние N репетиций на букву "
             "(например 5 из 10).",
    )
    p.add_argument(
        "--eeg-unit-scale", type=float, default=1e-6,
        help="Множитель перевода сэмплов потока в вольты "
             "(1e-6 если поток в µV; 1.0 если уже в V).",
    )
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--log", type=str, default=None)
    args = p.parse_args()

    cfg = InferenceConfig(
        model_path=args.model,
        marker_stream_name=args.marker_stream,
        feedback_stream_name=args.feedback_stream,
        buffer_capacity_sec=args.buffer_sec,
        filter_pad_sec=args.filter_pad,
        average_last=args.average_last,
        device=args.device,
        eeg_unit_scale=args.eeg_unit_scale,
        log_path=args.log,
    )

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
