"""Real-time P300 inference pipeline.

Модули:
- eeg_buffer: кольцевой буфер ЭЭГ с метками времени LSL
- lsl_streams: резолвинг и чтение EEG/marker потоков
- online_preprocessor: фильтрация/ресемплинг/нормализация на эпоху
- epoch_builder: нарезка эпох по маркерам
- letter_aggregator: накопление P(target) и решение по trial
- realtime_inference: главный оркестратор
"""
