# терминал 1 — инференс
python -m src.inference.run_inference --model models/p300_model.pth \
    --average-last 5 --log reports/online_trials.json
# терминал 2 — стимулятор (после "READY")
python BCI_P300/experiment_online.py


1. NeoRec 1.6 — запустить, подключить усилитель NVX136, убедиться что идёт сигнал, и включить LSL broadcast (в настройках NeoRec галочка/чекбокс про LSL — после этого в сети появляется EEG-поток).

2. Терминал 1 — инференс:

python -m src.inference.run_inference --model models/p300_model.pth --average-last 5 --log reports/online_trials.json

Дождись в логе:

EEG stream: name='NeoRec', sfreq=..., n_ch=34 — поток найден
Модель загружена: 18 каналов, 201 точка — чекпоинт ок
Получено N сэмплов. Готов. — буфер прогрет, отправлен READY
Инференс запущен. Ctrl+C для остановки.
Если на этом шаге ошибка "LSL EEG-поток не найден" — значит NeoRec LSL broadcast не включён, возвращайся к шагу 1.

3. Терминал 2 — стимулятор:

python BCI_P300/experiment_online.py

Стимулятор сам увидит READY от инференса (ждёт до 15с), затем начнёт показывать буквы из sentence.

Важные нюансы порядка:

Если запустить стимулятор раньше инференса, он прождёт wait_inference_ready_sec (15с), не дождётся READY и всё равно начнёт — но первые trial'ы уйдут "в никуда". Лучше так не делать.
Инференс должен запуститься после NeoRec, иначе resolve_eeg_stream сразу упадёт.
Остановка: Ctrl+C в терминале 1 (инференс сохранит JSON-лог), Esc или Space во время trial в окне стимулятора