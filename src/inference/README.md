# P300 Real-time Inference

Онлайн-инференс обученной модели `P300EEGNet`: читаем EEG из LSL-потока
NeoRec, маркеры — из стимулятора, возвращаем предсказанную таргет-букву
в стимулятор по LSL, где она подсвечивается.

## Архитектура

```
┌────────────────────────┐   annotations (markers)   ┌────────────────────┐
│ BCI_P300/               │ ───────────────────────►  │ src/inference/      │
│ experiment_online.py   │                            │ realtime_inference  │
│ (pygame, летающие       │   p300_feedback (pred)    │ (EEG buffer +       │
│  буквы, LSL outlet)    │ ◄───────────────────────  │  препроц + модель)  │
└────────────────────────┘                            └────────────────────┘
                                                               ▲
                                                               │ EEG LSL
                                                               │
                                                       ┌───────────────┐
                                                       │ NeoRec 1.6    │
                                                       │ (LSL stream)  │
                                                       └───────────────┘
```

**LSL-потоки:**

| Поток | Направление | Payload |
|---|---|---|
| NeoRec / type="EEG" | NeoRec → инференс | float samples, 34 канала |
| `annotations` (Markers, string) | стимулятор → инференс | `T:<letter>`, `F:<letter>`, `E`, `<ms>` (sync) |
| `p300_feedback` (Markers, string) | инференс → стимулятор | `READY`, `P:<letter>` |

**Протокол trial:**

1. Стимулятор показывает таргет-букву в рамке (cue, 1.5с).
2. Посылает `T:<target>`.
3. Все 44 буквы двигаются независимо; каждый раз, когда буква начинает
   активное движение (т.е. onset P300-эпохи), посылается `F:<letter>`.
4. Когда каждая буква стартовала ≥ `n_repetitions` (=10) раз, стимулятор
   посылает `E`.
5. Инференс нарезает окно `[tmarker−0.2, tmarker+0.6]` из EEG-буфера,
   прогоняет через модель, копит `P(target)` по буквам. На `E` делает
   `argmax` и возвращает `P:<letter>`.
6. Стимулятор подсвечивает эту букву зелёным (если совпало с cue) или
   красным (если нет) на `feedback_duration_ms` мс и переходит к
   следующему таргету из `sentence`.

## Запуск

### 1. NeoRec 1.6
В NeoRec включи LSL broadcast (см. руководство оператора). Убедись, что
частота дискретизации и имена каналов совпадают с тем, на чём училась
модель (см. `checkpoint['channel_names']`).

### 2. Инференс
```bash
cd <repo>
python -m src.inference.run_inference \
    --model models/p300_model.pth \
    --log reports/online_trials.json
```

Скрипт:
- резолвит EEG-поток (по имени "NeoRec" / "NVX136" / "EEG" или по
  `type=EEG`),
- резолвит marker stream `annotations`,
- создаёт свой outlet `p300_feedback`,
- прогревает буфер 2с и шлёт `READY`.

### 3. Стимулятор
```bash
cd BCI_P300
python experiment_online.py
```

Стимулятор сначала ждёт `READY` (≤ 15с), затем крутит буквы по `sentence`
из `settings.json`.

### Ключевые параметры CLI инференса

| Флаг | Значение |
|---|---|
| `--model` | путь к чекпоинту (`models/p300_model.pth`) |
| `--average-last N` | усреднять только последние N репетиций на букву (например 5 из 10 — второй блок репетиций, где испытуемый "прогрет") |
| `--buffer-sec` | длина EEG ring buffer (≥ tmax + filter_pad + запас) |
| `--filter-pad` | секунды-запас для FIR фильтра (1.0 достаточно) |
| `--eeg-unit-scale` | 1e-6 если поток в µV (стандарт NeoRec), 1.0 если уже в V |
| `--log` | JSON-лог трайлов (target/predicted/scores) |

### Online settings

В `BCI_P300/settings.json` появились (с дефолтами):

| Ключ | Значение |
|---|---|
| `n_repetitions` | сколько раз каждая буква должна стартовать в trial (= 10) |
| `feedback_duration_ms` | длительность подсветки предсказанной буквы |
| `wait_feedback_timeout_ms` | сколько ждать `P:*` от инференса |
| `wait_inference_ready_sec` | сколько ждать `READY` при старте |

## Внутренние детали

- **EEG ring buffer** (`eeg_buffer.py`) — потокобезопасный кольцевой
  массив на 10с с LSL timestamps. `get_window(t0, t1)` делает бинарный
  поиск и возвращает срез.
- **EEGReader** — фоновый поток, тянет `pull_chunk` у LSL inlet.
- **OnlineEpochPreprocessor** — для каждой эпохи:
  1. выбирает каналы модели (по `channel_names` из чекпоинта),
  2. строит `mne.io.RawArray` с запасом ±`filter_pad_sec`,
  3. bandpass 0.5–50 Гц (FIR), avg re-reference, resample на 250 Гц,
  4. вырезает точное окно `[tmin, tmax]`,
  5. baseline correction `(None, 0)`,
  6. robust-нормализация по `norm_stats` из чекпоинта.

  Это 1-в-1 воспроизводит offline-пайплайн `preprocess_raw` +
  `create_mne_epochs`.

- **Маркеры-числа** (`sync`) игнорируются — они остаются для
  совместимости с офлайн-логом.

## Диагностика

Если в консоли инференса:
- `Нет окна EEG для '<letter>'` — значит EEG-буфер не успевает. Проверь
  `latest_timestamp` и увеличь `--buffer-sec`.
- `Ни один целевой канал не найден` — имена каналов в LSL не совпадают с
  `channel_names` из чекпоинта. Проверь настройки NeoRec.
- `таймаут READY` — инференс ещё не запущен. Сначала запускай его,
  потом стимулятор.
