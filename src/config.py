"""Конфигурация для P300 BCI пайплайна."""

# --- Алфавит и сетка ---
ALPHABET = "1234567890_ЙЦУКЕНГШЩЗХФЫВАПРОЛДЖЭЁЯЧСМИТЬБЮЪ"
N_SYMBOLS = len(ALPHABET)  # 44
GRID_ROWS = 4
GRID_COLS = 11

# --- Параметры стимуляции ---
T0_MEAN = 1.0  # длительность активного движения буквы (сек)

# --- Наборы каналов ---
P300_CORE_CHANNELS = [
    "Fz", "Cz", "Pz", "P3", "P4", "P7", "P8",
    "Oz", "O1", "O2", "Cp1", "Cp2",
]

P300_EXTENDED_CHANNELS = [
    "Fz", "Cz", "Pz", "P3", "P4", "P7", "P8",
    "Oz", "O1", "O2", "Cp1", "Cp2",
    "Fc1", "Fc2", "C3", "C4", "Cp5", "Cp6",
]

ALL_EEG_CHANNELS = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "Ft9", "Fc5", "Fc1", "Fc2", "Fc6", "Ft10",
    "T7", "C3", "Cz", "C4", "T8",
    "Tp9", "Cp5", "Cp1", "Cp2", "Cp6", "Tp10",
    "P7", "P3", "Pz", "P4", "P8",
    "O1", "Oz", "O2", "P1", "P2",
]

CLEAR_EEG = ['Fp2','Fz','C3','Cz','C4','Cp5',
             'Cp1','Cp2','Cp6','Tp10','P7','P3',
             'P4','P8','Oz','P1','P2']

# --- Параметры эпох ---
EPOCH_TMIN = -0.2   # секунд до начала движения
EPOCH_TMAX = 0.6    # секунд после начала движения
RESAMPLE_HZ = 250   # частота дискретизации после ресемплинга
BASELINE = (None, 0)  # интервал для baseline correction

# --- Фильтрация ---
BANDPASS_LOW = 0.5   # нижняя граница полосового фильтра (Гц)
BANDPASS_HIGH = 50.0  # верхняя граница полосового фильтра (Гц)
NOTCH_FREQ = None    # частота режекторного фильтра (Гц)

# --- Артефакты ---
ARTIFACT_REJECT_UV = 150.0  # порог отбраковки (мкВ peak-to-peak)

# --- Обучение ---
DEFAULT_BATCH_SIZE = 64
DEFAULT_TARGET_WEIGHT = 1.0
DEFAULT_BALANCE_RATIO = None  # None = без undersample, используем class weights

# --- Старый словарь символов (для совместимости с Kaggle данными) ---
symbols_dict = {
    0: "a", 1: "b", 2: "c", 3: "d", 4: "e",
    5: "f", 6: "g", 7: "i", 8: "j", 9: "k",
}
