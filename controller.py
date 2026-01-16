# app/controller.py
from pylsl import resolve_streams
import subprocess
import time
import json
from datetime import datetime
from pathlib import Path
from app.lsl_markers import LSLMarkerService


class ExperimentController:
    def __init__(self):
        self.neorec_process = None
        self.lsl_service = LSLMarkerService()
        self.experiment_data = {
            "start_time": None,
            "end_time": None,
            "markers": [],
            "parameters": {}
        }
        
    def check_lsl_streams(self):
        """Проверка доступных LSL потоков"""
        print("Проверка LSL потоков...")
        streams = resolve_streams()
        for stream in streams:
            print(f"  - {stream.name()} ({stream.type()})")
        return len(streams) > 0
    
    def start_neorec(self, experiment_id: str):
        """Запуск NeoRec с передачей ID эксперимента"""
        # Пример команды для NeoRec
        cmd = [
            "NeoRec.exe",
            "--experiment-id", experiment_id,
            "--lsl-markers", "BCI_Experiment_Markers",
            "--output", f"data/{experiment_id}/eeg_data.eeg"
        ]
        
        try:
            self.neorec_process = subprocess.Popen(cmd)
            self.lsl_service.send_marker("NEOREC_STARTED", 
                                        experiment_id=experiment_id)
            return True
        except Exception as e:
            print(f"Ошибка запуска NeoRec: {e}")
            return False
    
    def start_experiment_session(self, params: dict):
        """Начало сессии эксперимента"""
        experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print(f"Начало эксперимента {experiment_id}")
        print(f"Параметры: {params}")
        
        # 1. Проверяем LSL
        if not self.check_lsl_streams():
            print("Предупреждение: LSL потоки не обнаружены")
        
        # 2. Запускаем NeoRec
        print("Запуск NeoRec...")
        if not self.start_neorec(experiment_id):
            return False
        
        # 3. Даем время на запуск
        time.sleep(3)
        
        # 4. Отправляем маркер начала
        self.lsl_service.send_marker("SESSION_START", {
            "experiment_id": experiment_id,
            "parameters": params,
            "timestamp": time.time()
        })
        
        self.experiment_data = {
            "experiment_id": experiment_id,
            "start_time": datetime.now().isoformat(),
            "parameters": params,
            "markers": []
        }
        
        return True
    
    def stop_experiment(self):
        """Завершение эксперимента"""
        if self.neorec_process:
            self.neorec_process.terminate()
            self.neorec_process.wait()
        
        self.experiment_data["end_time"] = datetime.now().isoformat()
        
        # Сохраняем данные
        output_file = Path(f"data/{self.experiment_data['experiment_id']}/experiment.json")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(self.experiment_data, f, indent=2, ensure_ascii=False)
        
        self.lsl_service.send_marker("SESSION_END", {
            "experiment_id": self.experiment_data["experiment_id"],
            "duration": time.time() - self.experiment_data.get("start_timestamp", 0)
        })
        
        print(f"Эксперимент завершен. Данные сохранены в {output_file}")