# app/lsl_markers.py
from pylsl import StreamInfo, StreamOutlet, local_clock
import json
import threading
import time
from typing import Any, Dict

class LSLMarkerService:
    """Сервис для отправки маркеров через LSL"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.initialized = False
            self.outlet = None
            self._initialize_lsl()
    
    def _initialize_lsl(self):
        """Инициализация LSL потока для маркеров"""
        try:
            info = StreamInfo(
                name='BCI_Experiment_Markers',
                type='Markers',
                channel_count=1,
                nominal_srate=0,
                channel_format='string',
                source_id='bci-experiment-interface'
            )
            
            # Добавляем метаданные
            info.desc().append_child_value("manufacturer", "BCI-Lab")
            channels = info.desc().append_child("channels")
            channels.append_child("channel")\
                .append_child_value("label", "markers")\
                .append_child_value("type", "marker")\
                .append_child_value("unit", "string")
            
            self.outlet = StreamOutlet(info)
            self.initialized = True
            print("✓ LSL поток маркеров создан")
        except Exception as e:
            print(f"✗ Ошибка инициализации LSL: {e}")
            self.initialized = False
    
    def send_marker(self, event_type: str, data: Dict[str, Any] = None):
        """Отправка маркера через LSL"""
        if not self.initialized or not self.outlet:
            print("LSL не инициализирован")
            return
        
        marker = {
            "timestamp": local_clock(),
            "event": event_type,
            "data": data or {},
            "system_time": time.time()
        }
        
        try:
            # Отправляем как JSON строку
            marker_str = json.dumps(marker, ensure_ascii=False)
            self.outlet.push_sample([marker_str])
            print(f"📤 LSL маркер: {event_type}")
        except Exception as e:
            print(f"Ошибка отправки маркера: {e}")
    
    def send_simple_marker(self, event_type: str):
        """Отправка простого маркера (только тип события)"""
        if not self.initialized or not self.outlet:
            return
        
        try:
            self.outlet.push_sample([event_type])
            print(f"📤 Простой маркер: {event_type}")
        except Exception as e:
            print(f"Ошибка отправки простого маркера: {e}")