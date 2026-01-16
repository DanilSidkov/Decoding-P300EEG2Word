# tools/lsl_monitor.py
from pylsl import resolve_streams, StreamInlet
import json

def monitor_markers():
    """Мониторинг LSL маркеров"""
    print("Поиск LSL потоков...")
    
    streams = resolve_streams()
    print(f"Найдено {len(streams)} потоков:")
    
    for i, stream in enumerate(streams):
        print(f"{i+1}. {stream.name()} ({stream.type()})")
    
    # Подключаемся к потоку маркеров
    marker_streams = resolve_streams('type', 'Markers')
    
    if not marker_streams:
        print("Маркеры не найдены!")
        return
    
    inlet = StreamInlet(marker_streams[0])
    print(f"\nПодключено к потоку: {marker_streams[0].name()}")
    print("Ожидание маркеров... (Ctrl+C для выхода)\n")
    
    try:
        while True:
            sample, timestamp = inlet.pull_sample(timeout=1.0)
            if sample:
                try:
                    marker_data = json.loads(sample[0])
                    print(f"🕒 {timestamp:.6f} | {marker_data['event']}")
                    if marker_data.get('data'):
                        print(f"   Данные: {marker_data['data']}")
                except:
                    print(f"🕒 {timestamp:.6f} | {sample[0]}")
    except KeyboardInterrupt:
        print("\nМониторинг завершен")

if __name__ == "__main__":
    monitor_markers()