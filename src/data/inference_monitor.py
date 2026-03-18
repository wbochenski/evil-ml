import os
import time
import psutil
import threading

def measure_inference(predict_func, *args):
    """
    Monitors CPU and RAM usage specifically during model inference.
    Calculates 'Pred RAM' to isolate the memory cost of the prediction.
    """
    p = psutil.Process(os.getpid())
    
    baseline_ram = p.memory_info().rss / (1024 * 1024)
    
    cpu_usage = []
    ram_usage = []
    monitoring = [True]

    def monitor_resources():
        p.cpu_percent(interval=None) 
        while monitoring[0]:
            cpu_usage.append(p.cpu_percent(interval=0.01))
            ram_usage.append(p.memory_info().rss / (1024 * 1024))

    t = threading.Thread(target=monitor_resources)
    t.start()

    start = time.time()
    y_pred = predict_func(*args)
    end = time.time()

    monitoring[0] = False
    t.join()

    # Capture RAM right after, just in case the execution was faster than the 0.01s thread tick
    final_ram = p.memory_info().rss / (1024 * 1024)
    ram_usage.append(final_ram)

    avg_cpu = sum(cpu_usage) / len(cpu_usage) if cpu_usage else 0.0
    max_cpu = max(cpu_usage) if cpu_usage else 0.0
    
    # 2. Find the absolute peak RAM used during the prediction
    max_ram = max(ram_usage)
    
    # Using max(0.0, ...) ensures we don't get negative numbers if Python's garbage collector randomly runs
    delta_ram = max(0.0, max_ram - baseline_ram)

    metrics = {
        "Time (s)": round(end - start, 4),
        "Avg CPU (%)": round(avg_cpu, 2),
        "Max CPU (%)": round(max_cpu, 2),
        # "Baseline RAM (MB)": round(baseline_ram, 2),
        # "Total RAM (MB)": round(max_ram, 2),
        "Pred RAM (MB)": round(delta_ram, 4)  
    }
    
    return y_pred, metrics