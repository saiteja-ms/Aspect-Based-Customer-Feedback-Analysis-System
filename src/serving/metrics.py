from prometheus_client import start_http_server, Gauge
import time
import os

# Export custom metrics (example: item pool size, model version)
ITEMS_GAUGE = Gauge("nextpick_item_count", "Number of items in candidate pool")

def run_metrics_exporter(item_count=0, port=8001):
    ITEMS_GAUGE.set(item_count)
    start_http_server(port)
    print("Prometheus metrics exporter running on port", port)
    try:
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    run_metrics_exporter(item_count=10000, port=int(os.getenv("METRICS_PORT", 8001)))
