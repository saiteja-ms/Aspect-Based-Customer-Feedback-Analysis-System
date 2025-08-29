import mlflow
from contextlib import contextmanager

@contextmanager
def start_mlflow_run(name):
    mlflow.set_experiment("nextpick_experiment")
    run = mlflow.start_run(run_name=name)
    try:
        yield run
    finally:
        mlflow.end_run()
