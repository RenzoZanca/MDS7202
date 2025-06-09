#lab9/dags/dag_lineal.py

from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator
from datetime import datetime
import os

from hiring_functions import create_folders, split_data, preprocess_and_train, gradio_interface

def download_data(**kwargs):
    execution_date = kwargs['ds']
    base_path = os.path.join(os.getcwd(), execution_date, 'raw')
    os.makedirs(base_path, exist_ok=True)
    os.system(f"curl -o {os.path.join(base_path, 'data_1.csv')} https://gitlab.com/eduardomoyab/laboratorio-13/-/raw/main/files/data_1.csv")

# funcion para poder manejar los argumentos de las funciones
def wrap_function(func):
    def wrapper(**kwargs):
        func(**kwargs)
    return wrapper

default_args = {
    'start_date': datetime(2024, 10, 1),
}

with DAG(
    dag_id='hiring_lineal',
    default_args=default_args,
    schedule=None,
    catchup=False
) as dag:

    start = EmptyOperator(
        task_id='start'
    )

    create_dirs = PythonOperator(
        task_id='create_folders',
        python_callable=create_folders
    )

    download = PythonOperator(
        task_id='download_data',
        python_callable=download_data
    )

    split = PythonOperator(
        task_id='split_data',
        python_callable=split_data
    )

    train = PythonOperator(
        task_id='train_model',
        python_callable=preprocess_and_train
    )

    deploy = PythonOperator(
        task_id='gradio_ui',
        python_callable=gradio_interface
    )

    # flujo de tareas
    start >> create_dirs >> download >> split >> train >> deploy
