# lab9/dags/dag_dynamic.py

import os
from datetime import datetime

from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.trigger_rule import TriggerRule


from hiring_dynamic_functions import (
    create_folders,
    load_ands_merge,
    split_data,
    train_model,
    evaluate_models
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# 1) Argumentos por defecto del DAG
default_args = {
    'start_date': datetime(2024, 10, 1),
}

def branch_download(**kwargs):
    """
    Si la fecha de ejecución (ds) es anterior al 2024-11-01, solo
    descargamos data_1; a partir de ese día, descargamos data_1 y data_2.
    """
    ds = kwargs['ds']
    if ds < "2024-11-01":
        return 'download_data_1'
    else:
        return 'download_data_1_and_2'


with DAG(
    dag_id='hiring_dynamic',
    default_args=default_args,
    schedule='0 15 5 * *',   # 05 de cada mes a las 15:00 UTC
    catchup=True,            # habilita backfill
    tags=['lab9']
) as dag:

    # 2) Placeholder de inicio
    start = EmptyOperator(task_id='start')

    # 3) Crear carpetas
    t_create = PythonOperator(
        task_id='create_folders',
        python_callable=create_folders,

    )

    # 4) Branching para descargar data_1.csv vs. data_1+data_2
    t_branch = BranchPythonOperator(
        task_id='branch_download',
        python_callable=branch_download,

    )

    # Descarga data_1.csv
    t_dl1 = BashOperator(
    task_id='download_data_1',
    bash_command=(
        "mkdir -p ${AIRFLOW_HOME}/{{ ds }}/raw && "
        "curl -s -o ${AIRFLOW_HOME}/{{ ds }}/raw/data_1.csv "
        "https://gitlab.com/eduardomoyab/laboratorio-13/-/raw/main/files/data_1.csv"
    )
    )


    # Descarga data_1.csv y data_2.csv
    t_dl12 = BashOperator(
        task_id='download_data_1_and_2',
    bash_command=(
        "mkdir -p ${AIRFLOW_HOME}/{{ ds }}/raw && "
        "curl -s -o ${AIRFLOW_HOME}/{{ ds }}/raw/data_1.csv "
          "https://gitlab.com/eduardomoyab/laboratorio-13/-/raw/main/files/data_1.csv && "
        "curl -s -o ${AIRFLOW_HOME}/{{ ds }}/raw/data_2.csv "
          "https://gitlab.com/eduardomoyab/laboratorio-13/-/raw/main/files/data_2.csv"
    )
)


    # Un operator para unir ambas ramas de descarga
    t_join_download = EmptyOperator(
        task_id='join_download',
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS
    )

    # 5) Concatenar datasets disponibles
    t_merge = PythonOperator(
        task_id='load_and_merge',
        python_callable=load_ands_merge,

    )

    # 6) Hold-out y guardado de splits
    t_split = PythonOperator(
        task_id='split_data',
        python_callable=split_data,

    )

    # 7) Entrenamientos en paralelo
    t_rf = PythonOperator(
        task_id='train_rf',
        python_callable=train_model,
        op_kwargs={
            'model': RandomForestClassifier(random_state=42),
            'model_name': 'rf'
        },

    )
    t_svm = PythonOperator(
        task_id='train_svm',
        python_callable=train_model,
        op_kwargs={
            'model': SVC(probability=True, random_state=42),
            'model_name': 'svm'
        },

    )
    t_dt = PythonOperator(
        task_id='train_dt',
        python_callable=train_model,
        op_kwargs={
            'model': DecisionTreeClassifier(random_state=42),
            'model_name': 'dt'
        },

    )

    # 8) Evaluar y elegir el mejor modelo
    t_eval = PythonOperator(
        task_id='evaluate_models',
        python_callable=evaluate_models,

        trigger_rule=TriggerRule.ALL_SUCCESS
    )

    # Definición del flujo
    start \
      >> t_create \
      >> t_branch \
      >> [t_dl1, t_dl12] \
      >> t_join_download \
      >> t_merge \
      >> t_split \
      >> [t_rf, t_svm, t_dt] \
      >> t_eval
