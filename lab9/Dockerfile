# lab9/Dockerfile

FROM python:3.10-slim

# 1. Variables de entorno de Airflow
ENV AIRFLOW_HOME=/app/airflow
ENV AIRFLOW__CORE__LOAD_EXAMPLES=False
ENV AIRFLOW__CORE__EXECUTOR=SequentialExecutor

WORKDIR ${AIRFLOW_HOME}

# 2. Dependencias del SO
RUN apt-get update && \
    apt-get install -y curl gcc g++ build-essential && \
    rm -rf /var/lib/apt/lists/*

# 3. Copiar e instalar Python requirements (incluye apache-airflow)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Inicializar Airflow
#    Crea el directorio AIRFLOW_HOME y la base de datos SQLite
RUN mkdir -p "${AIRFLOW_HOME}/dags" \
    && airflow db migrate

# 5. Copiar tu código
COPY dags/ "${AIRFLOW_HOME}/dags/"
COPY plugins/ "${AIRFLOW_HOME}/plugins/"
COPY dags/hiring_functions.py ${AIRFLOW_HOME}/hiring_functions.py

# 6. Exponer puerto de la UI
EXPOSE 8080

# 7. Comando por defecto
#    'airflow standalone' arranca webserver + scheduler en SQLite
CMD ["airflow", "standalone"]
