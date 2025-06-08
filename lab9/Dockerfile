# Imagen base recomendada
FROM python:3.10-slim

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Instala dependencias del sistema necesarias
RUN apt-get update && \
    apt-get install -y curl gcc g++ build-essential libffi-dev libpq-dev git && \
    apt-get clean

# Configura pip para evitar errores SSL intermitentes
RUN pip config set global.index-url https://pypi.org/simple

# Copia el archivo de dependencias primero para aprovechar el cache
COPY requirements.txt .

# Instala las dependencias de Python
RUN pip install --no-cache-dir --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt

# Copia el resto de la app (incluye dags/, hiring_functions.py, etc.)
COPY . .

# Exponer el puerto de Airflow webserver
EXPOSE 8080

# Comando por defecto al iniciar el contenedor
CMD ["airflow", "standalone"]