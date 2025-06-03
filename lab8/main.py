# main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pickle
import pandas as pd
import uvicorn

# 1) Definimos el esquema de entrada con Pydantic
class WaterSample(BaseModel):
    ph: float = Field(..., example=7.0)
    Hardness: float = Field(..., example=150.0)
    Solids: float = Field(..., example=20000.0)
    Chloramines: float = Field(..., example=3.0)
    Sulfate: float = Field(..., example=300.0)
    Conductivity: float = Field(..., example=400.0)
    Organic_carbon: float = Field(..., example=10.0)
    Trihalomethanes: float = Field(..., example=70.0)
    Turbidity: float = Field(..., example=3.0)

# 2) Cargamos el modelo serializado (Pipeline entrenado con imputer + XGBClassifier)

try:
    with open("models/best_model.pkl", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    raise RuntimeError("No se encontró el archivo 'models/best_model.pkl'. "
                       "Asegúrate de haber ejecutado la sección de optimización y "
                       "de que el pipeline esté guardado en esa ruta.")

# 3) Instanciamos la aplicación FastAPI
app = FastAPI(
    title="API de Predicción de Potabilidad de Agua",
    description="""
Servicio encargado de predecir si una muestra de agua es potable (1) o no (0),
a partir de lecturas de sensores químicos.  
- **GET /**: Descripción del servicio.  
- **POST /potabilidad/**: Recibe las 9 variables químicas y retorna `{"potabilidad": 0 }` o `{"potabilidad": 1}`.
""",
    version="1.0.0"
)

# 4) Ruta GET de bienvenida / descripción
@app.get("/")
def home():
    return {
        "mensaje": "Modelo de clasificación de potabilidad de agua usando XGBoost.",
        "endpoint_POST": "/potabilidad/",
        "input_JSON_ejemplo": {
            "ph": 10.3164,
            "Hardness": 217.2668,
            "Solids": 10676.5085,
            "Chloramines": 3.4455,
            "Sulfate": 397.7549,
            "Conductivity": 492.2065,
            "Organic_carbon": 12.8127,
            "Trihalomethanes": 72.2819,
            "Turbidity": 3.4073
        },
        "salida_ejemplo": {"potabilidad": 0}
    }

# 5) Ruta POST /potabilidad/ para predecir
@app.post("/potabilidad/")
def predecir_potabilidad(sample: WaterSample):
    # a) Convertimos el pydantic model a DataFrame de una fila
    data_dict = sample.dict()
    df = pd.DataFrame([data_dict])

    # b) Ejecutamos la predicción
    try:
        pred = model.predict(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al predecir: {e}")

    # c) model.predict devuelve un arreglo, tomamos el primer elemento (0 o 1)
    potabilidad = int(pred[0])

    return {"potabilidad": potabilidad}

# 6) Ejecutar el servidor con Uvicorn 
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
