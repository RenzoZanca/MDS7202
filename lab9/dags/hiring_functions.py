import os
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import gradio as gr

def create_folders(**kwargs):
    # Obtener la fecha de ejecución del contexto del DAG
    execution_date = kwargs['ds']
    base_path = os.path.join(os.getcwd(), execution_date)

    # Crear la carpeta base y subcarpetas
    os.makedirs(os.path.join(base_path, 'raw'), exist_ok=True)
    os.makedirs(os.path.join(base_path, 'splits'), exist_ok=True)
    os.makedirs(os.path.join(base_path, 'models'), exist_ok=True)


def split_data(**kwargs):
    execution_date = kwargs['ds']
    base_path = os.path.join(os.getcwd(), execution_date)

    # Leer el archivo de datos
    data_path = os.path.join(base_path, 'raw', 'data_1.csv')
    df = pd.read_csv(data_path)

    X = df.drop(columns=['HiringDecision'])
    y = df['HiringDecision']

    # Hold-out con 20% test, misma proporción de clases
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Reconstruir los dataframes para guardar
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    # Guardar en carpeta splits
    train_df.to_csv(os.path.join(base_path, 'splits', 'train.csv'), index=False)
    test_df.to_csv(os.path.join(base_path, 'splits', 'test.csv'), index=False)


def preprocess_and_train(**kwargs):
    execution_date = kwargs['ds']
    base_path = os.path.join(os.getcwd(), execution_date)

    # Leer data
    train_df = pd.read_csv(os.path.join(base_path, 'splits', 'train.csv'))
    test_df = pd.read_csv(os.path.join(base_path, 'splits', 'test.csv'))
    X_train = train_df.drop(columns=['HiringDecision'])
    y_train = train_df['HiringDecision']
    X_test = test_df.drop(columns=['HiringDecision'])
    y_test = test_df['HiringDecision']

    # Features
    numeric_features = [
        'Age', 'ExperienceYears', 
        'DistanceFromCompany', 'InterviewScore',
        'SkillScore', 'PersonalityScore'
    ]
    categorical_features = ['Gender', 'EducationLevel', 'RecruitmentStrategy', 'PreviousCompanies']

    # Preprocesamiento
    # nota: según el reporte, la data no posee valores nulos, no tiene outliers claros, y las variables categóricas no tienen cardinalidad alta.
    # Se aplica one-hot encoding a las categóricas y escalado a las numéricas.
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numeric_features), # no tiene mucho peso, pero por si cambia el modelo en el futuro
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

    # Pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced'))
    ])

    # Entrenar y evaluar
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Accuracy en test: {acc:.4f}")
    print(f"F1-score clase positiva (contratado): {f1:.4f}")

    # Guardar modelo
    model_path = os.path.join(base_path, 'models', 'rf_pipeline.joblib')
    joblib.dump(pipeline, model_path)


def predict(file, model_path):

    pipeline = joblib.load(model_path)
    input_data = pd.read_json(file)
    predictions = pipeline.predict(input_data)
    print(f'La prediccion es: {predictions}')
    labels = ["No contratado" if pred == 0 else "Contratado" for pred in predictions]

    return {'Predicción': labels[0]}


def gradio_interface(**kwargs):
    execution_date = kwargs['ds']
    model_path = os.path.join(os.getcwd(), execution_date, 'models', 'rf_pipeline.joblib') 

    interface = gr.Interface(
        fn=lambda file: predict(file, model_path),
        inputs=gr.File(label="Sube un archivo JSON"),
        outputs="json",
        title="Hiring Decision Prediction",
        description="Sube un archivo JSON con las características de entrada para predecir si Vale será contratada o no."
    )
    interface.launch(share=True)
    

