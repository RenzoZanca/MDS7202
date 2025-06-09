# lab9/dags/hiring_dynamic_functions.py

import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score



def create_folders(**kwargs):
    """
    Crea la carpeta base con nombre ds (YYYY-MM-DD) y las subcarpetas:
      - raw
      - preprocessed
      - splits
      - models
    """
    # 1. Extraemos la fecha de ejecución (ds) del contexto de Airflow
    execution_date = kwargs.get('ds')
    
    if execution_date is None:
        raise ValueError("La clave 'ds' no está en kwargs; asegúrate de usar provide_context=True")

    # 2. Definimos el directorio base
    base_path = os.path.join(os.getcwd(), execution_date)

    # 3. Creamos la carpeta base
    os.makedirs(base_path, exist_ok=True)
    print(f"[create_folders] Base folder created: {base_path}")

    # 4. Creamos las subcarpetas necesarias
    for sub in ['raw', 'preprocessed', 'splits', 'models']:
        path = os.path.join(base_path, sub)
        os.makedirs(path, exist_ok=True)
        print(f"[create_folders] Subfolder created: {path}")

def load_ands_merge(**kwargs):
    """
    Lee data_1.csv y data_2.csv desde raw/ (si existen), las concatena
    y guarda un solo archivo combined.csv en preprocessed/.
    """
    ds = kwargs.get('ds')
    if ds is None:
        raise ValueError("Se requiere el parámetro 'ds' en kwargs")

    base = os.path.join(os.getcwd(), ds)
    raw_dir = os.path.join(base, 'raw')
    preproc_dir = os.path.join(base, 'preprocessed')

    # Archivos posibles
    candidates = ['data_1.csv', 'data_2.csv']
    present = []
    for fname in candidates:
        path = os.path.join(raw_dir, fname)
        if os.path.isfile(path):
            present.append(path)
            print(f"[load_ands_merge] Encontrado: {fname}")
        else:
            print(f"[load_ands_merge] No existe (se omite): {fname}")

    if not present:
        raise FileNotFoundError(f"No se encontraron archivos en {raw_dir}")

    # Leer y concatenar
    dfs = [pd.read_csv(p) for p in present]
    df_all = pd.concat(dfs, ignore_index=True)
    print(f"[load_ands_merge] Concatenadas {len(dfs)} tablas, filas totales: {len(df_all)}")

    # Guardar resultado
    out_path = os.path.join(preproc_dir, 'combined.csv')
    df_all.to_csv(out_path, index=False)
    print(f"[load_ands_merge] Guardado combined.csv en {out_path}")


def split_data(**kwargs):
    """
    Lee combined.csv de preprocessed/, hace hold-out 80/20 (estratificado)
    sobre HiringDecision, y guarda train.csv y test.csv en splits/.
    """
    ds = kwargs.get('ds')
    if ds is None:
        raise ValueError("Se requiere 'ds' en kwargs para split_data")

    base_dir = os.path.join(os.getcwd(), ds)
    preproc_path = os.path.join(base_dir, 'preprocessed', 'combined.csv')
    splits_dir  = os.path.join(base_dir, 'splits')

    # Verificar existencia
    if not os.path.isfile(preproc_path):
        raise FileNotFoundError(f"No existe combined.csv en {preproc_path}")

    # Leer datos
    df = pd.read_csv(preproc_path)
    X = df.drop(columns=['HiringDecision'])
    y = df['HiringDecision']

    # Hold-out estratificado
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Reconstruir DataFrames
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df  = pd.concat([X_test,  y_test],  axis=1)

    # Guardar
    train_path = os.path.join(splits_dir, 'train.csv')
    test_path  = os.path.join(splits_dir, 'test.csv')
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path,   index=False)

    print(f"[split_data] Guardado train ({len(train_df)} filas) en {train_path}")
    print(f"[split_data] Guardado test  ({len(test_df)} filas) en {test_path}")

def train_model(model, model_name, **kwargs):
    """
    Entrena un pipeline sobre train.csv y serializa el resultado.
    
    model:        instancia del clasificador (e.g. RandomForestClassifier())
    model_name:   cadena corta para nombrar el archivo (.joblib)
    kwargs['ds']: fecha de ejecución en formato 'YYYY-MM-DD'
    """
    ds = kwargs.get('ds')
    if ds is None:
        raise ValueError("Falta el parámetro 'ds' en kwargs para train_model")
    
    base_dir   = os.path.join(os.getcwd(), ds)
    splits_dir = os.path.join(base_dir, 'splits')
    models_dir = os.path.join(base_dir, 'models')
    
    # 1) Leer train.csv
    train_path = os.path.join(splits_dir, 'train.csv')
    if not os.path.isfile(train_path):
        raise FileNotFoundError(f"No hallé {train_path}")
    df = pd.read_csv(train_path)
    
    X_train = df.drop(columns=['HiringDecision'])
    y_train = df['HiringDecision']
    
    # 2) Definir las columnas
    numeric_features     = ['Age', 'ExperienceYears', 'DistanceFromCompany',
                            'InterviewScore', 'SkillScore', 'PersonalityScore']
    categorical_features = ['Gender', 'EducationLevel', 'PreviousCompanies',
                            'RecruitmentStrategy']
    
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])
    
    # 3) Montar el pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    # 4) Entrenar
    pipeline.fit(X_train, y_train)
    
    # 5) Guardar el pipeline entrenado
    out_file = os.path.join(models_dir, f"{model_name}.joblib")
    joblib.dump(pipeline, out_file)
    print(f"[train_model] Modelo '{model_name}' entrenado y guardado en {out_file}")

def evaluate_models(**kwargs):
    """
    Lee todos los pipelines (.joblib) en models/, evalúa su accuracy
    sobre el test set y selecciona el mejor. Guarda ese mejor modelo
    como best_model.joblib en la misma carpeta.
    """
    ds = kwargs.get('ds')
    if ds is None:
        raise ValueError("Se requiere 'ds' en kwargs para evaluate_models")

    base_dir   = os.path.join(os.getcwd(), ds)
    splits_dir = os.path.join(base_dir, 'splits')
    models_dir = os.path.join(base_dir, 'models')

    # Cargar test set
    test_path = os.path.join(splits_dir, 'test.csv')
    if not os.path.isfile(test_path):
        raise FileNotFoundError(f"No existe test.csv en {test_path}")
    df_test = pd.read_csv(test_path)
    X_test  = df_test.drop(columns=['HiringDecision'])
    y_test  = df_test['HiringDecision']

    # Buscar todos los modelos
    files = [
        f for f in os.listdir(models_dir)
        if f.endswith('.joblib')
    ]
    if not files:
        raise FileNotFoundError(f"No se encontraron pipelines en {models_dir}")

    best_acc   = -1.0
    best_model = None
    best_name  = None

    # Evaluar cada modelo
    for fname in files:
        path = os.path.join(models_dir, fname)
        pipeline = joblib.load(path)
        preds    = pipeline.predict(X_test)
        acc      = accuracy_score(y_test, preds)
        print(f"[evaluate_models] Modelo {fname:<20} → Accuracy: {acc:.4f}")

        if acc > best_acc:
            best_acc   = acc
            best_model = pipeline
            best_name  = fname

    # Reportar y guardar el mejor
    print(f"[evaluate_models] Mejor modelo: {best_name} con accuracy {best_acc:.4f}")
    best_path = os.path.join(models_dir, 'best_model.joblib')
    joblib.dump(best_model, best_path)
    print(f"[evaluate_models] Best pipeline guardado en: {best_path}")
