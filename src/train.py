import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import joblib
import os

def main():
    # Carregar dataset
    df = pd.read_csv("data/churn.csv")

    # Converter coluna TotalCharges para numérico (ela vem como string em alguns casos)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Definir alvo
    y = (df["Churn"] == "Yes").astype(int)
    X = df.drop(["Churn", "customerID"], axis=1)

    # Separar colunas categóricas e numéricas
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    # Pré-processamento
    preproc = ColumnTransformer([
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), num_cols),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="constant", fill_value="missing")),
            ("ohe", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols)
    ])

    # Pipeline com RandomForest
    pipe = Pipeline([
        ("pre", preproc),
        ("clf", RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        ))
    ])

    # Split treino/teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Treinar
    pipe.fit(X_train, y_train)

    # Avaliar
    y_proba = pipe.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    print("ROC AUC:", round(auc, 4))

    # Salvar modelo
    os.makedirs("models", exist_ok=True)
    joblib.dump(pipe, "models/churn_pipeline.pkl")
    print("✅ Modelo salvo em models/churn_pipeline.pkl")

if __name__ == "__main__":
    main()
