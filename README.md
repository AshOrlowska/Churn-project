# 🔮 Churn Predictor

Projeto em Python para prever o cancelamento de clientes usando Machine Learning.

## 🚀 Funcionalidades
- Treinamento de modelo de churn (RandomForest)
- Dashboard interativo (Streamlit)
- API REST (FastAPI)

## 📂 Estrutura
```bash

churn-predictor/
│
├── data/                # datasets
│   └── churn.csv
├── models/              # modelos treinados
│   └── churn_pipeline.pkl
├── notebooks/
│   └── eda.ipynb        # exploração de dados
├── src/
│   ├── train.py         # treino do modelo
│   ├── predict.py       # função de predição
│   ├── streamlit_app.py # app interativo
│   └── api.py           # API FastAPI
├── requirements.txt
└── README.md
 

## ▶️ Como rodar

# instalar dependências
pip install -r requirements.txt

# treinar modelo
python src/train.py

# rodar streamlit
streamlit run src/streamlit_app.py

# rodar API
uvicorn src.api:app --reload --port 8000


