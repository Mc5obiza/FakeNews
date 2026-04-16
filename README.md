# Fake News Detection

This project detects whether a news article is real or fake.

It has three parts:

- `backend/` contains the FastAPI prediction API and the training script.
- `frontend/` contains the Streamlit UI.
- `data/` contains the training data files: `Fake.csv` and `True.csv`.

## How it works

The backend loads the dataset from `data/`, trains a machine learning pipeline, and saves the trained model as a `joblib` artifact in `backend/artifacts/`.

The FastAPI server then loads that saved model and exposes a `/predict` endpoint. The Streamlit frontend sends article details to that endpoint and displays the result.

## Run With Docker

1. Clone the repository:

```bash
git clone https://github.com/Mc5obiza/FakeNews.giy
cd Fake-news
```

2. Build and start the containers:

```bash
docker compose up --build
```

3. Open the apps:

- Streamlit frontend: http://localhost:8501
- FastAPI backend docs: http://localhost:8000/docs

## Local Development

If you want to run without Docker:

1. Install backend dependencies:

```bash
python -m pip install -r backend/requirements.txt
```

2. Train the model:

```bash
python backend/train_model.py
```

3. Start the API:

```bash
python backend/api.py
```

4. Install frontend dependencies:

```bash
python -m pip install -r frontend/requirements.txt
```

5. Start the Streamlit app:

```bash
streamlit run frontend/app.py
```

## Notes

- The frontend reads the API address from `API_BASE_URL` when running in Docker.
- The model is trained from the CSV files inside `data/` only.
