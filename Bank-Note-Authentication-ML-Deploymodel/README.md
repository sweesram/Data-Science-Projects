# Bank Note Authentication API

A simple machine learning project that predicts whether a bank note is **genuine or fake**, based on features extracted from an image of the note. The trained model is served through a **FastAPI** web API.

## How it works

1. A `RandomForestClassifier` (from scikit-learn) is trained on the [Bank Note Authentication dataset](https://www.kaggle.com/datasets/ritesaluja/bank-note-authentication-uci-data), which contains statistical features extracted from wavelet-transformed images of real and forged bank notes.
2. The trained model is saved as `Classifier.pkl`.
3. `app.py` loads this saved model and exposes it through a REST API endpoint, so anyone can send note measurements and get a prediction back.

## Project structure

```
├── modelTrain.ipynb      # Notebook used to train and export the model
├── Classifier.pkl        # Saved (trained) ML model
├── Banknote.py            # Defines the input data format (schema)
├── app.py                 # FastAPI app that serves predictions
├── requirements.txt       # Python dependencies
└── pyproject.toml         # Project metadata (for uv/pip)
```

## Input features

Each bank note is described by 4 numeric features:

| Feature    | Description                              |
|------------|-------------------------------------------|
| variance   | Variance of the wavelet-transformed image |
| skewness   | Skewness of the wavelet-transformed image |
| curtosis   | Kurtosis of the wavelet-transformed image |
| entropy    | Entropy of the image                      |

## Getting started

### 1. Clone the repo
```bash
git clone https://github.com/<your-username>/ML-Deploy.git
cd ML-Deploy
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the API
```bash
python app.py
```
The server will start at `http://127.0.0.1:5001`.

### 4. Make a prediction
Send a POST request to `/predict`:

```bash
curl -X POST "http://127.0.0.1:5001/predict" \
  -H "Content-Type: application/json" \
  -d '{"varience": 2.3, "skewness": 4.1, "curtosis": -1.2, "entrophy": 0.5}'
```

**Response:**
```json
{
  "prediction": "Its a Bank note"
}
```

## Tech stack

- **Python**
- **scikit-learn** – model training (Random Forest)
- **FastAPI** – serving the model as an API
- **Uvicorn / Gunicorn** – running the server
- **Pandas / NumPy** – data handling

## Notes

- This project is for learning/demo purposes and shows a basic end-to-end ML deployment workflow: train → save model → serve via API.
