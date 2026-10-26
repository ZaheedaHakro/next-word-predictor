
import pickle
import numpy as np

from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


BASE_DIR = Path(__file__).resolve().parent

model = None
tokenizer = None
max_len = None
input_length = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer, max_len, input_length

    with open(BASE_DIR / "tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    with open(BASE_DIR / "max_len.pkl", "rb") as f:
        max_len = pickle.load(f)

    model = load_model(BASE_DIR / "lstm_model.h5", compile=False)

    # Use the input length expected by the trained model.
    shape = model.input_shape
    if isinstance(shape, list):
        raise RuntimeError("This app expects a single-input model.")

    input_length = shape[1]

    if input_length is None:
        raise RuntimeError("The model has a variable input length.")

    input_length = int(input_length)

    yield


app = FastAPI(
    title="LSTM Next Word Predictor",
    lifespan=lifespan
)

app.mount(
    "/static",
    StaticFiles(directory=BASE_DIR / "static"),
    name="static"
)

templates = Jinja2Templates(directory=BASE_DIR / "templates")


class PredictionRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=10)
    generate: bool = False
    num_words: int = Field(default=10, ge=1, le=30)


def predict_next_words(text, top_k=5):
    text = text.strip()

    if not text:
        raise ValueError("Please enter some text.")

    token_list = tokenizer.texts_to_sequences([text])[0]

    if not token_list:
        raise ValueError(
            "No known words found. Try words from your training dataset."
        )

    # Keep the most recent tokens if input is too long.
    token_list = token_list[-input_length:]

    padded = pad_sequences(
        [token_list],
        maxlen=input_length,
        padding="pre",
        truncating="pre"
    )

    probabilities = np.asarray(
        model.predict(padded, verbose=0)
    )[0]

    # Tokenizer word_index maps words to token IDs.
    index_word = tokenizer.index_word

    valid_ids = [
        int(i) for i in index_word
        if 0 <= int(i) < len(probabilities)
    ]

    if not valid_ids:
        raise ValueError("The tokenizer has no usable word mappings.")

    valid_ids.sort(
        key=lambda i: float(probabilities[i]),
        reverse=True
    )

    results = []

    for token_id in valid_ids:
        word = index_word.get(token_id)

        if not word:
            continue

        results.append({
            "word": word,
            "probability": round(
                float(probabilities[token_id]) * 100, 2
            )
        })

        if len(results) >= top_k:
            break

    if not results:
        raise ValueError("Could not generate a word prediction.")

    return results


def generate_text(text, num_words=10):
    generated = text.strip()

    for _ in range(num_words):
        try:
            predictions = predict_next_words(generated, top_k=1)
        except ValueError:
            break

        next_word = predictions[0]["word"]
        generated += " " + next_word

    return generated


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )


@app.get("/api/health")
async def health():
    return {
        "status": "ready",
        "model": "LSTM",
        "input_length": input_length,
        "max_len": max_len
    }


@app.post("/api/predict")
async def predict(request: PredictionRequest):
    try:
        predictions = predict_next_words(
            request.text,
            request.top_k
        )

        generated_text = None

        if request.generate:
            generated_text = generate_text(
                request.text,
                request.num_words
            )

        return {
            "input_text": request.text,
            "predictions": predictions,
            "generated_text": generated_text
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc)
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Prediction failed. Check model and tokenizer "
                "compatibility."
            )
        ) from exc
