import os
import sys
import time
import json
import shutil
import asyncio
import numpy as np
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional

import tensorflow as tf
from tensorflow import keras

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Ensure src package is found
sys.path.append(str(Path(__file__).parent))
from src.preprocessing import DataPreprocessor
from src.prediction import CharacterPredictor
from database import engine, Base, init_db, SessionLocal
from database_model import Prediction

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Suppress TF logging
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU') # Force CPU only


# ============================================================
# CONFIG — use Path consistently throughout
# ============================================================
MODEL_PATH = Path("./models/final_alphanumeric_model.h5")
CLASS_NAMES_PATH = Path("./models/class_names.npy")
UPLOAD_DIR = Path("./uploads")
RETRAIN_DATA_DIR = Path("./data/retrain")

UPLOAD_DIR.mkdir(exist_ok=True)
RETRAIN_DATA_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# APP STATE
# ============================================================
class AppState:
    def __init__(self):
        self.model = None
        self.predictor = None
        self.start_time = datetime.now()
        self.prediction_count = 0
        self.total_inference_time = 0

        # Retraining
        self.is_training = False
        self.training_status = "idle"
        self.last_training_time = None
        self.model_load_error = None
        self.training_epoch_metrics = []   # stores per-epoch metrics for SSE streaming
        self.training_total_epochs = 0


app_state = AppState()


# ============================================================
# LIFESPAN HANDLER
# — removed @app.on_event("startup") to avoid double-init conflict
# ============================================================
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ---- STARTUP ----
    print("\n=== INITIALIZING DATABASE ===")
    try:
        await init_db()
        print("Database initialized.\n")
    except Exception as e:
        print(f"ERROR initializing database: {e}\n")

    print("=== LOADING MODEL ON STARTUP ===")

    # Validate model files exist before attempting to load
    if not MODEL_PATH.exists():
        msg = f"Model file not found: {MODEL_PATH.resolve()}"
        print(f"ERROR: {msg}")
        app_state.model_load_error = msg
    elif not CLASS_NAMES_PATH.exists():
        msg = f"Class names file not found: {CLASS_NAMES_PATH.resolve()}"
        print(f"ERROR: {msg}")
        app_state.model_load_error = msg
    else:
        try:
            # Try loading directly first with legacy safe_mode to handle
            # older models saved with batch_shape / Keras 2 InputLayer configs
            try:
                model = tf.keras.models.load_model(
                    str(MODEL_PATH),
                    compile=False,          # skip optimizer restore — avoids config mismatches
                    safe_mode=False         # allow custom/legacy layer configs
                )
                model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
                print("Model loaded via direct tf.keras (legacy-safe mode).")
            except Exception as direct_err:
                print(f"Direct load failed ({direct_err}), trying custom_object_scope fallback...")
                # Fallback: patch InputLayer to accept batch_shape kwarg
                from tensorflow.keras.layers import InputLayer

                class LegacyInputLayer(InputLayer):
                    def __init__(self, *args, **kwargs):
                        kwargs.pop("batch_shape", None)
                        super().__init__(*args, **kwargs)

                with tf.keras.utils.custom_object_scope({"InputLayer": LegacyInputLayer}):
                    model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
                model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
                print("Model loaded via legacy InputLayer patch.")

            # Load class names separately and inject into predictor
            class_names = np.load(str(CLASS_NAMES_PATH), allow_pickle=True)

            # Build predictor and inject already-loaded model to avoid double-load
            predictor = CharacterPredictor(
                model_path=str(MODEL_PATH),
                class_names_path=str(CLASS_NAMES_PATH)
            )
           # predictor.model = model          # override with our successfully loaded model
            #predictor.class_names = class_names

            app_state.predictor = predictor
            app_state.model = model
            app_state.model_load_error = None
            print(f"Model loaded successfully from: {MODEL_PATH.resolve()}")
            print(f"Total classes: {len(class_names)}\n")

        except Exception as e:
            msg = str(e)
            app_state.model_load_error = msg
            print(f"ERROR loading model: {msg}")
            print("API running WITHOUT loaded model.\n")

    yield

    # ---- SHUTDOWN ----
    print("Shutting down API...")


# ============================================================
# INIT APP
# ============================================================
app = FastAPI(
    title="Handwritten Character Recognition API",
    description="FastAPI backend for handwritten character model",
    version="1.0.1",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True
)

app.mount("/static", StaticFiles(directory="static"), name="static")


# ============================================================
# Pydantic Response Models
# ============================================================
class TopPrediction(BaseModel):
    character: str
    confidence: float
    confidence_percent: float


class PredictionResponse(BaseModel):
    predicted_character: str
    confidence: float
    confidence_percent: float
    top_k_predictions: List[TopPrediction]
    inference_time_ms: float


class ModelStatus(BaseModel):
    model_loaded: bool
    model_path: str
    uptime_seconds: float
    uptime_formatted: str
    total_predictions: int
    average_inference_time_ms: float
    is_training: bool
    last_training_time: Optional[str]
    model_info: dict


class RetrainRequest(BaseModel):
    epochs: int = 5
    learning_rate: float = 0.0001


# ============================================================
# ROOT
# ============================================================
@app.get("/")
async def root():
    return FileResponse("static/index.html")


@app.get("/training")
async def training():
    return FileResponse("static/retrain.html")


@app.get("/dashboard")
async def dashboard():
    return FileResponse("static/dashboard.html")


# ============================================================
# PREDICT SINGLE IMAGE
# ============================================================
@app.post("/predict", response_model=PredictionResponse)
async def predict_character(file: UploadFile = File(...), true_label: Optional[str] = Form(None)):
    if app_state.predictor is None:
        # Surface the actual load error so it's easier to diagnose
        detail = "Model not loaded"
        if app_state.model_load_error:
            detail += f": {app_state.model_load_error}"
        raise HTTPException(status_code=503, detail=detail)

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail=f"Invalid file type: {file.content_type}")

    temp_path = UPLOAD_DIR / f"temp_{time.time()}_{file.filename}"

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        start = time.time()
        result = app_state.predictor.predict(str(temp_path), top_k=3)
        inference_ms = (time.time() - start) * 1000

        app_state.prediction_count += 1
        app_state.total_inference_time += inference_ms

        # Format top predictions
        top_preds = result.get("top_k_predictions", [])
        formatted_top_preds = [
            {
                "character": pred.get("character") or pred.get("label") or "Unknown",
                "confidence": pred.get("confidence", 0.0),
                "confidence_percent": pred.get("confidence", 0.0) * 100
            } for pred in top_preds
        ]

        # Save to database
        session = SessionLocal()
        try:
            new_pred = Prediction(
                image_name=file.filename,
                predicted_character=result.get("predicted_character", "Unknown"),
                true_label=true_label or "unknown",
                confidence=result.get("confidence", 0.0),
                inference_time_ms=inference_ms
            )
            session.add(new_pred)
            session.commit()
        finally:
            session.close()

        return {
            "predicted_character": result.get("predicted_character", "Unknown"),
            "confidence": result.get("confidence", 0.0),
            "confidence_percent": result.get("confidence", 0.0) * 100,
            "top_k_predictions": formatted_top_preds,
            "inference_time_ms": round(inference_ms, 2)
        }

    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


# ============================================================
# BATCH PREDICTION
# ============================================================
@app.post("/predict_batch")
async def predict_batch(
    files: List[UploadFile] = File(...),
    true_labels: Optional[List[str]] = Form(None)
):
    if app_state.predictor is None:
        detail = "Model not loaded"
        if app_state.model_load_error:
            detail += f": {app_state.model_load_error}"
        raise HTTPException(status_code=503, detail=detail)

    if len(files) > 100:
        raise HTTPException(status_code=400, detail="Max 100 images allowed")

    saved_paths = []
    try:
        for f in files:
            if f.content_type and f.content_type.startswith("image/"):
                p = UPLOAD_DIR / f"batch_{time.time()}_{f.filename}"
                with open(p, "wb") as buffer:
                    shutil.copyfileobj(f.file, buffer)
                saved_paths.append(p)

        if not saved_paths:
            raise HTTPException(status_code=400, detail="No valid image files provided")

        start = time.time()
        preds = app_state.predictor.predict_batch([str(p) for p in saved_paths])
        total_ms = (time.time() - start) * 1000

        app_state.prediction_count += len(preds)
        app_state.total_inference_time += total_ms

        session = SessionLocal()
        try:
            for i, result in enumerate(preds):
                label = true_labels[i] if true_labels and i < len(true_labels) else "unknown"
                new_pred = Prediction(
                    image_name=result.get("image_name", f"img_{i}"),
                    predicted_character=result.get("predicted_character", "Unknown"),
                    true_label=label,
                    confidence=result.get("confidence", 0.0),
                    inference_time_ms=total_ms / len(preds)
                )
                session.add(new_pred)
            session.commit()
        finally:
            session.close()

        return {
            "total_images": len(preds),
            "total_time_ms": round(total_ms, 2),
            "average_time_per_image_ms": round(total_ms / len(preds), 2) if preds else 0,
            "predictions": preds
        }

    finally:
        for p in saved_paths:
            if p.exists():
                p.unlink(missing_ok=True)


# ============================================================
# UPLOAD TRAINING DATA
# ============================================================
@app.post("/upload_data")
async def upload_data(files: List[UploadFile] = File(...), label: str = Form(...)):
    if not label:
        raise HTTPException(status_code=400, detail="Label is required")

    label_dir = RETRAIN_DATA_DIR / label
    label_dir.mkdir(parents=True, exist_ok=True)
    saved = 0

    for f in files:
        if f.content_type and f.content_type.startswith("image/"):
            file_path = label_dir / f"{time.time()}_{f.filename}"
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(f.file, buffer)
            saved += 1

    return {"status": "success", "label": label, "files_saved": saved}


# ============================================================
# RETRAINING
# ============================================================
class EpochMetricsCallback(tf.keras.callbacks.Callback):
    """Pushes per-epoch metrics into app_state for SSE streaming."""
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        app_state.training_epoch_metrics.append({
            "epoch": epoch + 1,
            "loss": round(float(logs.get("loss", 0)), 4),
            "accuracy": round(float(logs.get("accuracy", 0)), 4),
            "val_loss": round(float(logs.get("val_loss", 0)), 4) if "val_loss" in logs else None,
            "val_accuracy": round(float(logs.get("val_accuracy", 0)), 4) if "val_accuracy" in logs else None,
            "timestamp": datetime.now().isoformat(),
        })
        app_state.training_status = (
            f"training - epoch {epoch + 1}/{app_state.training_total_epochs} "
            f"| loss: {logs.get('loss', 0):.4f} | acc: {logs.get('accuracy', 0):.4f}"
        )


async def retrain_model_task(epochs: int, learning_rate: float):
    try:
        print("Retraining started...")
        app_state.training_status = "preprocessing"
        app_state.training_epoch_metrics = []
        app_state.training_total_epochs = epochs

        pre = DataPreprocessor()
        X_new, y_new = pre.process_new_data_for_training(RETRAIN_DATA_DIR)
        if len(X_new) == 0:
            app_state.training_status = "failed - no data"
            app_state.is_training = False
            return

        num_classes = len(app_state.predictor.class_names)
        y_new_enc = keras.utils.to_categorical(y_new, num_classes)

        app_state.training_status = "training"
        model = keras.models.load_model(str(MODEL_PATH), compile=False, safe_mode=False)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )
        model.fit(
            X_new, y_new_enc,
            epochs=epochs,
            verbose=1,
            callbacks=[EpochMetricsCallback()]
        )

        retrained_path = MODEL_PATH.parent / MODEL_PATH.name.replace(".h5", "_retrained.h5")
        model.save(str(retrained_path))
        app_state.predictor.load_model(str(retrained_path))
        app_state.model = app_state.predictor.model

        app_state.training_status = "completed"
        app_state.last_training_time = datetime.now().isoformat()
        print("Retraining completed.")

    except Exception as e:
        app_state.training_status = f"failed: {e}"
        print("Retraining failed:", e)
    finally:
        app_state.is_training = False


@app.post("/retrain")
async def trigger_retrain(request: RetrainRequest = None):
    if app_state.is_training:
        return {"status": "already_training"}

    # FIX: safe check — iterdir() on empty dir raises StopIteration
    has_data = any(RETRAIN_DATA_DIR.iterdir()) if RETRAIN_DATA_DIR.exists() else False
    if not has_data:
        raise HTTPException(status_code=400, detail="No retraining data found")

    epochs = request.epochs if request else 5
    lr = request.learning_rate if request else 0.0001

    asyncio.create_task(retrain_model_task(epochs, lr))
    app_state.is_training = True
    app_state.training_status = "started"
    return {"status": "started", "epochs": epochs, "learning_rate": lr}


# ============================================================
# STATUS / METRICS / HEALTH
# ============================================================
@app.get("/metrics")
async def get_metrics():
    if app_state.model is None:
        return {
            "total_predictions": 0,
            "average_inference_time_ms": 0,
            "uptime_seconds": (datetime.now() - app_state.start_time).total_seconds(),
            "model_accuracy": None,
            "model_precision": None,
            "model_recall": None,
            "model_f1_score": None,
            "message": f"Model not loaded: {app_state.model_load_error or 'unknown reason'}"
        }

    X_val, y_val = [], []
    try:
        from PIL import Image
        class_dirs = sorted([d for d in RETRAIN_DATA_DIR.iterdir() if d.is_dir()])
        for idx, class_dir in enumerate(class_dirs):
            for f in class_dir.iterdir():
                if f.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                    img = Image.open(f).convert("L").resize((64, 64))
                    X_val.append(np.array(img, dtype=np.float32)[..., np.newaxis] / 255.0)
                    y_val.append(idx)
        X_val = np.array(X_val)
        y_val = np.array(y_val)
    except Exception as e:
        print("Error loading validation data:", e)
        return {"total_predictions": 0, "message": f"Validation data error: {e}"}

    if len(X_val) == 0:
        return {"total_predictions": 0, "message": "No validation data available in retrain directory"}

    if not hasattr(app_state.model, "optimizer") or app_state.model.optimizer is None:
        app_state.model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    start = time.time()
    preds = app_state.model.predict(X_val, verbose=0)
    inference_ms = (time.time() - start) * 1000
    y_pred = np.argmax(preds, axis=1)

    try:
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, average="macro", zero_division=0)
        recall = recall_score(y_val, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
    except Exception as e:
        print("Metrics calculation failed:", e)
        accuracy = precision = recall = f1 = None

    return {
        "total_predictions": len(y_val),
        "average_inference_time_ms": round(inference_ms / len(y_val), 2) if len(y_val) else 0,
        "uptime_seconds": (datetime.now() - app_state.start_time).total_seconds(),
        "model_accuracy": float(accuracy) if accuracy is not None else None,
        "model_precision": float(precision) if precision is not None else None,
        "model_recall": float(recall) if recall is not None else None,
        "model_f1_score": float(f1) if f1 is not None else None,
        "message": "Metrics computed successfully"
    }


@app.get("/training_status")
async def training_status():
    return {
        "is_training": app_state.is_training,
        "status": app_state.training_status,
        "last_training_time": app_state.last_training_time,
        "epoch_metrics": app_state.training_epoch_metrics,
        "total_epochs": app_state.training_total_epochs,
    }


@app.get("/training_stream")
async def training_stream():
    """SSE endpoint — streams epoch metrics in real time to the dashboard."""
    from fastapi.responses import StreamingResponse

    async def event_generator():
        last_sent = 0
        while True:
            current_metrics = app_state.training_epoch_metrics
            # Send any new epochs since last check
            if len(current_metrics) > last_sent:
                for metric in current_metrics[last_sent:]:
                    yield f"data: {json.dumps(metric)}\n\n"
                last_sent = len(current_metrics)

            # Send a heartbeat every cycle so the connection stays alive
            yield f"data: {json.dumps({'heartbeat': True, 'status': app_state.training_status, 'is_training': app_state.is_training})}\n\n"

            # If training finished and all epochs sent, close stream
            if not app_state.is_training and last_sent >= app_state.training_total_epochs and app_state.training_total_epochs > 0:
                yield f"data: {json.dumps({'done': True, 'final_metrics': app_state.training_epoch_metrics})}\n\n"
                break

            await asyncio.sleep(1)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # prevents nginx from buffering SSE
        }
    )


@app.get("/health")
async def health():
    return {
        "status": "healthy" if app_state.model else "degraded",
        "model_loaded": app_state.model is not None,
        "model_load_error": app_state.model_load_error,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/model_status")
async def model_status():
    """Frontend-facing model status endpoint."""
    uptime_seconds = (datetime.now() - app_state.start_time).total_seconds()
    avg_inference = (
        round(app_state.total_inference_time / app_state.prediction_count, 2)
        if app_state.prediction_count > 0 else 0
    )
    model_info = {}
    if app_state.model is not None:
        try:
            model_info = {
                "name": app_state.model.name,
                "total_params": app_state.model.count_params(),
                "num_layers": len(app_state.model.layers),
                "input_shape": str(app_state.model.input_shape),
                "output_shape": str(app_state.model.output_shape),
            }
        except Exception:
            model_info = {"note": "Could not read model details"}

    return {
        "model_loaded": app_state.model is not None,
        "model_path": str(MODEL_PATH),
        "model_load_error": app_state.model_load_error,
        "status": "healthy" if app_state.model else "degraded",
        "uptime_seconds": uptime_seconds,
        "uptime_formatted": format_uptime(uptime_seconds),
        "total_predictions": app_state.prediction_count,
        "average_inference_time_ms": avg_inference,
        "is_training": app_state.is_training,
        "last_training_time": app_state.last_training_time,
        "model_info": model_info,
        "timestamp": datetime.now().isoformat(),
    }


# ============================================================
# UPTIME FORMATTER
# ============================================================
def format_uptime(seconds: float):
    days = int(seconds // 86400)
    hours = int((seconds % 86400) // 3600)
    minutes = int((seconds % 3600) // 60)
    sec = int(seconds % 60)

    parts = []
    if days: parts.append(f"{days}d")
    if hours: parts.append(f"{hours}h")
    if minutes: parts.append(f"{minutes}m")
    parts.append(f"{sec}s")
    return " ".join(parts)


# ============================================================
# RUN DIRECTLY
# ============================================================
"""if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)"""