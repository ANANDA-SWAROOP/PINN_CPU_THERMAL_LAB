"""Local inference service for the v5 coordinate-conditioned CPU thermal PINN.

Expected files (exported by the v5 Colab notebook):
  assets/pinn_simulator_bundle.pt
  assets/recorded_cpu_dataset.npz   (optional, for actual-vs-predicted comparison)

Run: uvicorn model_service:app --host 127.0.0.1 --port 8000
"""
from __future__ import annotations
import math
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = Path(os.getenv("PINN_ASSETS_DIR", str(BASE_DIR / "assets/cpu_thermal_simulator_assets")))
MODEL_PATH = Path(os.getenv("PINN_MODEL_PATH", str(ASSETS_DIR / "pinn_simulator_bundle.pt")))
DATASET_PATH = Path(os.getenv("PINN_DATASET_PATH", str(ASSETS_DIR / "recorded_cpu_dataset.npz")))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FOURIER_FREQUENCIES = (1, 2, 4, 8, 16)


def encode_inputs(inputs: torch.Tensor) -> torch.Tensor:
    x, y = inputs[:, 0:1], inputs[:, 1:2]
    parts = [inputs[:, 0:3]]
    for freq in FOURIER_FREQUENCIES:
        parts.extend([
            torch.sin(2 * math.pi * freq * x),
            torch.cos(2 * math.pi * freq * x),
            torch.sin(2 * math.pi * freq * y),
            torch.cos(2 * math.pi * freq * y),
        ])
    parts.append(inputs[:, 3:])
    return torch.cat(parts, dim=1)


class ThermalField(nn.Module):
    def __init__(self, feature_dim: int, width: int = 96, depth: int = 4):
        super().__init__()
        layers = []
        in_dim = feature_dim + 3 + 4 * len(FOURIER_FREQUENCIES)
        for _ in range(depth):
            layers.extend([nn.Linear(in_dim, width), nn.Tanh()])
            in_dim = width
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, raw_inputs: torch.Tensor) -> torch.Tensor:
        return self.net(encode_inputs(raw_inputs))


class PredictRequest(BaseModel):
    features: list[Optional[float]] = Field(..., description="Raw 80-dimensional feature vector in training column order")
    tau: float = Field(..., ge=0.0, le=1.0, description="Normalized sample-order coordinate in [0, 1]")


app = FastAPI(title="CPU Thermal PINN Inference", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"], allow_headers=["*"],
)

model: Optional[ThermalField] = None
bundle: Optional[dict] = None
recorded_features: Optional[np.ndarray] = None
recorded_maps: Optional[np.ndarray] = None
recorded_splits: list[str] = []
load_error: Optional[str] = None


def load_assets() -> None:
    global model, bundle, recorded_features, recorded_maps, recorded_splits, load_error
    load_error = None
    if MODEL_PATH.exists():
        try:
            try:
                loaded = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
            except TypeError:
                loaded = torch.load(MODEL_PATH, map_location="cpu")
            feature_dim = int(loaded.get("feature_dim", len(loaded["feature_mean"])))
            width = int(loaded.get("hidden_width", 96))
            depth = int(loaded.get("hidden_depth", 4))
            candidate = ThermalField(feature_dim, width, depth)
            state = loaded.get("state_dict", loaded.get("model_state"))
            if state is None:
                raise ValueError("Checkpoint is missing state_dict/model_state.")
            candidate.load_state_dict(state)
            candidate.to(DEVICE).eval()
            loaded["feature_dim"] = feature_dim
            loaded["height"] = int(loaded.get("height", 177))
            loaded["width"] = int(loaded.get("width", 166))
            loaded["n_frames"] = int(loaded.get("n_frames", 100))
            loaded["hidden_width"] = width
            loaded["hidden_depth"] = depth
            loaded["feature_mean"] = np.asarray(loaded["feature_mean"], dtype=np.float32).reshape(-1)
            loaded["feature_std"] = np.asarray(loaded["feature_std"], dtype=np.float32).reshape(-1)
            loaded["temperature_mean_c"] = float(loaded["temperature_mean_c"])
            loaded["temperature_std_c"] = float(loaded["temperature_std_c"])
            if len(loaded["feature_mean"]) != feature_dim or len(loaded["feature_std"]) != feature_dim:
                raise ValueError("Checkpoint feature scaler length does not match feature_dim.")
            bundle, model = loaded, candidate
            print(f"Loaded {MODEL_PATH} on {DEVICE}; input feature dimension={feature_dim}")
        except Exception as exc:
            model, bundle = None, None
            load_error = f"Could not load model checkpoint: {exc}"
            print(load_error)
    else:
        load_error = f"Model bundle not found at {MODEL_PATH}. Export it from Colab and place it in assets/."
        print(load_error)

    if DATASET_PATH.exists():
        try:
            data = np.load(DATASET_PATH, allow_pickle=False)
            raw_x = np.asarray(data["features_raw"], dtype=np.float32)
            maps = np.asarray(data["temperature_maps_c"], dtype=np.float32)
            if raw_x.ndim != 2 or maps.ndim != 3 or raw_x.shape[0] != maps.shape[0]:
                raise ValueError("recorded_cpu_dataset.npz requires features_raw [frames, features] and temperature_maps_c [frames, height, width].")
            if bundle is not None and raw_x.shape[1] != bundle["feature_dim"]:
                raise ValueError("Recorded dataset feature dimension does not match model bundle.")
            recorded_features, recorded_maps = raw_x, maps
            split_labels = np.full(raw_x.shape[0], "unknown", dtype=object)
            if "train_frame_ids" in data:
                split_labels[np.asarray(data["train_frame_ids"], dtype=np.int64)] = "train"
            if "val_frame_ids" in data:
                split_labels[np.asarray(data["val_frame_ids"], dtype=np.int64)] = "validation"
            if "test_frame_ids" in data:
                split_labels[np.asarray(data["test_frame_ids"], dtype=np.int64)] = "test"
            recorded_splits = split_labels.tolist()
            print(f"Loaded recorded data: {raw_x.shape[0]} frames, maps={maps.shape[1:]}")
        except Exception as exc:
            recorded_features, recorded_maps, recorded_splits = None, None, []
            print(f"Could not load recorded CPU series: {exc}")
    else:
        recorded_features, recorded_maps = None, None


load_assets()


def _model_inputs(features_raw: np.ndarray, tau: float) -> np.ndarray:
    assert bundle is not None
    raw = np.asarray(features_raw, dtype=np.float32).reshape(-1)
    if raw.size != bundle["feature_dim"]:
        raise ValueError(f"Expected {bundle['feature_dim']} features, received {raw.size}.")
    mean = bundle["feature_mean"]
    std = bundle["feature_std"]
    filled = np.where(np.isfinite(raw), raw, mean).astype(np.float32)
    scaled = ((filled - mean) / std).astype(np.float32)
    h, w = bundle["height"], bundle["width"]
    rows, cols = np.indices((h, w), dtype=np.float32)
    x = cols.reshape(-1) / max(w - 1, 1)
    y = rows.reshape(-1) / max(h - 1, 1)
    t = np.full(h * w, float(tau), dtype=np.float32)
    scaled_rows = np.broadcast_to(scaled, (h * w, len(scaled)))
    return np.column_stack((x, y, t, scaled_rows)).astype(np.float32)


def predict_map(features_raw: np.ndarray, tau: float) -> np.ndarray:
    if model is None or bundle is None:
        raise HTTPException(status_code=503, detail=load_error or "Model is not loaded.")
    inputs = _model_inputs(features_raw, tau)
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(inputs), 8192):
            xb = torch.from_numpy(inputs[start:start + 8192]).to(DEVICE)
            yp = model(xb).squeeze(1).detach().cpu().numpy()
            predictions.append(yp)
    pred_std = np.concatenate(predictions)
    temp_c = pred_std * bundle["temperature_std_c"] + bundle["temperature_mean_c"]
    return temp_c.reshape(bundle["height"], bundle["width"]).astype(np.float32)


def _map_summary(values: np.ndarray) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(arr)
    if not valid.any():
        return {"mean_c": None, "peak_c": None, "p95_c": None}
    vals = arr[valid]
    return {"mean_c": round(float(np.mean(vals)), 3),
            "peak_c": round(float(np.max(vals)), 3),
            "p95_c": round(float(np.percentile(vals, 95)), 3)}


def _comparison_metrics(gt: np.ndarray, pred: np.ndarray) -> dict:
    valid = np.isfinite(gt) & np.isfinite(pred)
    if not valid.any():
        return {}
    a, b = gt[valid].astype(np.float64), pred[valid].astype(np.float64)
    err = b - a
    gt_flat = np.where(np.isfinite(gt), gt, -np.inf).reshape(-1)
    pr_flat = np.where(np.isfinite(pred), pred, -np.inf).reshape(-1)
    gy, gx = np.unravel_index(int(np.argmax(gt_flat)), gt.shape)
    py, px = np.unravel_index(int(np.argmax(pr_flat)), pred.shape)
    ag = np.isfinite(gt) & (gt >= np.nanpercentile(gt, 95))
    bg = np.isfinite(pred) & (pred >= np.nanpercentile(pred, 95))
    union = np.count_nonzero(ag | bg)
    iou = np.count_nonzero(ag & bg) / union if union else 1.0
    return {"mae_c": round(float(np.mean(np.abs(err))), 4),
            "rmse_c": round(float(np.sqrt(np.mean(err ** 2))), 4),
            "peak_location_error_px": round(float(math.hypot(py - gy, px - gx)), 3),
            "top5_iou": round(float(iou), 4)}


def _rounded_flat(a: np.ndarray) -> list[float | None]:
    flat = np.asarray(a, dtype=np.float32).reshape(-1)
    return [round(float(v), 2) if np.isfinite(v) else None for v in flat]


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "dataset_loaded": recorded_features is not None and recorded_maps is not None,
        "model_type": bundle.get("model_type", "pinn") if bundle else None,
        "feature_dim": bundle.get("feature_dim") if bundle else None,
        "height": bundle.get("height") if bundle else None,
        "width": bundle.get("width") if bundle else None,
        "n_recorded_frames": int(len(recorded_features)) if recorded_features is not None else 0,
        "device": str(DEVICE),
        "message": load_error if model is None else "Model ready",
    }


@app.get("/recorded-series")
def recorded_series() -> dict:
    if recorded_features is None or recorded_maps is None:
        raise HTTPException(status_code=404, detail="Recorded dataset assets are not installed. Export the NPZ from the notebook.")
    summary = []
    for i, m in enumerate(recorded_maps):
        s = _map_summary(m)
        summary.append({"frame": i, "split": recorded_splits[i] if i < len(recorded_splits) else "unknown", **s})
    return {"dataset_name": "CPU i5 · 7zip recording", "n_frames": int(len(recorded_features)),
            "height": int(recorded_maps.shape[1]), "width": int(recorded_maps.shape[2]),
            "time_semantics": "normalized sample order, not physical seconds", "summary": summary}


@app.post("/predict")
def predict(request: PredictRequest) -> dict:
    if model is None or bundle is None:
        raise HTTPException(status_code=503, detail=load_error or "Model is not loaded.")
    if len(request.features) != bundle["feature_dim"]:
        raise HTTPException(status_code=422, detail=f"Expected exactly {bundle['feature_dim']} raw features.")
    raw = np.asarray([np.nan if v is None else float(v) for v in request.features], dtype=np.float32)
    pred = predict_map(raw, request.tau)
    return {"frame": None, "tau": float(request.tau), "height": int(pred.shape[0]), "width": int(pred.shape[1]),
            "prediction_c": _rounded_flat(pred), "prediction_summary": _map_summary(pred), "ground_truth_c": None, "metrics": None}


@app.get("/predict-recorded/{frame_index}")
def predict_recorded(frame_index: int) -> dict:
    if recorded_features is None or recorded_maps is None:
        raise HTTPException(status_code=404, detail="Recorded dataset assets are not installed.")
    if model is None or bundle is None:
        raise HTTPException(status_code=503, detail=load_error or "Model is not loaded.")
    if not 0 <= frame_index < len(recorded_features):
        raise HTTPException(status_code=404, detail="Frame index outside recorded dataset.")
    tau = frame_index / max(len(recorded_features) - 1, 1)
    pred = predict_map(recorded_features[frame_index], tau)
    gt = recorded_maps[frame_index]
    metrics = _comparison_metrics(gt, pred)
    return {"frame": frame_index, "split": recorded_splits[frame_index] if frame_index < len(recorded_splits) else "unknown", "tau": tau, "height": int(pred.shape[0]), "width": int(pred.shape[1]),
            "prediction_c": _rounded_flat(pred), "ground_truth_c": _rounded_flat(gt),
            "prediction_summary": _map_summary(pred), "ground_truth_summary": _map_summary(gt), "metrics": metrics}
