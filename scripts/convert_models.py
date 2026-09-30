#!/usr/bin/env python3
"""
Downloads and converts the two on-device models used by the app:

1. GantMan nsfw_model MobileNetV2 (release 1.1.0, Keras .h5)  -> assets/models/nsfw_mobilenet_v2.tflite
2. NudeNet v3 detector 320n.onnx (release v3.4-weights)       -> assets/models/nudenet_320n.tflite

Usage (from the repository root):
    python3.12 -m venv .venv-convert && source .venv-convert/bin/activate
    pip install -r scripts/requirements.txt
    python scripts/convert_models.py [--work-dir /tmp/mi-convert/dl] [--skip-nsfw] [--skip-nudenet]

The script prints input/output tensor shapes and runs a smoke test on each model
so the TypeScript pre/post-processing can be aligned with the real layout.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

# The GantMan .h5 embeds a TF-Hub KerasLayer saved with Keras 2; it only loads
# with tf_keras, and tf.lite must see the same Keras flavour.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "assets" / "models"

NSFW_ZIP_URL = "https://github.com/GantMan/nsfw_model/releases/download/1.1.0/nsfw_mobilenet_v2_140_224.zip"
NSFW_OUT = MODELS_DIR / "nsfw_mobilenet_v2.tflite"
NSFW_SIZE = 224
NSFW_CLASSES = ["drawings", "hentai", "neutral", "porn", "sexy"]

# The public "browser_download_url" sometimes redirects to a GitHub login page for
# this repo; the API asset endpoint (with Accept: application/octet-stream) works.
NUDENET_URLS = [
    "https://github.com/notAI-tech/NudeNet/releases/download/v3.4-weights/320n.onnx",
    "https://api.github.com/repos/notAI-tech/NudeNet/releases/assets/176831997",
]
# Fallback: the nudenet PyPI wheel ships the same 320n.onnx.
NUDENET_WHEEL_URL = "https://files.pythonhosted.org/packages/1c/ee/1aa02d44ba958cc77e16ff1e41a0aac5e721037db7bf62b9c9d124917f87/nudenet-3.4.2-py3-none-any.whl"
NUDENET_OUT = MODELS_DIR / "nudenet_320n.tflite"
NUDENET_SIZE = 320

# Order copied from notAI-tech/NudeNet (branch v3) nudenet/nudenet.py `__labels`.
NUDENET_LABELS = [
    "FEMALE_GENITALIA_COVERED",
    "FACE_FEMALE",
    "BUTTOCKS_EXPOSED",
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_BREAST_EXPOSED",
    "ANUS_EXPOSED",
    "FEET_EXPOSED",
    "BELLY_COVERED",
    "FEET_COVERED",
    "ARMPITS_COVERED",
    "ARMPITS_EXPOSED",
    "FACE_MALE",
    "BELLY_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "ANUS_COVERED",
    "FEMALE_BREAST_COVERED",
    "BUTTOCKS_COVERED",
]


def log(msg: str) -> None:
    print(f"[convert] {msg}", flush=True)


def download(urls: list[str], dest: Path, min_bytes: int) -> bool:
    if dest.exists() and dest.stat().st_size >= min_bytes:
        log(f"cached: {dest} ({dest.stat().st_size} bytes)")
        return True
    for url in urls:
        try:
            log(f"downloading {url}")
            req = urllib.request.Request(
                url,
                headers={"Accept": "application/octet-stream", "User-Agent": "convert-models"},
            )
            with urllib.request.urlopen(req) as resp, open(dest, "wb") as f:
                shutil.copyfileobj(resp, f)
            size = dest.stat().st_size
            if size >= min_bytes:
                log(f"ok: {dest} ({size} bytes)")
                return True
            log(f"file too small ({size} bytes), probably an HTML page; trying next source")
        except Exception as e:  # noqa: BLE001
            log(f"download failed: {e}")
    return False


def describe_tflite(path: Path):
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        from tensorflow.lite import Interpreter  # type: ignore

    interpreter = Interpreter(model_path=str(path))
    interpreter.allocate_tensors()
    log(f"TFLite model: {path.name} ({path.stat().st_size / 1e6:.2f} MB)")
    for kind, details in (("input ", interpreter.get_input_details()), ("output", interpreter.get_output_details())):
        for d in details:
            log(
                f"  {kind} name={d['name']} shape={d['shape'].tolist()} "
                f"signature={d['shape_signature'].tolist()} dtype={d['dtype'].__name__}"
            )
    return interpreter


def run_tflite(interpreter, x: np.ndarray) -> list[np.ndarray]:
    inp = interpreter.get_input_details()[0]
    interpreter.set_tensor(inp["index"], x.astype(inp["dtype"]))
    interpreter.invoke()
    return [interpreter.get_tensor(d["index"]) for d in interpreter.get_output_details()]


def synthetic_image(size: int) -> np.ndarray:
    """Smooth RGB gradient in [0, 1], HWC."""
    y, x = np.mgrid[0:size, 0:size].astype(np.float32) / (size - 1)
    return np.stack([x, y, 1.0 - x * y], axis=-1)


def load_test_image(path: str | None, size: int, letterbox: bool) -> np.ndarray:
    if not path:
        return synthetic_image(size)
    from PIL import Image

    img = Image.open(path).convert("RGB")
    if letterbox:
        side = max(img.size)
        canvas = Image.new("RGB", (side, side), (0, 0, 0))
        canvas.paste(img, (0, 0))
        img = canvas
    img = img.resize((size, size), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


# ---------------------------------------------------------------------------
# 1. GantMan MobileNetV2
# ---------------------------------------------------------------------------


def convert_nsfw(work: Path, test_image: str | None) -> None:
    import tensorflow as tf

    zip_path = work / "nsfw_mobilenet_v2_140_224.zip"
    if not download([NSFW_ZIP_URL], zip_path, 50_000_000):
        raise RuntimeError("could not download GantMan model")
    extract_dir = work / "nsfw"
    if not (extract_dir / "mobilenet_v2_140_224").exists():
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(extract_dir)
    model_dir = extract_dir / "mobilenet_v2_140_224"
    h5_path = model_dir / "saved_model.h5"
    labels_path = model_dir / "class_labels.txt"
    if labels_path.exists():
        log(f"class_labels.txt: {labels_path.read_text().split()}")
    train_results = model_dir / "train_results.txt"
    if train_results.exists():
        log("train_results.txt:\n" + train_results.read_text())

    tflite_bytes = None
    try:
        import tensorflow_hub as hub
        import tf_keras

        model = tf_keras.models.load_model(
            str(h5_path), custom_objects={"KerasLayer": hub.KerasLayer}, compile=False
        )
        log(f"loaded .h5: input={model.input_shape} output={model.output_shape}")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_bytes = converter.convert()
        log("converted from .h5 with TFLiteConverter.from_keras_model")
    except Exception as e:  # noqa: BLE001
        log(f".h5 conversion failed ({type(e).__name__}: {e}); trying SavedModel directory")
        try:
            converter = tf.lite.TFLiteConverter.from_saved_model(str(model_dir))
            tflite_bytes = converter.convert()
            log("converted from SavedModel with TFLiteConverter.from_saved_model")
        except Exception as e2:  # noqa: BLE001
            log(f"SavedModel conversion failed ({type(e2).__name__}: {e2})")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if tflite_bytes is None:
        bundled = model_dir / "saved_model.tflite"
        log(f"falling back to the release's own float32 TFLite: {bundled}")
        shutil.copy(bundled, NSFW_OUT)
    else:
        NSFW_OUT.write_bytes(tflite_bytes)

    interpreter = describe_tflite(NSFW_OUT)
    # nsfwjs feeds this model RGB / 255 (range [0, 1]); the TF-Hub MobileNetV2
    # feature vector used by GantMan's trainer expects the same range.
    x = load_test_image(test_image, NSFW_SIZE, letterbox=False)[None, ...]
    probs = run_tflite(interpreter, x)[0][0]
    log("smoke test (input [0,1] RGB NHWC): " + json.dumps({c: round(float(p), 4) for c, p in zip(NSFW_CLASSES, probs)}))
    log(f"  sum of outputs = {float(np.sum(probs)):.4f} (≈1.0 means softmax is inside the model)")


# ---------------------------------------------------------------------------
# 2. NudeNet v3 320n
# ---------------------------------------------------------------------------


def fetch_nudenet_onnx(work: Path) -> Path:
    onnx_path = work / "320n.onnx"
    if download(NUDENET_URLS, onnx_path, 5_000_000):
        return onnx_path
    wheel = work / "nudenet.whl"
    if not download([NUDENET_WHEEL_URL], wheel, 5_000_000):
        raise RuntimeError("could not download NudeNet 320n.onnx")
    with zipfile.ZipFile(wheel) as z:
        member = next(n for n in z.namelist() if n.endswith("320n.onnx"))
        with z.open(member) as src, open(onnx_path, "wb") as dst:
            shutil.copyfileobj(src, dst)
    return onnx_path


def convert_nudenet(work: Path, test_image: str | None) -> None:
    import onnx
    import onnxruntime as ort

    onnx_path = fetch_nudenet_onnx(work)
    m = onnx.load(str(onnx_path))
    for t in m.graph.input:
        dims = [d.dim_value or d.dim_param for d in t.type.tensor_type.shape.dim]
        log(f"ONNX input  {t.name} {dims}")
    for t in m.graph.output:
        dims = [d.dim_value or d.dim_param for d in t.type.tensor_type.shape.dim]
        log(f"ONNX output {t.name} {dims}")

    # The exported ONNX has dynamic batch/height/width. onnx2tf's -ois flag is
    # ignored for this graph (the TFLite ends up with signature [-1,-1,-1,3]),
    # so freeze the input shape with onnxsim first.
    import onnxsim

    static_path = work / "320n_static.onnx"
    input_name = m.graph.input[0].name
    simplified, check = onnxsim.simplify(
        m, overwrite_input_shapes={input_name: [1, 3, NUDENET_SIZE, NUDENET_SIZE]}
    )
    if not check:
        raise RuntimeError("onnxsim could not validate the static-shape model")
    onnx.save(simplified, str(static_path))
    for t in simplified.graph.output:
        log(f"static ONNX output {t.name} {[d.dim_value for d in t.type.tensor_type.shape.dim]}")

    out_dir = work / "nudenet_tf"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    cmd = [
        sys.executable,
        "-m",
        "onnx2tf",
        "-i",
        str(static_path),
        "-o",
        str(out_dir),
        "-b",
        "1",
        "-n",
    ]
    log("running: " + " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        log(proc.stdout[-4000:])
        log(proc.stderr[-4000:])
        raise RuntimeError("onnx2tf failed (see output above and scripts/README.md)")
    candidates = sorted(out_dir.glob("*float32.tflite"))
    if not candidates:
        raise RuntimeError(f"onnx2tf produced no *_float32.tflite in {out_dir}: {os.listdir(out_dir)}")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(candidates[0], NUDENET_OUT)

    interpreter = describe_tflite(NUDENET_OUT)

    # NudeNet preprocessing: pad right/bottom to a square with black, resize to
    # 320, RGB / 255. ONNX expects NCHW, onnx2tf's TFLite expects NHWC.
    img = load_test_image(test_image, NUDENET_SIZE, letterbox=True)
    onnx_out = ort.InferenceSession(str(onnx_path)).run(None, {m.graph.input[0].name: img.transpose(2, 0, 1)[None]})[0]
    tfl_out = run_tflite(interpreter, img[None])[0]
    log(f"ONNX output shape {list(onnx_out.shape)}, TFLite output shape {list(tfl_out.shape)}")
    if onnx_out.shape == tfl_out.shape:
        log(f"  max |onnx - tflite| = {float(np.max(np.abs(onnx_out - tfl_out))):.5f}")
    elif onnx_out.shape == tfl_out.transpose(0, 2, 1).shape:
        log("  TFLite output is transposed vs ONNX ([1, anchors, 4+classes])")
        log(f"  max |onnx - tflite^T| = {float(np.max(np.abs(onnx_out - tfl_out.transpose(0, 2, 1)))):.5f}")

    # YOLOv8 head: [1, 4 + num_classes, num_anchors]; rows 0..3 = cx, cy, w, h in
    # input pixels (0..320), rows 4.. = per-class sigmoid scores (no objectness).
    out = tfl_out[0]
    if out.shape[0] != 4 + len(NUDENET_LABELS):
        out = out.T
    log(f"decoded layout: channels={out.shape[0]} (4 box + {out.shape[0] - 4} classes), anchors={out.shape[1]}")
    log(f"  box rows range: min={float(out[:4].min()):.2f} max={float(out[:4].max()):.2f} (pixels of the 320 input)")
    scores = out[4:]
    best = np.unravel_index(np.argmax(scores), scores.shape)
    log(f"  max class score = {float(scores.max()):.4f} label={NUDENET_LABELS[best[0]]} anchor={best[1]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--work-dir", default=str(ROOT / "scripts" / ".work"))
    parser.add_argument("--test-image", default=None, help="optional image for the smoke tests")
    parser.add_argument("--skip-nsfw", action="store_true")
    parser.add_argument("--skip-nudenet", action="store_true")
    args = parser.parse_args()

    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    ok = True
    if not args.skip_nsfw:
        try:
            convert_nsfw(work, args.test_image)
        except Exception as e:  # noqa: BLE001
            ok = False
            log(f"NSFW classifier FAILED: {e}")
    if not args.skip_nudenet:
        try:
            convert_nudenet(work, args.test_image)
        except Exception as e:  # noqa: BLE001
            ok = False
            log(f"NudeNet FAILED: {e}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
