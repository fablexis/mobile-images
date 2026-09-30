# Model conversion

`convert_models.py` produces the two TFLite models the app bundles:

| Output | Source | Input | Output |
| --- | --- | --- | --- |
| `assets/models/nsfw_mobilenet_v2.tflite` | GantMan/nsfw_model release `1.1.0`, `nsfw_mobilenet_v2_140_224.zip` → `mobilenet_v2_140_224/saved_model.h5` | float32 `[1,224,224,3]` RGB in `[0,1]` | float32 `[1,5]` softmax |
| `assets/models/nudenet_320n.tflite` | notAI-tech/NudeNet release `v3.4-weights`, `320n.onnx` | float32 `[1,320,320,3]` NHWC, RGB in `[0,1]` | float32 `[1,22,2100]` |

## Running it

onnx2tf 2.x requires Python >= 3.12 (macOS system Python 3.9 is too old).

```bash
# from the repo root
python3.12 -m venv .venv-convert
source .venv-convert/bin/activate
pip install -r scripts/requirements.txt
python scripts/convert_models.py                 # both models
python scripts/convert_models.py --skip-nudenet  # only the classifier
python scripts/convert_models.py --test-image some.jpg   # smoke test on a real image
```

Downloads and intermediate files go to `scripts/.work/` (git-ignored; override with `--work-dir`).
The script is idempotent: cached downloads are reused.

If you don't have Python 3.12, `uv` works without touching the system:

```bash
pip3 install --user uv
uv venv -p 3.12 .venv-convert && source .venv-convert/bin/activate
uv pip install -r scripts/requirements.txt
```

## What the script verifies (last run)

```
class_labels.txt: ['drawings', 'hentai', 'neutral', 'porn', 'sexy']
loaded .h5: input=(None, 224, 224, 3) output=(None, 5)
converted from .h5 with TFLiteConverter.from_keras_model
nsfw_mobilenet_v2.tflite (24.43 MB)
  input  shape=[1, 224, 224, 3] dtype=float32
  output shape=[1, 5]           dtype=float32
  smoke test (synthetic gradient, [0,1]): neutral 0.894, sum of outputs = 1.0000

ONNX input  images [batch, 3, height, width]    (dynamic)
static ONNX output output0 [1, 22, 2100]
nudenet_320n.tflite (12.13 MB)
  input  shape=[1, 320, 320, 3]  signature=[1, 320, 320, 3]  dtype=float32   (NHWC)
  output shape=[1, 22, 2100]     signature=[-1, 22, -1]      dtype=float32
  max |onnx - tflite| = 0.00023 on the same input
  decoded layout: channels=22 (4 box + 18 classes), anchors=2100
  box rows range 4.98..322.37 -> cx, cy, w, h in pixels of the 320 input
```

## Layout notes (mirrored in `src/ml/*.ts`)

### GantMan classifier
- Class order: `drawings, hentai, neutral, porn, sexy` (from `class_labels.txt`; the app calls the first one `drawing`).
- Normalization: RGB / 255 (range `[0,1]`), no ImageNet mean/std. That matches nsfwjs and the
  TF-Hub MobileNetV2 feature vector GantMan trained on. The softmax is inside the model.
- The `.h5` embeds a TF-Hub `KerasLayer` saved with Keras 2, so it is loaded with
  `tf_keras` + `tensorflow_hub` and `TF_USE_LEGACY_KERAS=1`. If that fails the script falls back
  to `TFLiteConverter.from_saved_model` on the SavedModel shipped in the same zip, and finally to
  the release's own `saved_model.tflite`.
- App preprocessing: gallery images are stretched to 224x224 with Skia; camera frames are
  center-cropped + resized by `vision-camera-resize-plugin` with `dataType: 'float32'`, which already
  yields `[0,1]`.

### NudeNet 320n (YOLOv8n)
- Labels: the order from `nudenet/nudenet.py` (`__labels`) on the `v3` branch. It differs from the order
  often quoted in docs (e.g. index 4 is `FEMALE_GENITALIA_EXPOSED`, not `FEMALE_BREAST_COVERED`, and
  `BUTTOCKS_COVERED` is index 17).
- Preprocessing (same as NudeNet): pad the image right/bottom with black to a square, resize to 320,
  RGB / 255.
- Output `[1, 22, 2100]`: rows 0-3 are `cx, cy, w, h` in 320-input pixels, rows 4-21 are per-class
  scores (already sigmoid, no objectness). Boxes map back to the original image with
  `/ (320 / max(w, h))`. NMS is class-agnostic, IoU 0.45 (NudeNet uses `cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45)`).

## Known conversion issues

- **`onnx2tf -ois images:1,3,320,320` is ignored for this graph.** The exported ONNX has dynamic
  `batch/height/width`; with `-ois` alone the TFLite came out with input signature `[-1,-1,-1,3]`
  (reported as `[1,1,1,3]`) and inference failed with `Dimension mismatch`. The script therefore
  freezes the shape first with `onnxsim.simplify(..., overwrite_input_shapes={'images': [1,3,320,320]})`
  and converts the static ONNX with `onnx2tf -b 1`.
- The output tensor keeps a partially dynamic signature `[-1, 22, -1]`. After `allocate_tensors()` its
  shape is `[1, 22, 2100]`; the app derives the anchor count from the buffer length to be safe.
- `https://github.com/notAI-tech/NudeNet/releases/download/v3.4-weights/320n.onnx` redirected to a
  GitHub login page at the time of writing. The script falls back to the REST asset endpoint
  (`api.github.com/.../releases/assets/<id>` with `Accept: application/octet-stream`) and then to
  the `nudenet` PyPI wheel, which ships the same `320n.onnx`.
