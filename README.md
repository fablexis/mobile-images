# mobile-images

On-device face detection and NSFW screening for React Native. Photos are analyzed on the phone. Nothing is uploaded.

The app does not run in Expo Go. Native modules (camera, ML Kit, TFLite) require a development build.

## Run the app

Requirements: Node.js, Xcode (iOS), and a simulator or a physical iPhone. Android also needs a JDK and the Android SDK.

```bash
npm install
npx expo run:ios --device "iPhone 17"    # simulator
npx expo run:ios --device                # a connected iPhone
npx expo run:android
```

`npm start` only launches Metro for an already-installed dev build (`expo start --dev-client`).

Check types with `npm run typecheck`.

### Simulator

The iOS Simulator has no camera, so the live screen cannot preview frames there. The gallery screen works:

1. Drag a JPG or PNG onto the Simulator window. It is saved to Photos.
2. In the app, open Gallery and pick that photo.
3. Allow photo access when asked.

ML Kit ships an arm64 slice built for real devices. `ios/Podfile` runs `scripts/patch_mlkit_simulator.py` after `pod install` so the Apple Silicon simulator can link it. A physical iPhone does not need that patch. `ios/` and `android/` are generated (`npx expo prebuild`) and are not part of the git tree.

### Regenerate the models

The `.tflite` files in `assets/models/` are already built. Rebuild them only if you change the conversion:

```bash
python3.12 -m venv .venv-convert
source .venv-convert/bin/activate
pip install -r scripts/requirements.txt
python scripts/convert_models.py
```

Details and the last measured tensor shapes are in `scripts/README.md`.

## What it does

The home screen offers two modes.

**Gallery.** Pick a photo. The app detects faces, classifies the whole image, and, when the image is not clearly safe, finds exposed regions and blurs only those regions.

**Camera.** Detects faces on every frame and runs the NSFW classifier about three times per second. It shows a verdict. It does not run the region detector and it does not blur the preview.

Verdicts come from `src/ml/policy.ts`:

```
porn + hentai  >  0.7     ->  NSFW
sexy           >  0.6     ->  Review
otherwise                 ->  Safe
```

The five classifier scores are Drawing, Hentai, Neutral, Porn, and Sexy.

Blur applies to NudeNet labels that end in `_EXPOSED`, except faces and feet. Covered regions are drawn but not blurred.

## Architecture

Everything below runs on the device. There is no backend.

```
+------------------+     +------------------+     +-----------------------+
|  Gallery         |     |  Live camera     |     |  On-device runtimes   |
|  expo-image-     |     |  Vision Camera   |     |                       |
|  picker          |     |  frame processor |     |  ML Kit     faces     |
+--------+---------+     +--------+---------+     |  TFLite     NSFW      |
         |                        |               |  Skia       draw/blur |
         v                        v               +-----------------------+
+------------------+     +------------------+              ^    ^
| Normalize JPEG   |     | resize plugin    |              |    |
| EXIF baked in    |     | 224 RGB, ~3 fps  |--------------+    |
| longest side     |     +--------+---------+                   |
| <= 1600 px       |              |                              |
+--------+---------+              |                              |
         |                        v                              |
         +--------------->  GantMan classifier  ----------------+
         |                 MobileNetV2 224
         |
         |   faces, in parallel
         +--------------->  ML Kit face detection
         |
         |   only when verdict is Review or NSFW
         +--------------->  NudeNet 320n
                            boxes + blur
```

### Gallery pipeline

```
picked photo
    |
    v
copy to a real file if needed
    |
    v
re-encode JPEG, bake EXIF rotation, cap the long side at 1600 px
    |
    +---------------------------+
    |                           |
    v                           v
ML Kit faces              GantMan TFLite
box per face              5 class scores
    |                           |
    |                           v
    |                     Safe / Review / NSFW
    |                           |
    |                     Safe? ----yes----> skip NudeNet
    |                           |
    |                           no
    |                           |
    |                           v
    |                     NudeNet TFLite
    |                     YOLO decode + NMS
    |                           |
    +-------------+-------------+
                  |
                  v
         Skia overlay
         face boxes
         zone boxes
         blur on exposed zones
```

NudeNet runs only after the cheap classifier flags the image. That keeps the heavy detector off photos that are already Safe.

### Camera pipeline

```
camera frame
    |
    +-- every frame --------> face detector plugin
    |                         boxes already in preview coordinates
    |
    +-- about 3 times/sec --> resize to 224x224 RGB
                              GantMan TFLite (runSync in the frame processor)
                                  |
                                  v
                              verdict + score bars
```

NudeNet is not part of the live path.

### Models

```
assets/models/nsfw_mobilenet_v2.tflite     224 x 224 RGB in [0, 1]
                                           out: [1, 5] softmax

assets/models/nudenet_320n.tflite          320 x 320 RGB in [0, 1]
                                           out: [1, 22, 2100]
                                           4 box values (cx, cy, w, h)
                                           + 18 class scores, no objectness
```

Both are float32, NHWC. Core ML is enabled on iOS and the GPU delegate on Android. If a delegate rejects a model, loading falls back to CPU (`src/ml/modelLoader.ts`).

Inference uses `react-native-fast-tflite`. Face detection uses `@react-native-ml-kit/face-detection` for still photos and `react-native-vision-camera-face-detector` for the live preview.

### Source layout

```
app/index.tsx              home
app/gallery.tsx            pick a photo, show scores and overlay
app/camera.tsx             live faces + throttled classifier
src/ml/pipeline.ts         gallery cascade
src/ml/policy.ts           thresholds and which labels get blurred
src/ml/nsfwClassifier.ts   GantMan scores
src/ml/nudenet.ts          YOLO decode and NMS
src/ml/faces.ts            still-image ML Kit wrapper
src/ml/preprocess.ts       Skia pixels and camera resize -> float tensor
src/components/            score bars and the Skia overlay
scripts/convert_models.py  optional: rebuild the .tflite files
scripts/patch_mlkit_simulator.py
```

## Stack

Expo SDK 54, React Native 0.81, expo-router, expo-dev-client. Vision Camera stays on v4 so it matches the face-detector and TFLite versions pinned in `package.json`.
