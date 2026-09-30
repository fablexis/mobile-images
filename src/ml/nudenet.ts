import type { SkImage } from '@shopify/react-native-skia';
import type { TensorflowModel } from 'react-native-fast-tflite';

import { useModelWithFallback } from './modelLoader';
import { DEFAULT_POLICY, NUDENET_LABELS, type NudeNetLabel, type Policy, shouldBlur } from './policy';
import { imageToTensor, NUDENET_INPUT_SIZE } from './preprocess';

// NudeNet v3 320n (YOLOv8n) converted with onnxsim (static 1x3x320x320) + onnx2tf.
// Verified by scripts/convert_models.py:
//   input  float32 [1,320,320,3] NHWC, RGB / 255, image padded right/bottom to square
//   output float32 [1,22,2100] = [batch, 4 box + 18 classes, anchors]
//          rows 0..3: cx, cy, w, h in input pixels (0..320); rows 4..21: class scores
//          (already sigmoid, no objectness column)
const NUDENET_MODEL = require('../../assets/models/nudenet_320n.tflite');

const NUM_CLASSES = NUDENET_LABELS.length;
const NUM_CHANNELS = 4 + NUM_CLASSES;

export interface Box {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface NudeNetDetection {
  label: NudeNetLabel;
  score: number;
  /** In original image pixels. */
  box: Box;
  blur: boolean;
}

export function useNudeNet() {
  return useModelWithFallback(NUDENET_MODEL);
}

function iou(a: Box, b: Box): number {
  const x1 = Math.max(a.x, b.x);
  const y1 = Math.max(a.y, b.y);
  const x2 = Math.min(a.x + a.width, b.x + b.width);
  const y2 = Math.min(a.y + a.height, b.y + b.height);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const union = a.width * a.height + b.width * b.height - inter;
  return union <= 0 ? 0 : inter / union;
}

/** Greedy class-agnostic NMS, same as cv2.dnn.NMSBoxes used by NudeNet. */
function nms<T extends { box: Box; score: number }>(items: T[], iouThreshold: number): T[] {
  const sorted = [...items].sort((a, b) => b.score - a.score);
  const kept: T[] = [];
  for (const item of sorted) {
    if (kept.every((k) => iou(k.box, item.box) <= iouThreshold)) kept.push(item);
  }
  return kept;
}

/**
 * Decodes the YOLOv8 head. Supports both [1, C, N] (what our TFLite emits) and
 * [1, N, C] in case a different converter transposes the output.
 */
export function decodeYolo(
  output: Float32Array,
  outputShape: number[],
  scale: number,
  imageWidth: number,
  imageHeight: number,
  policy: Policy = DEFAULT_POLICY,
): NudeNetDetection[] {
  // The output signature is [-1, 22, -1], so trust the buffer length for N.
  const dims = outputShape.length === 3 ? outputShape.slice(1) : outputShape;
  const channelsFirst = !(dims[1] === NUM_CHANNELS && dims[0] !== NUM_CHANNELS);
  const numAnchors = Math.floor(output.length / NUM_CHANNELS);
  const at = channelsFirst
    ? (c: number, a: number) => output[c * numAnchors + a]
    : (c: number, a: number) => output[a * NUM_CHANNELS + c];

  const candidates: NudeNetDetection[] = [];
  for (let a = 0; a < numAnchors; a++) {
    let best = 0;
    let bestClass = 0;
    for (let c = 0; c < NUM_CLASSES; c++) {
      const s = at(4 + c, a);
      if (s > best) {
        best = s;
        bestClass = c;
      }
    }
    if (best < policy.detectionScoreThreshold) continue;

    const cx = at(0, a);
    const cy = at(1, a);
    const w = at(2, a);
    const h = at(3, a);
    // Letterbox padding is right/bottom only, so un-scaling is a single divide.
    let x = (cx - w / 2) / scale;
    let y = (cy - h / 2) / scale;
    let bw = w / scale;
    let bh = h / scale;
    x = Math.max(0, Math.min(x, imageWidth));
    y = Math.max(0, Math.min(y, imageHeight));
    bw = Math.min(bw, imageWidth - x);
    bh = Math.min(bh, imageHeight - y);
    if (bw <= 1 || bh <= 1) continue;

    const label = NUDENET_LABELS[bestClass];
    candidates.push({
      label,
      score: best,
      box: { x, y, width: bw, height: bh },
      blur: shouldBlur(label, policy),
    });
  }
  return nms(candidates, policy.nmsIouThreshold);
}

export async function detectNudity(
  model: TensorflowModel,
  image: SkImage,
  policy: Policy = DEFAULT_POLICY,
): Promise<NudeNetDetection[]> {
  const { tensor, scale } = imageToTensor(image, NUDENET_INPUT_SIZE, 'letterbox');
  const [out] = await model.run([tensor]);
  const shape = model.outputs[0]?.shape ?? [1, NUM_CHANNELS, (out as Float32Array).length / NUM_CHANNELS];
  return decodeYolo(out as Float32Array, shape, scale, image.width(), image.height(), policy);
}
