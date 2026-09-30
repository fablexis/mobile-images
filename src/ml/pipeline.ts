import type { SkImage } from '@shopify/react-native-skia';
import { ImageManipulator, SaveFormat } from 'expo-image-manipulator';
import type { TensorflowModel } from 'react-native-fast-tflite';

import { detectFaces, ensureFileUri } from './faces';
import { classifyImage } from './nsfwClassifier';
import { detectNudity, type Box, type NudeNetDetection } from './nudenet';
import { classify, DEFAULT_POLICY, type NsfwScores, type Policy, type Verdict } from './policy';
import { decodeImage } from './preprocess';

const MAX_SIDE = 1600;

export interface GalleryAnalysis {
  uri: string;
  width: number;
  height: number;
  image: SkImage;
  faces: Box[];
  scores: NsfwScores;
  verdict: Verdict;
  /** null when NudeNet was skipped because the classifier said "safe". */
  detections: NudeNetDetection[] | null;
  timingsMs: { prepare: number; faces: number; classifier: number; nudenet: number | null };
}

/**
 * Re-encodes the picked image so that EXIF orientation is baked into the pixels
 * and the longest side is at most MAX_SIDE. Skia (which ignores EXIF), ML Kit and
 * the on-screen overlay then all share the same pixel coordinate space.
 */
async function normalizeImage(uri: string, width: number, height: number) {
  const fileUri = await ensureFileUri(uri);
  const ctx = ImageManipulator.manipulate(fileUri);
  if (Math.max(width, height) > MAX_SIDE) {
    ctx.resize(width >= height ? { width: MAX_SIDE } : { height: MAX_SIDE });
  }
  const ref = await ctx.renderAsync();
  return ref.saveAsync({ format: SaveFormat.JPEG, compress: 0.92 });
}

export async function analyzeGalleryImage(
  asset: { uri: string; width: number; height: number },
  models: { classifier: TensorflowModel; nudenet?: TensorflowModel },
  onStep: (text: string) => void,
  policy: Policy = DEFAULT_POLICY,
): Promise<GalleryAnalysis> {
  let t = Date.now();
  onStep('Preparando imagen…');
  const normalized = await normalizeImage(asset.uri, asset.width, asset.height);
  const image = await decodeImage(normalized.uri);
  const prepare = Date.now() - t;

  onStep('Detectando rostros y clasificando…');
  t = Date.now();
  const facesPromise = detectFaces(normalized.uri).then((f) => ({ faces: f, ms: Date.now() - t }));
  const tc = Date.now();
  const scores = await classifyImage(models.classifier, image);
  const classifierMs = Date.now() - tc;
  const { faces, ms: facesMs } = await facesPromise;

  const verdict = classify(scores, policy);

  let detections: NudeNetDetection[] | null = null;
  let nudenetMs: number | null = null;
  if (verdict !== 'safe' && models.nudenet) {
    onStep('Buscando zonas expuestas…');
    const tn = Date.now();
    detections = await detectNudity(models.nudenet, image, policy);
    nudenetMs = Date.now() - tn;
  }

  return {
    uri: normalized.uri,
    width: image.width(),
    height: image.height(),
    image,
    faces: faces.map((f) => f.box),
    scores,
    verdict,
    detections,
    timingsMs: { prepare, faces: facesMs, classifier: classifierMs, nudenet: nudenetMs },
  };
}
