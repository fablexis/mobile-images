import type { SkImage } from '@shopify/react-native-skia';
import type { TensorflowModel } from 'react-native-fast-tflite';

import { useModelWithFallback } from './modelLoader';
import type { NsfwScores } from './policy';
import { imageToTensor, NSFW_INPUT_SIZE } from './preprocess';

// GantMan nsfw_model MobileNetV2 1.4 (release 1.1.0), converted by scripts/convert_models.py.
// Input float32 [1,224,224,3] RGB in [0,1]; output float32 [1,5] softmax
// in the order drawings, hentai, neutral, porn, sexy.
const NSFW_MODEL = require('../../assets/models/nsfw_mobilenet_v2.tflite');

export function useNsfwClassifier() {
  return useModelWithFallback(NSFW_MODEL);
}

/** Maps the raw [1,5] output to named scores. Safe to call from a worklet. */
export function toScores(out: ArrayLike<number>): NsfwScores {
  'worklet';
  return {
    drawing: Number(out[0]),
    hentai: Number(out[1]),
    neutral: Number(out[2]),
    porn: Number(out[3]),
    sexy: Number(out[4]),
  };
}

export async function classifyImage(model: TensorflowModel, image: SkImage): Promise<NsfwScores> {
  const { tensor } = imageToTensor(image, NSFW_INPUT_SIZE, 'stretch');
  const [out] = await model.run([tensor]);
  return toScores(out as Float32Array);
}
