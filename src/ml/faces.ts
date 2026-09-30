import FaceDetection, { type Face } from '@react-native-ml-kit/face-detection';
import { File, Paths } from 'expo-file-system';

import type { Box } from './nudenet';

export interface DetectedFace {
  box: Box;
}

/**
 * ML Kit on iOS reads the image with `NSData dataWithContentsOfURL`, so it needs a
 * real file:// URL. Image-picker can hand back ph:// (iOS) or content:// (Android)
 * URIs; copy those into the cache first.
 */
export async function ensureFileUri(uri: string): Promise<string> {
  if (uri.startsWith('file://')) return uri;
  if (uri.startsWith('/')) return `file://${uri}`;
  const ext = uri.split('?')[0].split('.').pop();
  const name = `face-input-${Date.now()}.${ext && ext.length <= 4 ? ext : 'jpg'}`;
  const dest = new File(Paths.cache, name);
  new File(uri).copy(dest);
  return dest.uri;
}

/** Static-image face detection. Boxes are in the pixel space of the file at `uri`. */
export async function detectFaces(uri: string): Promise<DetectedFace[]> {
  const fileUri = await ensureFileUri(uri);
  const faces: Face[] = await FaceDetection.detect(fileUri, {
    performanceMode: 'accurate',
    landmarkMode: 'none',
    contourMode: 'none',
    classificationMode: 'none',
    minFaceSize: 0.05,
  });
  return faces.map((f) => ({
    box: { x: f.frame.left, y: f.frame.top, width: f.frame.width, height: f.frame.height },
  }));
}
