import { AlphaType, ColorType, Skia, type SkImage } from '@shopify/react-native-skia';
import type { Frame } from 'react-native-vision-camera';
import type { ResizePlugin } from 'vision-camera-resize-plugin';

export const NSFW_INPUT_SIZE = 224;
export const NUDENET_INPUT_SIZE = 320;

/** Decodes an encoded image (file:// URI) into a Skia image. */
export async function decodeImage(uri: string): Promise<SkImage> {
  const data = await Skia.Data.fromURI(uri);
  const image = Skia.Image.MakeImageFromEncoded(data);
  if (!image) throw new Error(`No se pudo decodificar la imagen: ${uri}`);
  return image;
}

/** Converts RGBA uint8 pixels into a packed RGB float32 tensor in [0, 1] (NHWC). */
function rgbaToRgbFloat(rgba: Uint8Array | Float32Array, pixelCount: number): Float32Array {
  const out = new Float32Array(pixelCount * 3);
  for (let i = 0, j = 0; i < pixelCount; i++, j += 3) {
    const k = i * 4;
    out[j] = rgba[k] / 255;
    out[j + 1] = rgba[k + 1] / 255;
    out[j + 2] = rgba[k + 2] / 255;
  }
  return out;
}

/**
 * Draws `image` into a size x size offscreen surface and returns RGB float32 [0,1].
 * - 'stretch': plain resize, like nsfwjs does for the GantMan classifier.
 * - 'letterbox': pad right/bottom with black to a square, then resize, like NudeNet.
 *   Returns `scale` = size / max(w, h) to map boxes back to original pixels.
 */
export function imageToTensor(
  image: SkImage,
  size: number,
  mode: 'stretch' | 'letterbox',
): { tensor: Float32Array; scale: number } {
  const surface = Skia.Surface.MakeOffscreen(size, size) ?? Skia.Surface.Make(size, size);
  if (!surface) throw new Error('No se pudo crear la superficie de Skia');

  const w = image.width();
  const h = image.height();
  const canvas = surface.getCanvas();
  canvas.clear(Skia.Color('black'));

  const paint = Skia.Paint();
  let scale = 1;
  const src = Skia.XYWHRect(0, 0, w, h);
  if (mode === 'stretch') {
    canvas.drawImageRect(image, src, Skia.XYWHRect(0, 0, size, size), paint);
  } else {
    scale = size / Math.max(w, h);
    canvas.drawImageRect(image, src, Skia.XYWHRect(0, 0, w * scale, h * scale), paint);
  }
  surface.flush();

  const snapshot = surface.makeImageSnapshot();
  const pixels = snapshot.readPixels(0, 0, {
    width: size,
    height: size,
    colorType: ColorType.RGBA_8888,
    alphaType: AlphaType.Unpremul,
  });
  if (!pixels) throw new Error('readPixels devolvió null');
  return { tensor: rgbaToRgbFloat(pixels, size * size), scale };
}

type FrameRotation = '0deg' | '90deg' | '180deg' | '270deg';

/**
 * Rotation that makes a VisionCamera v4 frame upright. Frames arrive in sensor
 * orientation; `frame.orientation` says how the buffer is rotated.
 */
export function frameRotation(frame: Frame): FrameRotation {
  'worklet';
  switch (frame.orientation) {
    case 'landscape-right':
      return '90deg';
    case 'portrait-upside-down':
      return '180deg';
    case 'landscape-left':
      return '270deg';
    default:
      return '0deg';
  }
}

/**
 * Camera-frame path: center-crop + resize to size x size RGB float32 in [0,1]
 * with vision-camera-resize-plugin. The classifier input tensor is float32
 * [1,224,224,3] and expects /255 normalization, which is exactly what the
 * plugin's 'float32' dataType produces, so no extra conversion is needed.
 */
export function frameToTensor(frame: Frame, resize: ResizePlugin['resize'], size: number): Float32Array {
  'worklet';
  return resize(frame, {
    scale: { width: size, height: size },
    pixelFormat: 'rgb',
    dataType: 'float32',
    rotation: frameRotation(frame),
  });
}
