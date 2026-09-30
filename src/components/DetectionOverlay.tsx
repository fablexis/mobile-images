import {
  Blur,
  Canvas,
  Group,
  Image as SkiaImage,
  Rect,
  rect,
  type SkImage,
} from '@shopify/react-native-skia';
import { StyleSheet, Text, View } from 'react-native';

import type { Box } from '../ml/nudenet';

export const FACE_COLOR = '#4da3ff';
export const ZONE_COLOR = '#ff4d4f';
export const ZONE_INFO_COLOR = '#f5a623';

export interface OverlayZone {
  box: Box;
  label: string;
  blur: boolean;
}

interface Props {
  /** Size of the drawing area on screen. */
  width: number;
  height: number;
  /** Pixel space the boxes are expressed in (image size, or the view size for camera). */
  sourceWidth: number;
  sourceHeight: number;
  /** When provided the image is drawn (aspect fit) under the boxes and zones get blurred. */
  image?: SkImage | null;
  faces: Box[];
  zones?: OverlayZone[];
  showZoneBoxes?: boolean;
}

function fitContain(width: number, height: number, srcW: number, srcH: number) {
  const scale = Math.min(width / srcW, height / srcH);
  const drawW = srcW * scale;
  const drawH = srcH * scale;
  return { scale, offsetX: (width - drawW) / 2, offsetY: (height - drawH) / 2, drawW, drawH };
}

export function DetectionOverlay({
  width,
  height,
  sourceWidth,
  sourceHeight,
  image,
  faces,
  zones = [],
  showZoneBoxes = true,
}: Props) {
  if (width <= 0 || height <= 0 || sourceWidth <= 0 || sourceHeight <= 0) return null;

  const { scale, offsetX, offsetY, drawW, drawH } = fitContain(width, height, sourceWidth, sourceHeight);
  const toView = (b: Box) => ({
    x: offsetX + b.x * scale,
    y: offsetY + b.y * scale,
    width: b.width * scale,
    height: b.height * scale,
  });

  const blurZones = image ? zones.filter((z) => z.blur).map((z) => toView(z.box)) : [];

  return (
    <View style={{ width, height }} pointerEvents="none">
      <Canvas style={StyleSheet.absoluteFill}>
        {image ? (
          <SkiaImage image={image} x={offsetX} y={offsetY} width={drawW} height={drawH} fit="fill" />
        ) : null}

        {blurZones.map((r, i) => (
          <Group key={`blur-${i}`} clip={rect(r.x, r.y, r.width, r.height)}>
            <SkiaImage image={image!} x={offsetX} y={offsetY} width={drawW} height={drawH} fit="fill">
              <Blur blur={Math.max(14, Math.min(r.width, r.height) * 0.25)} mode="clamp" />
            </SkiaImage>
            <Rect x={r.x} y={r.y} width={r.width} height={r.height} color="rgba(0,0,0,0.25)" />
          </Group>
        ))}

        {showZoneBoxes &&
          zones.map((z, i) => {
            const r = toView(z.box);
            return (
              <Rect
                key={`zone-${i}`}
                x={r.x}
                y={r.y}
                width={r.width}
                height={r.height}
                color={z.blur ? ZONE_COLOR : ZONE_INFO_COLOR}
                style="stroke"
                strokeWidth={2}
              />
            );
          })}

        {faces.map((f, i) => {
          const r = toView(f);
          return (
            <Rect
              key={`face-${i}`}
              x={r.x}
              y={r.y}
              width={r.width}
              height={r.height}
              color={FACE_COLOR}
              style="stroke"
              strokeWidth={3}
            />
          );
        })}
      </Canvas>

      {showZoneBoxes &&
        zones.map((z, i) => {
          const r = toView(z.box);
          return (
            <Text
              key={`zone-label-${i}`}
              numberOfLines={1}
              style={[
                styles.label,
                { left: r.x, top: Math.max(0, r.y - 18), backgroundColor: z.blur ? ZONE_COLOR : ZONE_INFO_COLOR },
              ]}
            >
              {z.label}
            </Text>
          );
        })}
    </View>
  );
}

const styles = StyleSheet.create({
  label: {
    position: 'absolute',
    color: '#fff',
    fontSize: 11,
    fontWeight: '600',
    paddingHorizontal: 4,
    paddingVertical: 1,
    borderRadius: 3,
    overflow: 'hidden',
  },
});
