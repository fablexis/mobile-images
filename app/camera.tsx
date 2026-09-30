import { useFocusEffect } from 'expo-router';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  ActivityIndicator,
  AppState,
  Linking,
  Pressable,
  StyleSheet,
  Text,
  View,
  type LayoutChangeEvent,
} from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import {
  Camera,
  runAtTargetFps,
  useCameraDevice,
  useCameraPermission,
  useFrameProcessor,
} from 'react-native-vision-camera';
import { type FrameFaceDetectionOptions, useFaceDetector } from 'react-native-vision-camera-face-detector';
import { useResizePlugin } from 'vision-camera-resize-plugin';
import { useRunOnJS } from 'react-native-worklets-core';

import { DetectionOverlay } from '../src/components/DetectionOverlay';
import { ScoreBar } from '../src/components/ScoreBar';
import type { Box } from '../src/ml/nudenet';
import { toScores, useNsfwClassifier } from '../src/ml/nsfwClassifier';
import { classify, type NsfwScores } from '../src/ml/policy';
import { frameToTensor, NSFW_INPUT_SIZE } from '../src/ml/preprocess';
import { colors } from '../src/theme';

const CLASSIFIER_FPS = 3;

export default function CameraScreen() {
  const { hasPermission, requestPermission } = useCameraPermission();
  const [asked, setAsked] = useState(false);

  useEffect(() => {
    if (!hasPermission && !asked) {
      setAsked(true);
      requestPermission();
    }
  }, [hasPermission, asked, requestPermission]);

  if (!hasPermission) {
    return (
      <View style={styles.center}>
        <Text style={styles.title}>Permiso de cámara requerido</Text>
        <Text style={styles.muted}>
          La detección en vivo necesita la cámara. Las imágenes se procesan en el dispositivo y no se guardan.
        </Text>
        <Pressable style={styles.button} onPress={() => requestPermission()}>
          <Text style={styles.buttonText}>Conceder permiso</Text>
        </Pressable>
        <Pressable style={[styles.button, styles.secondary]} onPress={() => Linking.openSettings()}>
          <Text style={styles.buttonText}>Abrir ajustes</Text>
        </Pressable>
      </View>
    );
  }

  return <LiveCamera />;
}

function LiveCamera() {
  const insets = useSafeAreaInsets();
  const [position, setPosition] = useState<'back' | 'front'>('back');
  const device = useCameraDevice(position);

  const [focused, setFocused] = useState(true);
  const [appActive, setAppActive] = useState(AppState.currentState === 'active');
  useFocusEffect(
    useCallback(() => {
      setFocused(true);
      return () => setFocused(false);
    }, []),
  );
  useEffect(() => {
    const sub = AppState.addEventListener('change', (s) => setAppActive(s === 'active'));
    return () => sub.remove();
  }, []);

  const [layout, setLayout] = useState({ width: 0, height: 0 });
  const onLayout = (e: LayoutChangeEvent) => {
    const { width, height } = e.nativeEvent.layout;
    setLayout({ width, height });
  };

  const [faces, setFaces] = useState<Box[]>([]);
  const [scores, setScores] = useState<NsfwScores | null>(null);
  const [classifierFps, setClassifierFps] = useState(0);
  const lastScoreAt = useRef(0);

  const classifier = useNsfwClassifier();
  const model = classifier.model;

  // autoMode makes the plugin return bounds already scaled/rotated/mirrored to
  // the preview's size, so the overlay can draw them 1:1.
  const faceOptions = useMemo<FrameFaceDetectionOptions>(
    () => ({
      performanceMode: 'fast',
      landmarkMode: 'none',
      contourMode: 'none',
      classificationMode: 'none',
      minFaceSize: 0.1,
      autoMode: true,
      windowWidth: layout.width || 1,
      windowHeight: layout.height || 1,
      cameraFacing: position,
    }),
    [layout.width, layout.height, position],
  );
  const faceDetector = useFaceDetector(faceOptions);
  const { detectFaces } = faceDetector;
  const { resize } = useResizePlugin();

  useEffect(() => () => faceDetector.stopListeners(), [faceDetector]);

  const onFaces = useRunOnJS((next: Box[]) => setFaces(next), []);
  const onScores = useRunOnJS((raw: number[]) => {
    const now = Date.now();
    if (lastScoreAt.current > 0) setClassifierFps(1000 / (now - lastScoreAt.current));
    lastScoreAt.current = now;
    setScores(toScores(raw));
  }, []);

  const frameProcessor = useFrameProcessor(
    (frame) => {
      'worklet';
      const detected = detectFaces(frame);
      const boxes: Box[] = [];
      for (let i = 0; i < detected.length; i++) {
        const b = detected[i].bounds;
        boxes.push({ x: b.x, y: b.y, width: b.width, height: b.height });
      }
      onFaces(boxes);

      if (model == null) return;
      runAtTargetFps(CLASSIFIER_FPS, () => {
        'worklet';
        const input = frameToTensor(frame, resize, NSFW_INPUT_SIZE);
        const out = model.runSync([input])[0];
        onScores([out[0] as number, out[1] as number, out[2] as number, out[3] as number, out[4] as number]);
      });
    },
    [detectFaces, resize, model, onFaces, onScores],
  );

  const verdict = scores ? classify(scores) : null;
  const isActive = focused && appActive;

  if (device == null) {
    return (
      <View style={styles.center}>
        <Text style={styles.title}>No se encontró una cámara</Text>
        <Text style={styles.muted}>Este dispositivo no tiene una cámara {position === 'back' ? 'trasera' : 'frontal'}.</Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <View style={StyleSheet.absoluteFill} onLayout={onLayout}>
        <Camera
          style={StyleSheet.absoluteFill}
          device={device}
          isActive={isActive}
          frameProcessor={frameProcessor}
          pixelFormat="yuv"
          resizeMode="cover"
          photo={false}
          video={false}
          audio={false}
        />
        <DetectionOverlay
          width={layout.width}
          height={layout.height}
          sourceWidth={layout.width}
          sourceHeight={layout.height}
          faces={faces}
        />
      </View>

      <View style={[styles.panel, { paddingBottom: insets.bottom + 12 }]}>
        {classifier.state === 'loading' && (
          <View style={styles.inline}>
            <ActivityIndicator color={colors.accent} />
            <Text style={styles.muted}>Cargando clasificador…</Text>
          </View>
        )}
        {classifier.state === 'error' && (
          <Text style={[styles.muted, { color: colors.danger }]}>
            No se pudo cargar el clasificador: {classifier.error.message}
          </Text>
        )}
        <ScoreBar
          scores={scores}
          verdict={verdict}
          faceCount={faces.length}
          compact
          subtitle={
            classifier.state === 'loaded'
              ? `${classifier.delegate} · ${classifierFps > 0 ? classifierFps.toFixed(1) : '--'} inf/s`
              : undefined
          }
        />
        <Pressable
          style={[styles.button, styles.secondary]}
          onPress={() => setPosition((p) => (p === 'back' ? 'front' : 'back'))}
        >
          <Text style={styles.buttonText}>Cambiar a cámara {position === 'back' ? 'frontal' : 'trasera'}</Text>
        </Pressable>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#000' },
  center: {
    flex: 1,
    backgroundColor: colors.background,
    alignItems: 'center',
    justifyContent: 'center',
    padding: 24,
    gap: 12,
  },
  title: { color: colors.text, fontSize: 20, fontWeight: '700', textAlign: 'center' },
  muted: { color: colors.muted, fontSize: 14, textAlign: 'center' },
  button: {
    backgroundColor: colors.accent,
    borderRadius: 12,
    paddingVertical: 12,
    paddingHorizontal: 20,
    alignItems: 'center',
    alignSelf: 'stretch',
  },
  secondary: { backgroundColor: 'rgba(255,255,255,0.12)' },
  buttonText: { color: '#fff', fontSize: 15, fontWeight: '700' },
  panel: {
    position: 'absolute',
    left: 0,
    right: 0,
    bottom: 0,
    paddingHorizontal: 16,
    paddingTop: 14,
    gap: 10,
    backgroundColor: 'rgba(11,13,18,0.82)',
    borderTopLeftRadius: 18,
    borderTopRightRadius: 18,
  },
  inline: { flexDirection: 'row', alignItems: 'center', gap: 8, justifyContent: 'center' },
});
