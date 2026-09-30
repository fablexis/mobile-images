import * as ImagePicker from 'expo-image-picker';
import { useCallback, useState } from 'react';
import {
  ActivityIndicator,
  Linking,
  Pressable,
  ScrollView,
  StyleSheet,
  Switch,
  Text,
  useWindowDimensions,
  View,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { DetectionOverlay, type OverlayZone } from '../src/components/DetectionOverlay';
import { ScoreBar } from '../src/components/ScoreBar';
import { useNsfwClassifier } from '../src/ml/nsfwClassifier';
import { useNudeNet } from '../src/ml/nudenet';
import { analyzeGalleryImage, type GalleryAnalysis } from '../src/ml/pipeline';
import { NUDENET_TEXT } from '../src/ml/policy';
import { colors } from '../src/theme';

type Status =
  | { kind: 'idle' }
  | { kind: 'denied'; canAskAgain: boolean }
  | { kind: 'working'; step: string }
  | { kind: 'done'; result: GalleryAnalysis }
  | { kind: 'error'; message: string };

export default function GalleryScreen() {
  const classifier = useNsfwClassifier();
  const nudenet = useNudeNet();
  const [status, setStatus] = useState<Status>({ kind: 'idle' });
  const [showBoxes, setShowBoxes] = useState(true);
  const { width: windowWidth } = useWindowDimensions();

  const modelsReady = classifier.state === 'loaded';
  const busy = status.kind === 'working';

  const pick = useCallback(async () => {
    if (classifier.state !== 'loaded') return;

    const perm = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (!perm.granted) {
      setStatus({ kind: 'denied', canAskAgain: perm.canAskAgain });
      return;
    }

    const picked = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ['images'],
      quality: 1,
      exif: false,
    });
    if (picked.canceled || !picked.assets?.length) return;

    const asset = picked.assets[0];
    try {
      const result = await analyzeGalleryImage(
        asset,
        { classifier: classifier.model, nudenet: nudenet.model },
        (step) => setStatus({ kind: 'working', step }),
      );
      setStatus({ kind: 'done', result });
    } catch (e) {
      console.error(e);
      setStatus({ kind: 'error', message: e instanceof Error ? e.message : String(e) });
    }
  }, [classifier, nudenet]);

  const imageBoxWidth = windowWidth - 32;

  return (
    <SafeAreaView style={styles.container} edges={['bottom']}>
      <ScrollView contentContainerStyle={styles.content}>
        <ModelStatus
          classifier={classifier.state}
          nudenet={nudenet.state}
          errorText={
            classifier.state === 'error'
              ? classifier.error.message
              : nudenet.state === 'error'
                ? nudenet.error.message
                : undefined
          }
        />

        <Pressable
          onPress={pick}
          disabled={!modelsReady || busy}
          style={({ pressed }) => [styles.button, (!modelsReady || busy) && styles.disabled, pressed && styles.pressed]}
        >
          <Text style={styles.buttonText}>
            {status.kind === 'done' ? 'Elegir otra foto' : 'Elegir foto de la galería'}
          </Text>
        </Pressable>

        {status.kind === 'denied' && (
          <View style={styles.card}>
            <Text style={styles.cardTitle}>Permiso de fotos denegado</Text>
            <Text style={styles.muted}>
              Necesitamos acceso a tus fotos para analizarlas. El análisis se hace en el dispositivo.
            </Text>
            {!status.canAskAgain && (
              <Pressable onPress={() => Linking.openSettings()} style={[styles.button, styles.secondary]}>
                <Text style={styles.buttonText}>Abrir ajustes</Text>
              </Pressable>
            )}
          </View>
        )}

        {status.kind === 'working' && (
          <View style={styles.card}>
            <ActivityIndicator color={colors.accent} />
            <Text style={styles.muted}>{status.step}</Text>
          </View>
        )}

        {status.kind === 'error' && (
          <View style={styles.card}>
            <Text style={[styles.cardTitle, { color: colors.danger }]}>Error al analizar</Text>
            <Text style={styles.muted}>{status.message}</Text>
          </View>
        )}

        {status.kind === 'done' && (
          <Result
            result={status.result}
            boxWidth={imageBoxWidth}
            showBoxes={showBoxes}
            onToggleBoxes={setShowBoxes}
            nudenetAvailable={nudenet.state === 'loaded'}
          />
        )}
      </ScrollView>
    </SafeAreaView>
  );
}

function ModelStatus({
  classifier,
  nudenet,
  errorText,
}: {
  classifier: 'loading' | 'loaded' | 'error';
  nudenet: 'loading' | 'loaded' | 'error';
  errorText?: string;
}) {
  if (classifier === 'loaded' && nudenet === 'loaded') return null;
  const label = (s: typeof classifier) => (s === 'loading' ? 'cargando…' : s === 'loaded' ? 'listo' : 'error');
  return (
    <View style={styles.card}>
      {(classifier === 'loading' || nudenet === 'loading') && <ActivityIndicator color={colors.accent} />}
      <Text style={styles.muted}>Clasificador: {label(classifier)}</Text>
      <Text style={styles.muted}>NudeNet: {label(nudenet)}</Text>
      {errorText ? <Text style={[styles.muted, { color: colors.danger }]}>{errorText}</Text> : null}
    </View>
  );
}

function Result({
  result,
  boxWidth,
  showBoxes,
  onToggleBoxes,
  nudenetAvailable,
}: {
  result: GalleryAnalysis;
  boxWidth: number;
  showBoxes: boolean;
  onToggleBoxes: (v: boolean) => void;
  nudenetAvailable: boolean;
}) {
  const aspect = result.width / result.height;
  const boxHeight = Math.min(boxWidth / aspect, boxWidth * 1.4);

  const zones: OverlayZone[] = (result.detections ?? []).map((d) => ({
    box: d.box,
    label: `${NUDENET_TEXT[d.label]} ${(d.score * 100).toFixed(0)}%`,
    blur: d.blur,
  }));
  const blurred = zones.filter((z) => z.blur).length;

  let zonesText: string;
  if (result.detections === null) {
    zonesText = result.verdict === 'safe' ? 'NudeNet no fue necesario' : 'NudeNet no disponible';
  } else {
    zonesText = blurred > 0 ? `${blurred} zona(s) desenfocada(s)` : 'Sin zonas expuestas';
  }
  if (!nudenetAvailable && result.verdict !== 'safe') zonesText = 'NudeNet no cargado';

  const t = result.timingsMs;
  const timing = `prep ${t.prepare} ms · rostros ${t.faces} ms · clasif ${t.classifier} ms${
    t.nudenet != null ? ` · NudeNet ${t.nudenet} ms` : ''
  }`;

  return (
    <View style={styles.result}>
      <View style={[styles.imageBox, { width: boxWidth, height: boxHeight }]}>
        <DetectionOverlay
          width={boxWidth}
          height={boxHeight}
          sourceWidth={result.width}
          sourceHeight={result.height}
          image={result.image}
          faces={showBoxes ? result.faces : []}
          zones={zones}
          showZoneBoxes={showBoxes}
        />
      </View>

      <View style={styles.row}>
        <Text style={styles.muted}>Mostrar cajas</Text>
        <Switch value={showBoxes} onValueChange={onToggleBoxes} />
      </View>

      <View style={styles.card}>
        <ScoreBar scores={result.scores} verdict={result.verdict} faceCount={result.faces.length} subtitle={zonesText} />
      </View>
      <Text style={styles.timing}>{timing}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: colors.background },
  content: { padding: 16, gap: 14 },
  button: {
    backgroundColor: colors.accent,
    borderRadius: 12,
    paddingVertical: 14,
    alignItems: 'center',
  },
  secondary: { backgroundColor: colors.border, marginTop: 8 },
  disabled: { opacity: 0.4 },
  pressed: { opacity: 0.75 },
  buttonText: { color: '#fff', fontSize: 16, fontWeight: '700' },
  card: {
    backgroundColor: colors.surface,
    borderColor: colors.border,
    borderWidth: 1,
    borderRadius: 14,
    padding: 16,
    gap: 8,
  },
  cardTitle: { color: colors.text, fontSize: 16, fontWeight: '700' },
  muted: { color: colors.muted, fontSize: 14 },
  result: { gap: 12 },
  imageBox: { borderRadius: 12, overflow: 'hidden', backgroundColor: '#000' },
  row: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' },
  timing: { color: '#6b7280', fontSize: 11, textAlign: 'center' },
});
