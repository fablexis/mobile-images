import { Link } from 'expo-router';
import { Pressable, StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { colors } from '../src/theme';

export default function Home() {
  return (
    <SafeAreaView style={styles.container} edges={['bottom']}>
      <View style={styles.hero}>
        <Text style={styles.title}>Rostros y contenido sensible</Text>
        <Text style={styles.body}>
          Todo el análisis ocurre en tu dispositivo. Ninguna imagen sale del teléfono.
        </Text>
      </View>

      <View style={styles.actions}>
        <Link href="/gallery" asChild>
          <Pressable style={({ pressed }) => [styles.button, pressed && styles.pressed]}>
            <Text style={styles.buttonTitle}>Galería</Text>
            <Text style={styles.buttonBody}>Analiza una foto y desenfoca las zonas expuestas</Text>
          </Pressable>
        </Link>

        <Link href="/camera" asChild>
          <Pressable style={({ pressed }) => [styles.button, styles.buttonAlt, pressed && styles.pressed]}>
            <Text style={styles.buttonTitle}>Cámara</Text>
            <Text style={styles.buttonBody}>Detección de rostros y alerta NSFW en tiempo real</Text>
          </Pressable>
        </Link>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: colors.background, padding: 20, justifyContent: 'space-between' },
  hero: { marginTop: 24, gap: 10 },
  title: { color: colors.text, fontSize: 30, fontWeight: '800' },
  body: { color: colors.muted, fontSize: 16, lineHeight: 22 },
  actions: { gap: 14, marginBottom: 12 },
  button: {
    backgroundColor: colors.surface,
    borderColor: colors.border,
    borderWidth: 1,
    borderRadius: 16,
    padding: 20,
    gap: 6,
  },
  buttonAlt: { borderColor: colors.accent },
  pressed: { opacity: 0.7 },
  buttonTitle: { color: colors.text, fontSize: 22, fontWeight: '700' },
  buttonBody: { color: colors.muted, fontSize: 14 },
});
