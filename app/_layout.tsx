import { Stack } from 'expo-router';
import { StatusBar } from 'expo-status-bar';

import { colors } from '../src/theme';

export default function RootLayout() {
  return (
    <>
      <StatusBar style="light" />
      <Stack
        screenOptions={{
          headerStyle: { backgroundColor: colors.background },
          headerTintColor: colors.text,
          headerTitleStyle: { fontWeight: '700' },
          contentStyle: { backgroundColor: colors.background },
          headerBackButtonDisplayMode: 'minimal',
        }}
      >
        <Stack.Screen name="index" options={{ title: 'Detector on-device' }} />
        <Stack.Screen name="gallery" options={{ title: 'Galería' }} />
        <Stack.Screen name="camera" options={{ title: 'Cámara en vivo', headerTransparent: true }} />
      </Stack>
    </>
  );
}
