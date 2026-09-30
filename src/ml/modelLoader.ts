import { useEffect, useState } from 'react';
import { Platform } from 'react-native';
import {
  loadTensorflowModel,
  type TensorflowModel,
  type TensorflowModelDelegate,
} from 'react-native-fast-tflite';

export type ModelState =
  | { state: 'loading'; model: undefined }
  | { state: 'loaded'; model: TensorflowModel; delegate: TensorflowModelDelegate }
  | { state: 'error'; model: undefined; error: Error };

const PREFERRED_DELEGATE: TensorflowModelDelegate = Platform.select({
  ios: 'core-ml',
  android: 'android-gpu',
  default: 'default',
});

/**
 * Loads a bundled .tflite with the platform GPU delegate (CoreML / Android GPU),
 * falling back to the CPU delegate when the accelerator rejects the graph.
 * Same contract as fast-tflite's `useTensorflowModel`, plus the fallback.
 */
export function useModelWithFallback(source: number): ModelState {
  const [state, setState] = useState<ModelState>({ state: 'loading', model: undefined });

  useEffect(() => {
    let cancelled = false;
    (async () => {
      const delegates: TensorflowModelDelegate[] =
        PREFERRED_DELEGATE === 'default' ? ['default'] : [PREFERRED_DELEGATE, 'default'];
      let lastError: Error = new Error('No delegate available');
      for (const delegate of delegates) {
        try {
          const model = await loadTensorflowModel(source, delegate);
          if (!cancelled) setState({ state: 'loaded', model, delegate });
          return;
        } catch (e) {
          lastError = e instanceof Error ? e : new Error(String(e));
          console.warn(`[tflite] delegate "${delegate}" failed, trying next`, lastError.message);
        }
      }
      if (!cancelled) setState({ state: 'error', model: undefined, error: lastError });
    })();
    return () => {
      cancelled = true;
    };
  }, [source]);

  return state;
}
