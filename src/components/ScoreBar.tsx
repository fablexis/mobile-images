import { StyleSheet, Text, View } from 'react-native';

import {
  CLASS_TEXT,
  NSFW_CLASSES,
  type NsfwScores,
  type Verdict,
  VERDICT_COLOR,
  VERDICT_TEXT,
} from '../ml/policy';

const BAR_COLOR: Record<(typeof NSFW_CLASSES)[number], string> = {
  drawing: '#8e7dff',
  hentai: '#ff7ab6',
  neutral: '#2ecc71',
  porn: '#ff4d4f',
  sexy: '#f5a623',
};

interface VerdictProps {
  verdict: Verdict | null;
  faceCount: number;
  subtitle?: string;
  compact?: boolean;
}

export function VerdictHeader({ verdict, faceCount, subtitle, compact }: VerdictProps) {
  const color = verdict ? VERDICT_COLOR[verdict] : '#6b7280';
  return (
    <View style={styles.header}>
      <Text style={[styles.verdict, compact && styles.verdictCompact, { color }]}>
        {verdict ? VERDICT_TEXT[verdict] : '—'}
      </Text>
      <View style={styles.headerRight}>
        <Text style={styles.faces}>
          {faceCount} {faceCount === 1 ? 'rostro' : 'rostros'}
        </Text>
        {subtitle ? <Text style={styles.subtitle}>{subtitle}</Text> : null}
      </View>
    </View>
  );
}

interface Props {
  scores: NsfwScores | null;
  verdict: Verdict | null;
  faceCount: number;
  subtitle?: string;
  compact?: boolean;
}

export function ScoreBar({ scores, verdict, faceCount, subtitle, compact }: Props) {
  return (
    <View style={styles.container}>
      <VerdictHeader verdict={verdict} faceCount={faceCount} subtitle={subtitle} compact={compact} />
      {NSFW_CLASSES.map((c) => {
        const value = scores?.[c] ?? 0;
        return (
          <View key={c} style={[styles.row, compact && styles.rowCompact]}>
            <Text style={styles.name}>{CLASS_TEXT[c]}</Text>
            <View style={styles.track}>
              <View
                style={[styles.fill, { width: `${Math.round(value * 100)}%`, backgroundColor: BAR_COLOR[c] }]}
              />
            </View>
            <Text style={styles.value}>{scores ? `${(value * 100).toFixed(0)}%` : '--'}</Text>
          </View>
        );
      })}
    </View>
  );
}

const styles = StyleSheet.create({
  container: { gap: 6 },
  header: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', marginBottom: 6 },
  headerRight: { alignItems: 'flex-end' },
  verdict: { fontSize: 40, fontWeight: '800', letterSpacing: 0.5 },
  verdictCompact: { fontSize: 30 },
  faces: { color: '#e5e7eb', fontSize: 16, fontWeight: '600' },
  subtitle: { color: '#9ca3af', fontSize: 12, marginTop: 2 },
  row: { flexDirection: 'row', alignItems: 'center', gap: 10, height: 22 },
  rowCompact: { height: 16 },
  name: { color: '#d1d5db', width: 64, fontSize: 13 },
  track: { flex: 1, height: 8, borderRadius: 4, backgroundColor: '#1f2430', overflow: 'hidden' },
  fill: { height: '100%', borderRadius: 4 },
  value: { color: '#9ca3af', width: 40, textAlign: 'right', fontSize: 12, fontVariant: ['tabular-nums'] },
});
