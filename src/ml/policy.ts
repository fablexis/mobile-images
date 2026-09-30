export const NSFW_CLASSES = ['drawing', 'hentai', 'neutral', 'porn', 'sexy'] as const;
export type NsfwClass = (typeof NSFW_CLASSES)[number];
export type NsfwScores = Record<NsfwClass, number>;

// Order from notAI-tech/NudeNet (branch v3) nudenet/nudenet.py `__labels`.
// It differs from the order listed in older docs; the model output follows this one.
export const NUDENET_LABELS = [
  'FEMALE_GENITALIA_COVERED',
  'FACE_FEMALE',
  'BUTTOCKS_EXPOSED',
  'FEMALE_BREAST_EXPOSED',
  'FEMALE_GENITALIA_EXPOSED',
  'MALE_BREAST_EXPOSED',
  'ANUS_EXPOSED',
  'FEET_EXPOSED',
  'BELLY_COVERED',
  'FEET_COVERED',
  'ARMPITS_COVERED',
  'ARMPITS_EXPOSED',
  'FACE_MALE',
  'BELLY_EXPOSED',
  'MALE_GENITALIA_EXPOSED',
  'ANUS_COVERED',
  'FEMALE_BREAST_COVERED',
  'BUTTOCKS_COVERED',
] as const;
export type NudeNetLabel = (typeof NUDENET_LABELS)[number];

export type Verdict = 'safe' | 'borderline' | 'nsfw';

export interface Policy {
  /** porn + hentai above this => nsfw */
  nsfwThreshold: number;
  /** sexy above this => borderline */
  borderlineThreshold: number;
  /** Minimum NudeNet class score for a detection to be kept. */
  detectionScoreThreshold: number;
  /** IoU above which overlapping NudeNet boxes are suppressed. */
  nmsIouThreshold: number;
  /** NudeNet labels whose boxes get blurred. */
  blurLabels: ReadonlySet<NudeNetLabel>;
}

const isBlurredByDefault = (label: NudeNetLabel): boolean =>
  label.endsWith('_EXPOSED') && !label.startsWith('FACE_') && label !== 'FEET_EXPOSED';

export const DEFAULT_POLICY: Policy = {
  nsfwThreshold: 0.7,
  borderlineThreshold: 0.6,
  detectionScoreThreshold: 0.25,
  nmsIouThreshold: 0.45,
  blurLabels: new Set(NUDENET_LABELS.filter(isBlurredByDefault)),
};

export function classify(scores: NsfwScores, policy: Policy = DEFAULT_POLICY): Verdict {
  if (scores.porn + scores.hentai > policy.nsfwThreshold) return 'nsfw';
  if (scores.sexy > policy.borderlineThreshold) return 'borderline';
  return 'safe';
}

export function shouldBlur(label: NudeNetLabel, policy: Policy = DEFAULT_POLICY): boolean {
  return policy.blurLabels.has(label);
}

export const VERDICT_TEXT: Record<Verdict, string> = {
  safe: 'Seguro',
  borderline: 'Revisar',
  nsfw: 'NSFW',
};

export const VERDICT_COLOR: Record<Verdict, string> = {
  safe: '#2ecc71',
  borderline: '#f5a623',
  nsfw: '#ff4d4f',
};

export const CLASS_TEXT: Record<NsfwClass, string> = {
  drawing: 'Dibujo',
  hentai: 'Hentai',
  neutral: 'Neutral',
  porn: 'Porno',
  sexy: 'Sexy',
};

export const NUDENET_TEXT: Record<NudeNetLabel, string> = {
  FEMALE_GENITALIA_COVERED: 'Genitales (cubiertos)',
  FACE_FEMALE: 'Rostro (mujer)',
  BUTTOCKS_EXPOSED: 'Glúteos',
  FEMALE_BREAST_EXPOSED: 'Pecho',
  FEMALE_GENITALIA_EXPOSED: 'Genitales',
  MALE_BREAST_EXPOSED: 'Pecho (hombre)',
  ANUS_EXPOSED: 'Ano',
  FEET_EXPOSED: 'Pies',
  BELLY_COVERED: 'Vientre (cubierto)',
  FEET_COVERED: 'Pies (cubiertos)',
  ARMPITS_COVERED: 'Axilas (cubiertas)',
  ARMPITS_EXPOSED: 'Axilas',
  FACE_MALE: 'Rostro (hombre)',
  BELLY_EXPOSED: 'Vientre',
  MALE_GENITALIA_EXPOSED: 'Genitales',
  ANUS_COVERED: 'Ano (cubierto)',
  FEMALE_BREAST_COVERED: 'Pecho (cubierto)',
  BUTTOCKS_COVERED: 'Glúteos (cubiertos)',
};
