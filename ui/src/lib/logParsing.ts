export interface LossPoint {
  step: number;
  loss: number;
}

const LOSS_RE = /loss=([0-9.]+)\s+step=(\d+)/;

export function parseLossPoints(lines: string[]): LossPoint[] {
  const points: LossPoint[] = [];
  for (const line of lines) {
    const match = line.match(LOSS_RE);
    if (match) {
      points.push({ loss: parseFloat(match[1]), step: parseInt(match[2]) });
    }
  }
  return points;
}
