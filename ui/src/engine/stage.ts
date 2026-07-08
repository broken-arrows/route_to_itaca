export const STAGE_W = 1512;
export const STAGE_H = 860;

export function computeScale(
  vw: number,
  vh: number,
  stageW: number = STAGE_W,
  stageH: number = STAGE_H,
): number {
  return Math.min(vw / stageW, vh / stageH);
}
