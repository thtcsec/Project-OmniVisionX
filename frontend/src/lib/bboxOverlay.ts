/** Map source-frame bbox (pixels) to % positions inside a box that uses CSS object-contain for video. */
export function mapBboxToObjectContainPercent(
  x1: number,
  y1: number,
  x2: number,
  y2: number,
  srcW: number,
  srcH: number,
  containerW: number,
  containerH: number,
): { x: number; y: number; w: number; h: number } | null {
  if (srcW <= 0 || srcH <= 0 || containerW <= 0 || containerH <= 0) return null;
  const scale = Math.min(containerW / srcW, containerH / srcH);
  const dw = srcW * scale;
  const dh = srcH * scale;
  const offX = (containerW - dw) / 2;
  const offY = (containerH - dh) / 2;
  const left = offX + x1 * scale;
  const top = offY + y1 * scale;
  const width = (x2 - x1) * scale;
  const height = (y2 - y1) * scale;
  return {
    x: (left / containerW) * 100,
    y: (top / containerH) * 100,
    w: (width / containerW) * 100,
    h: (height / containerH) * 100,
  };
}
