export type CalibrationData = {
  pixelTopLeft: [number, number]
  pixelBottomRight: [number, number]
  worldTopLeft: [number, number]
  worldBottomRight: [number, number]
}

export function pixelToWorld(px: number, py: number, cal: CalibrationData): [number, number] {
  const { pixelTopLeft, pixelBottomRight, worldTopLeft, worldBottomRight } = cal
  const scaleX = (worldBottomRight[0] - worldTopLeft[0]) / (pixelBottomRight[0] - pixelTopLeft[0])
  const scaleY = (worldBottomRight[1] - worldTopLeft[1]) / (pixelBottomRight[1] - pixelTopLeft[1])
  return [
    worldTopLeft[0] + (px - pixelTopLeft[0]) * scaleX,
    worldTopLeft[1] + (py - pixelTopLeft[1]) * scaleY,
  ]
}

export function worldToPixel(wx: number, wy: number, cal: CalibrationData): [number, number] {
  const { pixelTopLeft, pixelBottomRight, worldTopLeft, worldBottomRight } = cal
  const scaleX = (pixelBottomRight[0] - pixelTopLeft[0]) / (worldBottomRight[0] - worldTopLeft[0])
  const scaleY = (pixelBottomRight[1] - pixelTopLeft[1]) / (worldBottomRight[1] - worldTopLeft[1])
  return [
    pixelTopLeft[0] + (wx - worldTopLeft[0]) * scaleX,
    pixelTopLeft[1] + (wy - worldTopLeft[1]) * scaleY,
  ]
}
