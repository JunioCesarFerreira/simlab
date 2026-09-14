import { computeRanksWithDuplicates, type SortablePoint } from "../utils/nonDominatedSort";

self.onmessage = (event: MessageEvent<{ points: SortablePoint[]; minimize: boolean[] }>) => {
  const { points, minimize } = event.data;
  self.postMessage(computeRanksWithDuplicates(points, minimize));
};
