"""
Region partition and trajectory coverage constraint for Problem P1.

Divides Ω = [xmin,xmax]×[ymin,ymax] into 8 triangular regions W1..W8
using the centre point (mx, my) and the two main diagonals of the rectangle.

                W1    |    W3
            ----------+----------
                W2    |    W4
            -------(mx,my)--------
                W8    |    W5
            ----------+----------
                W7    |    W6

The main benefit over a brute-force O(N×m) distance scan is spatial pruning:
each trajectory point w_j only needs to be checked against positions that
fall in the regions within radius R of w_j, not all positions in Ω.

All heavy structures (triangle vertices, bounding boxes, reachable-region
lists per trajectory point) are computed once at problem-instance build time
and reused across every chromosome evaluation in the evolutionary loop.
"""
import math
import numpy as np
from typing import Sequence

from lib.util.trajectory_sampling import sample_trajectories

Point2D = tuple[float, float]

# Canonical ordering of the 8 regions (must stay stable — used as dict keys)
_REGION_ORDER: tuple[str, ...] = ("W1", "W2", "W3", "W4", "W5", "W6", "W7", "W8")


# ──────────────────────────────────────────────────────────────────────────────
# Low-level geometry (scalar, no allocations)
# ──────────────────────────────────────────────────────────────────────────────

def _point_in_triangle(px: float, py: float, tri: np.ndarray, eps: float = 1e-12) -> bool:
    """
    Orientation (cross-product) test.  Returns True when (px, py) is inside
    or on the boundary of triangle `tri` (shape 3×2).
    """
    (ax, ay), (bx, by), (cx, cy) = tri[0], tri[1], tri[2]
    c1 = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
    c2 = (cx - bx) * (py - by) - (cy - by) * (px - bx)
    c3 = (ax - cx) * (py - cy) - (ay - cy) * (px - cx)
    has_neg = (c1 < -eps) or (c2 < -eps) or (c3 < -eps)
    has_pos = (c1 >  eps) or (c2 >  eps) or (c3 >  eps)
    return not (has_neg and has_pos)


def _segment_dist_sq(px: float, py: float,
                     ax: float, ay: float, bx: float, by: float) -> float:
    """Squared distance from (px, py) to segment AB.  No sqrt."""
    abx, aby = bx - ax, by - ay
    ab2 = abx * abx + aby * aby
    if ab2 == 0.0:
        return (px - ax) ** 2 + (py - ay) ** 2
    t = max(0.0, min(1.0, ((px - ax) * abx + (py - ay) * aby) / ab2))
    ex, ey = ax + t * abx - px, ay + t * aby - py
    return ex * ex + ey * ey


def _triangle_distance(px: float, py: float, tri: np.ndarray) -> float:
    """Exact distance from (px, py) to triangle `tri`.  Returns 0 if inside."""
    if _point_in_triangle(px, py, tri):
        return 0.0
    (ax, ay), (bx, by), (cx, cy) = tri[0], tri[1], tri[2]
    d1 = _segment_dist_sq(px, py, ax, ay, bx, by)
    d2 = _segment_dist_sq(px, py, bx, by, cx, cy)
    d3 = _segment_dist_sq(px, py, cx, cy, ax, ay)
    return math.sqrt(min(d1, d2, d3))


# ──────────────────────────────────────────────────────────────────────────────
# Region partition
# ──────────────────────────────────────────────────────────────────────────────

class RegionPartition:
    """
    Partition of Ω = [xmin,xmax]×[ymin,ymax] into 8 triangular sub-regions.

    The partition uses the centre (mx, my) and the two diagonals of the
    rectangle as boundaries.  Each triangle is labelled W1..W8 following
    the counter-clockwise ordering from the upper-left corner.

    All triangle data and bounding boxes are pre-computed in __init__ and
    reused on every call to `classify` or `reachable_regions`.

    Parameters
    ----------
    xmin, xmax, ymin, ymax : float
        Bounds of the rectangular region Ω.
    """

    def __init__(
        self,
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float,
    ) -> None:
        a, b = xmin, xmax   # x bounds
        c, d = ymin, ymax   # y bounds
        mx = (a + b) / 2.0
        my = (c + d) / 2.0
        self._mx, self._my = mx, my
        if b - a == 0.0:
            raise ValueError(f"Invalid region: zero width (xmin={a}, xmax={b})")
        if d - c == 0.0:
            raise ValueError(f"Invalid region: zero height (ymin={c}, ymax={d})")
        self._lam = (d - c) / (b - a)   # diagonal slope

        self._regions: dict[str, np.ndarray] = {
            "W1": np.array([[a, my], [mx, d], [a, d]], dtype=float),
            "W2": np.array([[a, my], [mx, d], [mx, my]], dtype=float),
            "W3": np.array([[mx, d], [b, my], [b, d]], dtype=float),
            "W4": np.array([[mx, d], [b, my], [mx, my]], dtype=float),
            "W5": np.array([[mx, my], [b, my], [mx, c]], dtype=float),
            "W6": np.array([[mx, c], [b, my], [b, c]], dtype=float),
            "W7": np.array([[a, c], [a, my], [mx, c]], dtype=float),
            "W8": np.array([[a, my], [mx, my], [mx, c]], dtype=float),
        }

        # Pre-computed axis-aligned bounding boxes for fast rejection
        self._boxes: dict[str, tuple[float, float, float, float]] = {
            name: (
                float(np.min(tri[:, 0])), float(np.max(tri[:, 0])),
                float(np.min(tri[:, 1])), float(np.max(tri[:, 1])),
            )
            for name, tri in self._regions.items()
        }

    def classify(self, px: float, py: float) -> str:
        """
        Classify (px, py) into one of W1..W8 in O(1).

        Uses sign comparisons on quadrant membership and the diagonal
        distance `r = |dy| − λ|dx|` to distinguish inner from outer
        triangles without testing all 8 triangles explicitly.
        """
        dx = px - self._mx
        dy = py - self._my
        r  = abs(dy) - self._lam * abs(dx)

        if dx <= 0 and dy >= 0:      # upper-left quadrant
            return "W1" if r >= 0 else "W2"
        if dx >= 0 and dy >= 0:      # upper-right quadrant
            return "W3" if r >= 0 else "W4"
        if dx >= 0 and dy <= 0:      # lower-right quadrant
            return "W6" if r <= 0 else "W5"
        # lower-left quadrant
        return "W7" if r <= 0 else "W8"

    def reachable_regions(self, px: float, py: float, R: float) -> list[str]:
        """
        Return the sub-regions whose triangles intersect the disk B(px, py, R).

        Uses bounding-box rejection first (O(1) per region) and falls back
        to exact triangle distance only when the bbox check passes.

        Parameters
        ----------
        px, py : float
            Query point.
        R : float
            Radius of the disk.

        Returns
        -------
        list[str]
            Names of the regions (subset of W1..W8) reachable from (px, py).
        """
        result: list[str] = []
        for name in _REGION_ORDER:
            bxmin, bxmax, bymin, bymax = self._boxes[name]
            if px < bxmin - R or px > bxmax + R or py < bymin - R or py > bymax + R:
                continue
            if _triangle_distance(px, py, self._regions[name]) <= R:
                result.append(name)
        return result


# ──────────────────────────────────────────────────────────────────────────────
# Trajectory coverage constraint for P1
# ──────────────────────────────────────────────────────────────────────────────

class TrajectoryConstraintP1:
    """
    Discrete trajectory coverage constraint for Problem P1 using region-based
    spatial pruning (sampled trajectory coverage).

    Unlike P2 (which uses a pre-built binary matrix over a fixed candidate set),
    P1 chromosomes carry continuous relay positions that change every generation.
    The matrix approach is therefore impractical.  Instead:

    - At build time (once per problem instance):
        * Sample trajectory points  W  with arc-length step  R/2.
        * For each  w_j ∈ W  pre-compute  reachable[j]: the sub-regions of Ω
          whose triangles intersect  B(w_j, R).  Only positions classified into
          those regions can possibly cover  w_j.

    - At evaluation time (once per chromosome):
        * Classify every active position (sink + relays) into its W_i region.
        * For each  w_j, scan only the positions in  reachable[j]  and check
          exact distance.  Expected scan per point: O(|reachable| × |Ω/8|)
          instead of O(m) for a brute-force pass.

    The score is a float in [0, 100] (percentage of covered points):
        100 → every trajectory point covered
        0   → no trajectory point covered

    Parameters
    ----------
    sink : Point2D
        Sink position (always active).
    mobile_nodes : list[MobileNode]
        Mobile nodes whose trajectories define the coverage requirement.
    R : float
        Communication radius.
    region : list[float]
        Bounding box [xmin, ymin, xmax, ymax] of Ω.
    """

    def __init__(
        self,
        sink: Point2D,
        mobile_nodes,
        R: float,
        region: Sequence[float],
    ) -> None:
        xmin, ymin, xmax, ymax = region   # format: [xmin, ymin, xmax, ymax]
        self._partition = RegionPartition(xmin, xmax, ymin, ymax)
        self._R_sq = R * R

        # Sample trajectories once (step = R/2 → no gap > R/2 along path)
        W: list[Point2D] = sample_trajectories(mobile_nodes, step=R / 2)
        self._W = W
        self._N = len(W)

        # Pre-compute reachable regions for each trajectory point
        self._reachable: list[list[str]] = [
            self._partition.reachable_regions(px, py, R)
            for px, py in W
        ]

        # Sink is always active — include it in every evaluation
        self._sink = sink

    # ------------------------------------------------------------------
    @property
    def has_points(self) -> bool:
        """True when W is non-empty; False means constraint is vacuous."""
        return self._N > 0

    @property
    def n_points(self) -> int:
        """Number of sampled trajectory points."""
        return self._N

    @property
    def sampled_points(self) -> tuple[Point2D, ...]:
        """Sampled trajectory points used by the discrete coverage check."""
        return tuple(self._W)

    # ------------------------------------------------------------------
    def coverage_flags(self, relay_positions: Sequence[Point2D]) -> list[bool]:
        """
        Return one boolean per sampled point indicating current coverage.

        This deliberately uses a direct distance scan. It is intended for
        low-budget repair heuristics that need point-level feedback, while
        ``check_coverage`` remains the optimized scoring path.
        """
        if not self.has_points:
            return []

        active_positions = [self._sink, *relay_positions]
        R_sq = self._R_sq
        flags: list[bool] = []

        for wx, wy in self._W:
            covered = False
            for px, py in active_positions:
                dx, dy = wx - px, wy - py
                if dx * dx + dy * dy <= R_sq:
                    covered = True
                    break
            flags.append(covered)

        return flags

    def uncovered_points(self, relay_positions: Sequence[Point2D]) -> list[Point2D]:
        """Return sampled trajectory points not covered by sink + relays."""
        flags = self.coverage_flags(relay_positions)
        return [point for point, covered in zip(self._W, flags) if not covered]

    def relay_exclusive_cover_counts(self, relay_positions: Sequence[Point2D]) -> list[int]:
        """
        Count sampled points covered by exactly one relay and not by the sink.

        A low count marks a relay as a good candidate for repositioning during
        coverage repair, because fewer currently covered points depend only on it.
        """
        counts = [0] * len(relay_positions)
        if not self.has_points or not relay_positions:
            return counts

        sx, sy = self._sink
        R_sq = self._R_sq

        for wx, wy in self._W:
            dx, dy = wx - sx, wy - sy
            if dx * dx + dy * dy <= R_sq:
                continue

            covering_relays: list[int] = []
            for i, (px, py) in enumerate(relay_positions):
                dx, dy = wx - px, wy - py
                if dx * dx + dy * dy <= R_sq:
                    covering_relays.append(i)
                    if len(covering_relays) > 1:
                        break

            if len(covering_relays) == 1:
                counts[covering_relays[0]] += 1

        return counts

    # ------------------------------------------------------------------
    def check_coverage(self, relay_positions: Sequence[Point2D]) -> float:
        """
        Compute the discrete trajectory coverage score for the given
        relay positions (chromosome genes).

        The sink is automatically included as an always-active position.

        Parameters
        ----------
        relay_positions : Sequence[Point2D]
            Relay positions from the chromosome (continuous 2D points).

        Returns
        -------
        float
            Coverage score in [0, 100] (percentage of covered points).
            100 → every sampled trajectory point is within R of at least
                  one active element (sink or relay).
            0   → no trajectory point is covered.
        """
        if not self.has_points:
            return 100.0

        # Group active positions by W_i region.
        # A position on a shared boundary belongs to multiple triangles, so we
        # use reachable_regions(_, _, 0) to collect every region that contains
        # it; otherwise a relay at the partition boundary could be missed by a
        # trajectory point whose reachable set happens to exclude the single
        # region classify() picked.
        by_region: dict[str, list[Point2D]] = {name: [] for name in _REGION_ORDER}
        sx, sy = self._sink
        for rname in self._partition.reachable_regions(sx, sy, 0.0):
            by_region[rname].append(self._sink)
        for pos in relay_positions:
            px, py = pos
            for rname in self._partition.reachable_regions(px, py, 0.0):
                by_region[rname].append(pos)

        R_sq = self._R_sq
        covered = 0

        for j in range(self._N):
            wx, wy = self._W[j]
            point_covered = False

            for rname in self._reachable[j]:
                for (px, py) in by_region[rname]:
                    dx, dy = wx - px, wy - py
                    if dx * dx + dy * dy <= R_sq:
                        point_covered = True
                        break
                if point_covered:
                    break

            if point_covered:
                covered += 1

        return covered / self._N * 100.0
