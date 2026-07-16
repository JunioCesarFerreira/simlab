"""
Parametric trajectory sampling for Problems P1 and P2.

Each trajectory is described by one or more parametric segments (x(t), y(t))
with t in [0, 1]. The sampling function evaluates these expressions at
uniformly spaced t values, where the number of points is derived from the
arc length of the segment divided by a caller-supplied step parameter.

Arc length is estimated numerically using a fine reference grid of
_ARC_REF_SAMPLES points before the actual sampling is performed.
"""
import math
import numpy as np
from pylib.config.problems import MobileNode

# Each row is a bitset encoded as a plain Python int (arbitrary precision).
# Bit j of row i is set iff candidate j is within radius R of sample point i.
CoverageMatrix = list[int]

Point2D = tuple[float, float]

# Number of points used for the internal arc-length estimate.
# Higher values give a more accurate length but add negligible overhead.
_ARC_REF_SAMPLES = 500


def _arc_length(x_expr: str, y_expr: str) -> float:
    """
    Estimate the arc length of a parametric segment (x(t), y(t)), t in [0, 1],
    by summing chord lengths over _ARC_REF_SAMPLES uniformly spaced t values.
    """
    prev_x = float(eval(x_expr, {"np": np, "t":0.0}))
    prev_y = float(eval(y_expr, {"np": np, "t":0.0}))
    length = 0.0

    for k in range(1, _ARC_REF_SAMPLES):
        t = k / (_ARC_REF_SAMPLES - 1)
        cx = float(eval(x_expr, {"np": np, "t":t}))
        cy = float(eval(y_expr, {"np": np, "t":t}))
        length += math.hypot(cx - prev_x, cy - prev_y)
        prev_x, prev_y = cx, cy

    return length


def sample_trajectories(mobile_nodes: list[MobileNode], step: float) -> list[Point2D]:
    """
    Convert all P1-P2 mobile-node trajectories into a set of plane points.

    For each parametric segment (x(t), y(t)) of each mobile node, the
    arc length of the segment is estimated numerically and then divided by
    ``step`` to determine how many sample points to generate.  The actual
    points are placed at uniformly spaced t values in [0, 1], always
    including both endpoints (t=0 and t=1).

    Parameters
    ----------
    mobile_nodes : list[MobileNode]
        Mobile nodes of the P1-P2 problems.  Each node must expose a
        ``path_segments`` attribute — a list of (x_expr, y_expr) string
        tuples representing parametric equations in the variable ``t``.
    step : float
        Maximum arc-length distance between consecutive sampled points.
        Smaller values produce denser sampling.

    Returns
    -------
    list[Point2D]
        All sampled points concatenated in order
        (node 0 seg 0, node 0 seg 1, …, node 1 seg 0, …).

    Raises
    ------
    ValueError
        If ``step`` is not positive.
    """
    if step <= 0.0:
        raise ValueError(f"step must be positive, got {step!r}")

    points: list[Point2D] = []

    for mn in mobile_nodes:
        for x_expr, y_expr in mn.path_segments:
            length = _arc_length(x_expr, y_expr)

            # At least 2 points (the two endpoints); ceil ensures the step
            # constraint is satisfied even for very short segments.
            n_points = max(2, math.ceil(length / step) + 1)

            t_values = [i / (n_points - 1) for i in range(n_points)]

            for t in t_values:
                x = float(eval(x_expr, {"np": np, "t":t}))  # noqa: S307
                y = float(eval(y_expr, {"np": np, "t":t}))
                points.append((x, y))

    return points


def build_coverage_matrix(
    sampled_points: list[Point2D],
    candidates: list[Point2D],
    radius: float,
) -> CoverageMatrix:
    """
    Build a bitset coverage matrix M of shape (N, m).

    M[i] is a Python int whose j-th bit is 1 iff candidate j is within
    ``radius`` of sampled point i:

        bit_j( M[i] ) = 1  <==>  dist(sampled_points[i], candidates[j]) <= radius

    Using Python's arbitrary-precision integers as bitsets keeps memory compact
    and makes row-wise OR/AND operations fast without external dependencies.

    Parameters
    ----------
    sampled_points : list[Point2D]
        Output of ``sample_trajectories`` — the N trajectory sample points.
    candidates : list[Point2D]
        The m candidate relay positions Q.
    radius : float
        Communication radius R.

    Returns
    -------
    CoverageMatrix
        List of N integers; the j-th bit of element i encodes coverage of
        sample point i by candidate j.
    """
    radius_sq = radius * radius
    matrix: CoverageMatrix = []

    for (sx, sy) in sampled_points:
        row = 0
        for j, (cx, cy) in enumerate(candidates):
            dx = sx - cx
            dy = sy - cy
            if dx * dx + dy * dy <= radius_sq:
                row |= 1 << j
        matrix.append(row)

    return matrix


def build_candidate_cover_bits(matrix: CoverageMatrix, n_candidates: int) -> list[int]:
    """
    Transpose a coverage matrix into per-candidate point bitsets.

    Element j of the result is a Python int whose i-th bit is 1 iff candidate j
    covers sampled point i:

        bit_i( cover_bits[j] ) = 1  <==>  bit_j( matrix[i] ) = 1

    Building this once per problem instance lets greedy coverage repair score
    each candidate with a single AND + popcount instead of re-scanning the
    matrix rows on every call.

    Parameters
    ----------
    matrix : CoverageMatrix
        Output of ``build_coverage_matrix`` — N bitset rows, one per
        sampled trajectory point.
    n_candidates : int
        Number of candidate positions m (bit width of each matrix row).

    Returns
    -------
    list[int]
        m integers; the i-th bit of element j encodes coverage of sample
        point i by candidate j.
    """
    cover_bits = [0] * n_candidates

    for i, row in enumerate(matrix):
        point_bit = 1 << i
        j = 0
        while row:
            if row & 1:
                cover_bits[j] |= point_bit
            row >>= 1
            j += 1

    return cover_bits


def greedy_coverage_repair_mask(
    matrix: CoverageMatrix,
    cover_bits: list[int],
    mask: list[int],
    min_pct: float,
    budget: int,
) -> list[int]:
    """
    Repair a P2 chromosome by greedily activating candidates until the
    trajectory coverage reaches ``min_pct`` or ``budget`` activations are spent.

    Classic greedy set-cover restricted to activation only: existing 1-bits are
    never cleared, so the parent's genetic material is preserved.  At each step
    the inactive candidate covering the most still-uncovered points is
    activated; ties break on the lowest index, making the repair fully
    deterministic (no RNG involved).  The loop stops early when no inactive
    candidate covers any uncovered point (irreparable scenario — the caller's
    penalty mechanism remains the safety net).

    Parameters
    ----------
    matrix : CoverageMatrix
        Output of ``build_coverage_matrix`` — N bitset rows.
    cover_bits : list[int]
        Output of ``build_candidate_cover_bits`` for the same matrix.
    mask : list[int]
        Binary chromosome of the P2 problem — m bits, one per candidate.
        Not mutated; a repaired copy is returned.
    min_pct : float
        Target coverage percentage in [0, 100].
    budget : int
        Maximum number of candidates to activate.

    Returns
    -------
    list[int]
        Repaired copy of ``mask`` (equal to the input when the mask is already
        feasible, the budget is non-positive, or no activation helps).
    """
    n_points = len(matrix)
    if n_points == 0 or budget <= 0:
        return mask[:]

    out = mask[:]

    covered = 0
    for j, bit in enumerate(out):
        if bit:
            covered |= cover_bits[j]

    all_points = (1 << n_points) - 1
    uncovered = all_points & ~covered
    required = math.ceil(min_pct / 100.0 * n_points)

    for _ in range(budget):
        if n_points - uncovered.bit_count() >= required:
            break

        best_j = -1
        best_gain = 0
        for j, bit in enumerate(out):
            if bit:
                continue
            gain = (cover_bits[j] & uncovered).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_j = j

        if best_j < 0:
            break  # no inactive candidate covers any uncovered point

        out[best_j] = 1
        uncovered &= ~cover_bits[best_j]

    return out


def check_coverage(matrix: CoverageMatrix, mask: list[int]) -> float:
    """
    Score how well a P2 chromosome covers the sampled trajectory points.

    Performs a boolean matrix-vector product (OR-semiring, where 1+1=1):

        covered[i] = OR_j ( matrix[i][j] AND mask[j] )

    and returns an integer in [0, 100] proportional to the fraction of
    covered points:

        score = round( covered_count / N * 100 )

    So 100 means full coverage, 0 means no point is covered, and
    intermediate values scale linearly with the coverage ratio.
    Returns 100 for an empty matrix (vacuously fully covered).

    Parameters
    ----------
    matrix : CoverageMatrix
        Output of ``build_coverage_matrix`` — N bitset rows, one per
        sampled trajectory point.
    mask : list[int]
        Binary chromosome of the P2 problem — m bits, one per candidate.

    Returns
    -------
    int
        Coverage score in [0, 100].
    """
    if not matrix:
        return 100

    mask_bits = 0
    for j, bit in enumerate(mask):
        if bit:
            mask_bits |= 1 << j

    covered = sum(1 for row in matrix if row & mask_bits)
    return covered / len(matrix) * 100.0
