from typing import Sequence

ObjectiveVec = Sequence[float]

def dominates(obj1: Sequence[float], obj2: Sequence[float]):
    """
    Minimization Dominance
    Return True if obj1 dominates obj2 (minimization):
    obj1 <= obj2 component-wise and < in at least one objective.
    """
    return all(x <= y for x, y in zip(obj1, obj2)) and any(x < y for x, y in zip(obj1, obj2))

def fast_nondominated_sort(objectives: Sequence[ObjectiveVec]) -> list[list[int]]:    
    """
    Deb's Fast Non-Dominated Sorting (NSGA-II).

    Implements the canonical non-dominated sorting procedure from
    Deb et al. (2002) to partition a population into Pareto fronts.
    Assumes minimization for all objectives and uses Deb's scheme:
    for each solution p, keep the set S[p] of solutions dominated by p
    and the domination count n[q] (how many solutions dominate q).
    The first front contains all solutions with n == 0; subsequent
    fronts are built by iteratively decreasing n over S[p].

    Parameters
    ----------
    objectives : Sequence[Sequence[float]]
        Population objective vectors (shape: N x M), where N is the
        number of solutions and M the number of objectives. All
        objectives are treated as "minimize".

    Returns
    -------
    fronts : list[list[int]]
        Ordered list of Pareto fronts (by indices into `objectives`);
        `fronts[0]` is the first (best) front, `fronts[1]` the second, etc.

    Notes
    -----
    - Complexity: O(M * N^2) in the worst case (M = #objectives, N = #solutions).
    - This is the classical NSGA-II sorting routine:
      K. Deb, A. Pratap, S. Agarwal, T. Meyarivan (2002),
      "A fast and elitist multiobjective genetic algorithm: NSGA-II".
    - Duplicated objective vectors are allowed and end up in the same front.
    """
    N = len(objectives)
    S = [[] for _ in range(N)]
    n = [0] * N
    fronts: list[list[int]] = [[]]

    # compare each pair once (j > i); skip i == j
    for i in range(N):
        for j in range(i + 1, N):
            if dominates(objectives[i], objectives[j]):
                S[i].append(j)
                n[j] += 1
            elif dominates(objectives[j], objectives[i]):
                S[j].append(i)
                n[i] += 1

    # first front
    for i in range(N):
        if n[i] == 0:
            fronts[0].append(i)

    # build next fronts
    i = 0
    while fronts[i]:
        next_front: list[int] = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    next_front.append(q)
        i += 1
        fronts.append(next_front)

    fronts.pop()  # remove trailing empty front
    return fronts