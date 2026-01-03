from typing import Any


def dominates(a: dict[str, float], b: dict[str, float], goals: dict[str, str]) -> bool:
    """
    Returns True if a dominates b.
    
    goals: {objective_name: "min" | "max"}
    """
    better_or_equal = True
    strictly_better = False

    for k, goal in goals.items():
        if goal == "min":
            if a[k] > b[k]:
                better_or_equal = False
            if a[k] < b[k]:
                strictly_better = True
        else:
            if a[k] < b[k]:
                better_or_equal = False
            if a[k] > b[k]:
                strictly_better = True

    return better_or_equal and strictly_better


def pareto_front(
    items: list[dict[str, Any]],
    goals: dict[str, str]
) -> list[dict[str, Any]]:
    front = []

    for i, a in enumerate(items):
        dominated = False
        for j, b in enumerate(items):
            if i == j:
                continue
            if dominates(b["objectives"], a["objectives"], goals):
                dominated = True
                break
        if not dominated:
            front.append(a)

    return front
