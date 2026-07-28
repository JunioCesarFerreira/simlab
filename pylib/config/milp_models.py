"""
MILP model catalog — single source of truth for model metadata.

Shared by the REST API (GET /milp/models) and by the milp-engine, whose model
classes load their parameter schemas from here. Pure data: no solver imports.

Each parameter entry:
  name        model parameter identifier (passed to the solver build)
  description shown in the sweep-builder GUI
  default     value used when the parameter is not swept nor fixed
  sweepable   False for parameters that must stay scalar (e.g. horizon)
"""

# Solver backends the milp-engine can run (see milp-engine/lib/solver).
MILP_ALLOWED_BACKENDS = ("gurobi", "highs")

MILP_MODEL_SPECS: dict[str, dict] = {
    "milp_p2_mobile": {
        "key": "milp_p2_mobile",
        "problem_key": "problem2",
        "title": "P2 — Mobile Coverage MILP",
        "description": (
            "Selects fixed relay installations (y_j over the candidate set) so "
            "that every mobile mote can route B units of flow to the sink at "
            "every time step, minimizing installation cost plus transmission "
            "energy. Multi-period minimum-cost flow with installation gating."
        ),
        "parameters": [
            {
                "name": "C0",
                "description": "Nominal channel capacity at distance 0",
                "default": 310.0,
                "sweepable": True,
            },
            {
                "name": "kdecay",
                "description": "Capacity attenuation factor per distance unit",
                "default": 0.25,
                "sweepable": True,
            },
            {
                "name": "B",
                "description": "Flow demand injected by each mobile mote per time step",
                "default": 25.0,
                "sweepable": True,
            },
            {
                "name": "w_install",
                "description": "Installation cost weight in the objective",
                "default": 1e6,
                "sweepable": True,
            },
            {
                "name": "duration",
                "description": "Planning horizon in seconds (defines T)",
                "default": 60.0,
                "sweepable": False,
            },
        ],
        "solver_defaults": {"time_limit_s": 300.0, "mip_gap": 0.01},
        "formulation": r"""
$$
\begin{aligned}
  \min_{y,z,x}\quad
    & w\sum_{j\in\mathcal J} y_j
    + \sum_{t\in\mathcal T}\sum_{(i,j)\in \mathcal E_t}e_{ij}(t)\, x_{ij}(t)
    \\
\text{s.t.}\quad
  & z_{ij}(t)\le y_i,\quad z_{ij}(t)\le y_j,
  && \forall (i,j)\in\mathcal E_t\cap(\mathcal{J}\times\mathcal{J}),\ \forall t,
  \\
  & 0\le x_{ij}(t)\le C_{ij}(t)\, z_{ij}(t),
  && \forall (i,j)\in\mathcal E_t,\ \forall t,
  \\
  & \textstyle\sum_{i} x_{mi}(t) - \sum_{i} x_{im}(t) = B,
  && \forall m\in\mathcal M,\ \forall t,
  \\
  & \textstyle\sum_{i} x_{ji}(t) - \sum_{i} x_{ij}(t) = 0,
  && \forall j\in\mathcal J,\ \forall t,
  \\
  & \textstyle\sum_{i} x_{is}(t) = B\,|\mathcal M|,
  && \forall t,
  \\
  & y_j\in\{0,1\},\quad z_{ij}(t)\in\{0,1\},\quad x_{ij}(t)\ge 0,
\end{aligned}
$$

with $C_{ij}(t)=\max\{0,\,C_0(1-k_{decay}\,d_{ij}(t))^2\}$ and
$e_{ij}(t)=d_{ij}^2(t)$ for $0<d_{ij}(t)\le R_{com}$.
""".strip(),
    },
}
