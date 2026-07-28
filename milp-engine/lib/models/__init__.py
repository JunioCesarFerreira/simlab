from .base import BuiltModel, MilpModel, ParamSpec
from .p2_mobile import P2MobileModel

# Model key -> singleton instance (mirrors PROBLEM_REGISTRY in the mo-engine).
# P3 (target coverage) lands here in a later phase.
MILP_MODEL_REGISTRY: dict[str, MilpModel] = {
    P2MobileModel.key: P2MobileModel(),
}


def get_model(key: str) -> MilpModel:
    model = MILP_MODEL_REGISTRY.get(key)
    if model is None:
        known = ", ".join(sorted(MILP_MODEL_REGISTRY))
        raise ValueError(f"Unknown MILP model '{key}'. Known: {known}")
    return model
