"""Topology utilities shared by the topology-aware P2 variant.

Nothing in this package depends on a strategy or on MongoDB: it is pure
geometry / graph reasoning over a P2 scenario, so it can be unit-tested and
reused by any future problem that needs a sink-rooted structural view.
"""
from .builder import (
    DEFAULT_TREE_COST,
    TreeCostWeights,
    active_components,
    build_sink_rooted_tree,
    edge_cost,
)
from .descriptors import (
    HISTORICAL_DESCRIPTOR_NAMES,
    STRUCTURAL_DESCRIPTOR_NAMES,
    TopologyDescriptorExtractor,
    TopologyDescriptors,
)
from .repair import (
    DEFAULT_REPAIR_WEIGHTS,
    RepairResult,
    RepairWeights,
    TopologyRepair,
)
from .rooted_tree import SINK_NODE, ParentArrayTree, RootedTreeBackend
from .routing import (
    RoutingKnowledge,
    RoutingObservation,
    merge_observations,
    observation_from_dodag,
)
from .scenario import MAX_TIME_SLICES, ScenarioTopology, TimeSlice
from .tree_operators import (
    INACTIVE,
    CoverageModel,
    TreeOperators,
    mask_from_tree,
)
from .two_level_tree import TwoLevelTree

__all__ = [
    "CoverageModel",
    "DEFAULT_REPAIR_WEIGHTS",
    "DEFAULT_TREE_COST",
    "HISTORICAL_DESCRIPTOR_NAMES",
    "INACTIVE",
    "MAX_TIME_SLICES",
    "ParentArrayTree",
    "RepairResult",
    "RepairWeights",
    "RootedTreeBackend",
    "RoutingKnowledge",
    "RoutingObservation",
    "SINK_NODE",
    "STRUCTURAL_DESCRIPTOR_NAMES",
    "ScenarioTopology",
    "TimeSlice",
    "TopologyDescriptorExtractor",
    "TopologyDescriptors",
    "TopologyRepair",
    "TreeCostWeights",
    "TreeOperators",
    "TwoLevelTree",
    "active_components",
    "build_sink_rooted_tree",
    "edge_cost",
    "mask_from_tree",
    "merge_observations",
    "observation_from_dodag",
]
