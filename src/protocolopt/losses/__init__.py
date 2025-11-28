from .standard import LogicGateEndpointLossBase, StandardLogicGateLoss
from .functional import variance_loss, work_loss, temporal_smoothness_penalty

__all__ = ["LogicGateEndpointLossBase", "StandardLogicGateLoss", "variance_loss", "work_loss", "temporal_smoothness_penalty"]