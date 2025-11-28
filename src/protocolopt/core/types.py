from typing import Annotated
import torch

# Define reusable types for clarity
ControlSignal = Annotated[torch.Tensor, "Control_Dim", "Time_Steps"]
StateSpace = Annotated[torch.Tensor, "Batch", "Spatial_Dim"]
ControlVector = Annotated[torch.Tensor, "Control_Dim"]
MicrostatePaths = Annotated[torch.Tensor, "Batch", "Spatial_Dim", "Time_Steps_Plus_1", "Phase_Dim(pos, vel)"]
PotentialTensor = Annotated[torch.Tensor, "Batch", "Time_Steps"]
MalliavinWeight = Annotated[torch.Tensor, "Batch", "Control_Dim", "Time_Steps"]
