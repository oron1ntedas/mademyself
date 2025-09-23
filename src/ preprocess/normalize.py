#Imports
import numpy as np
import torch

#File Imports

#normalize pointcloud
def norm_pointcloud(points: np.ndarray) -> torch.Tensor:
    centroid = np.mean(points, axis=0)
    points_centered = points - centroid

    max_val = np.max(np.abs(points_centered))
    points_normalized = points_centered / max_val

    return torch.tensor(points_normalized, dtype=torch.float32)


# !!! check is norm_pointcloud clear_3dmodel is similar to norm_pointcloud crush_3dmodel!!!


