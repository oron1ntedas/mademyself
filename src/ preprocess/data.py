# Imports
import os
import numpy as np
import trimesh
import torch
import random
from typing import Tuple, Optional, List
from torch.utils.data import Dataset

# File Imports
from normalize import norm_pointcloud

class MeshData(Dataset):

    def __init__(self, root_dir: str, num_points: int = 512,
                 normalize_pointcloud: bool = True, 
                 simulate_damage: bool = True):
        self.root_dir = root_dir
        self.num_points = num_points
        self.simulate_damage = simulate_damage
        self.normalize_pointcloud = normalize_pointcloud
        
        self.file_list = self._load_obj_files()
        print(f"Found {len(self.file_list)} .obj files in {root_dir}")

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        try:
            file_path = self.file_list[idx]
            mesh = trimesh.load(file_path, process=False)
            
            if not isinstance(mesh, trimesh.Trimesh) or not hasattr(mesh, 'vertices') or len(mesh.vertices) == 0:
                print(f"Invalid mesh or no vertices found in {file_path}")
                return self._create_fallback_data()

            points = self._mesh_to_pointcloud(mesh)

            if self.normalize_pointcloud:
                points = norm_pointcloud(points)

            if self.simulate_damage:
                damaged_points, damage_mask, hole_labels = self._simulate_holes(points, self.num_points)
                return points, damaged_points, damage_mask, hole_labels
            else:
                hole_loops = self._find_mesh_holes(mesh)
                real_hole_labels = self._create_hole_labels(points.numpy(), hole_loops)
                return points, points, None, real_hole_labels
            
        except Exception as e:
            print(f"Error loading file {self.file_list[idx]}: {e}")
            return self._create_fallback_data()

    def _load_obj_files(self) -> List[str]:
        file_list = []
        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(f"Directory not found: {self.root_dir}")
        for dirpath, _, filenames in os.walk(self.root_dir):
            for filename in filenames:
                if filename.endswith('.obj'):
                    file_list.append(os.path.join(dirpath, filename))
        
        if not file_list:
            raise FileNotFoundError(f"No .obj files found in {self.root_dir}")
        
        return file_list

    def _mesh_to_pointcloud(self, mesh: trimesh.Trimesh) -> torch.Tensor:
        if len(mesh.vertices) < self.num_points:
            repeat_count = (self.num_points // len(mesh.vertices)) + 1
            points = np.tile(mesh.vertices, (repeat_count, 1))
            points = points[:self.num_points]
        else:
            points, _ = trimesh.sample.sample_surface(mesh, self.num_points)
        
        return torch.tensor(points, dtype=torch.float32)

    def _find_mesh_holes(self, mesh: trimesh.Trimesh) -> list:
        if mesh.is_watertight:
            return []

        boundary_edges = trimesh.graph.edges_unique(mesh.faces.reshape(-1, 3))
        edge_counts = np.bincount(boundary_edges.flatten(), minlength=mesh.edges.max() + 1)
        boundary_edges = mesh.edges[edge_counts[mesh.edges].sum(axis=1) == 1]
        loops = trimesh.graph.split(boundary_edges, mesh.edges_unique.flatten())
        hole_loops = [loop[trimesh.graph.get_path_sequence(loop)] for loop in loops]
        
        return hole_loops

    def _create_hole_labels(self, points: np.ndarray, hole_loops: list) -> torch.Tensor:
        hole_labels = np.zeros(self.num_points, dtype=np.float32)
        if hole_loops:
            all_hole_vertices = np.vstack(hole_loops)
            distances = np.linalg.norm(points[:, np.newaxis, :] - all_hole_vertices, axis=2)
            min_distances = np.min(distances, axis=1)
            threshold = 0.01
            hole_labels[min_distances < threshold] = 1.0
        return torch.tensor(hole_labels, dtype=torch.float32)

    def _simulate_holes(self, points: torch.Tensor, 
                         num_points: int
                        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        damaged_points = points.clone()
        damage_mask = torch.ones(num_points, dtype=torch.bool)

        center_idx = random.randint(0, num_points - 1)
        center = damaged_points[center_idx]
        
        radius = random.uniform(0.1, 0.4)
        
        distances = torch.norm(damaged_points - center, dim=1)
        points_to_remove = distances < radius
        
        damaged_points[points_to_remove] = 0.0
        damage_mask[points_to_remove] = False
        
        hole_labels = torch.zeros(num_points, dtype=torch.float32)
        hole_labels[points_to_remove] = 1.0
        
        return damaged_points, damage_mask, hole_labels

    def _create_fallback_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Creates an empty tensor to avoid batching errors when a file fails to load."""
        fallback_points = torch.zeros(self.num_points, 3, dtype=torch.float32)
        fallback_mask = torch.ones(self.num_points, dtype=torch.bool)
        fallback_labels = torch.zeros(self.num_points, dtype=torch.float32)
        return fallback_points, fallback_points, fallback_mask, fallback_labels