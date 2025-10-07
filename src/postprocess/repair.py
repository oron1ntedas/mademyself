#Imports
import numpy as np
import open3d as o3d
import logging
from typing import Tuple, Dict, Optional, List

logger = logging.getLogger(__name__)

#File Imports

# pointcloud into mesh
class MeshReconstructor:
    
    def __init__(self):
        self.reconstruction_params = {}
    
    def pointcloud_to_mesh(self, points: np.ndarray, 
                          method: str = 'poisson', 
                          **kwargs) -> Tuple[o3d.geometry.TriangleMesh, Dict]:

        logger.info(f"Реконструкция mesh методом {method}: {len(points)} точек")
        
        # Создание объекта облака точек
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Вычисление нормалей
        pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )   

        # Ориентация нормалей к точке обзора (например, [0,0,0])
        pcd.orient_normals_towards_camera_location(camera_location=[0, 0, 0])

        # Альтернативная ориентация для локальной согласованности нормалей
        pcd.orient_normals_consistent_tangent_plane(k=30)
        
        # Выбор метода реконструкции
        if method == 'poisson':
            mesh, info = self._poisson_reconstruction(pcd, **kwargs)
        elif method == 'ball_pivoting':
            mesh, info = self._ball_pivoting_reconstruction(pcd, **kwargs)
        else:
            raise ValueError(f"Неизвестный метод: {method}")
        
        logger.info(f"Реконструкция завершена: {len(mesh.vertices)} вершин, {len(mesh.triangles)} граней")
        return mesh, info
    
    def _poisson_reconstruction(self, pcd: o3d.geometry.PointCloud, 
                               depth: int = 6, #width: float = 0.0, scale: float = 1.0
                               ) -> Tuple[o3d.geometry.TriangleMesh, Dict]:
        """Poisson Surface Reconstruction"""
        try:

            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=depth, #width=width, scale=scale
            )
            
            # Удаляем области с низкой плотностью (артефакты)
            densities = np.asarray(densities)
            density_threshold = np.quantile(densities, 0.01)  # убираем нижние 1%
            vertices_to_remove = densities < density_threshold
            mesh.remove_vertices_by_mask(vertices_to_remove)
            
            info = {
                'method': 'poisson',
                'depth': depth,
                'original_vertices': len(mesh.vertices),
                'removed_vertices': np.sum(vertices_to_remove),
                'density_threshold': density_threshold
            }
            
            return mesh, info
            
        except Exception as e:
            logger.error(f"Ошибка Poisson реконструкции: {e}")
            raise
    
    def _ball_pivoting_reconstruction(self, pcd: o3d.geometry.PointCloud,
                                    radii: Optional[List[float]] = None) -> Tuple[o3d.geometry.TriangleMesh, Dict]:
        """Ball Pivoting Algorithm"""
        try:
            if radii is None:
                # Вычисляем расстояния до ближайших соседей
                distances = pcd.compute_nearest_neighbor_distance()
                avg_dist = np.mean(distances)
                
                # Устанавливаем радиусы: базовый, средний и крупный масштаб
                radii = [avg_dist, avg_dist * 2, avg_dist * 4]
            
            radii_vector = o3d.utility.DoubleVector(radii)
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                pcd, radii_vector
            )
            
            info = {
                'method': 'ball_pivoting',
                'radii': radii,
                'vertices': len(mesh.vertices),
                'triangles': len(mesh.triangles)
            }
            
            return mesh, info
            
        except Exception as e:
            logger.error(f"Ошибка Ball Pivoting: {e}")
            raise


