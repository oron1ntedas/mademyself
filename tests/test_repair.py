#Imports

#File Imports

# CHECK pointcloud into mesh
"""
Тесты для модуля repair.py - восстановление мешей из облаков точек.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sys
sys.path.insert(0, '/home/ubuntu/upload')

# Мокируем open3d, так как он может быть не установлен в тестовой среде
@pytest.fixture
def mock_o3d():
    with patch('repair.o3d') as mock:
        # Настраиваем мок для PointCloud
        mock_pcd = Mock()
        mock_pcd.points = Mock()
        mock_pcd.estimate_normals = Mock()
        mock_pcd.orient_normals_towards_camera_location = Mock()
        mock_pcd.orient_normals_consistent_tangent_plane = Mock()
        
        mock.geometry.PointCloud.return_value = mock_pcd
        mock.utility.Vector3dVector = Mock(return_value="mocked_vector3d")
        
        # Настраиваем мок для TriangleMesh
        mock_mesh = Mock()
        mock_mesh.vertices = np.random.rand(100, 3)
        mock_mesh.triangles = np.random.randint(0, 100, (50, 3))
        mock_mesh.remove_vertices_by_mask = Mock()
        
        # Настраиваем методы реконструкции
        mock.geometry.TriangleMesh.create_from_point_cloud_poisson.return_value = (
            mock_mesh, np.random.rand(100)
        )
        mock.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting.return_value = mock_mesh
        
        # Настраиваем KDTreeSearchParam
        mock.geometry.KDTreeSearchParamHybrid = Mock()
        mock.utility.DoubleVector = Mock()
        
        yield mock

class TestMeshReconstructor:
    """Тесты для класса MeshReconstructor."""
    
    def test_init(self, mock_o3d):
        """Тест инициализации MeshReconstructor."""
        from postprocess.repair import MeshReconstructor
        
        reconstructor = MeshReconstructor()
        assert hasattr(reconstructor, 'reconstruction_params')
        assert isinstance(reconstructor.reconstruction_params, dict)
    
    def test_pointcloud_to_mesh_poisson_default(self, mock_o3d):
        """Тест реконструкции меша методом Пуассона с параметрами по умолчанию."""
        from postprocess.repair import MeshReconstructor
        
        reconstructor = MeshReconstructor()
        points = np.random.rand(100, 3).astype(np.float32)
        
        mesh, info = reconstructor.pointcloud_to_mesh(points, method='poisson')
        
        # Проверяем, что создался PointCloud
        mock_o3d.geometry.PointCloud.assert_called_once()
        
        # Проверяем, что вызвались методы обработки нормалей
        mock_pcd = mock_o3d.geometry.PointCloud.return_value
        mock_pcd.estimate_normals.assert_called_once()
        mock_pcd.orient_normals_towards_camera_location.assert_called_once()
        mock_pcd.orient_normals_consistent_tangent_plane.assert_called_once()
        
        # Проверяем, что вызвался метод Пуассона
        mock_o3d.geometry.TriangleMesh.create_from_point_cloud_poisson.assert_called_once()
        
        # Проверяем результат
        assert mesh is not None
        assert isinstance(info, dict)
        assert info['method'] == 'poisson'
        assert 'depth' in info
        assert 'original_vertices' in info
    
    def test_pointcloud_to_mesh_poisson_custom_params(self, mock_o3d):
        """Тест реконструкции Пуассона с пользовательскими параметрами."""
        from postprocess.repair import MeshReconstructor
        
        reconstructor = MeshReconstructor()
        points = np.random.rand(50, 3).astype(np.float32)
        
        mesh, info = reconstructor.pointcloud_to_mesh(
            points, 
            method='poisson', 
            depth=8
        )
        
        # Проверяем, что параметры переданы правильно
        call_args = mock_o3d.geometry.TriangleMesh.create_from_point_cloud_poisson.call_args
        assert call_args[1]['depth'] == 8
        
        assert info['depth'] == 8
    
    def test_pointcloud_to_mesh_ball_pivoting_default(self, mock_o3d):
        """Тест реконструкции методом Ball Pivoting с параметрами по умолчанию."""
        from postprocess.repair import MeshReconstructor
        
        # Настраиваем мок для compute_nearest_neighbor_distance
        mock_pcd = mock_o3d.geometry.PointCloud.return_value
        mock_pcd.compute_nearest_neighbor_distance.return_value = np.array([0.1, 0.15, 0.12, 0.08, 0.2])
        
        reconstructor = MeshReconstructor()
        points = np.random.rand(100, 3).astype(np.float32)
        
        mesh, info = reconstructor.pointcloud_to_mesh(points, method='ball_pivoting')
        
        # Проверяем, что вычислились расстояния до соседей
        mock_pcd.compute_nearest_neighbor_distance.assert_called_once()
        
        # Проверяем, что вызвался Ball Pivoting
        mock_o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting.assert_called_once()
        
        # Проверяем результат
        assert mesh is not None
        assert isinstance(info, dict)
        assert info['method'] == 'ball_pivoting'
        assert 'radii' in info
        assert 'vertices' in info
        assert 'triangles' in info
    
    def test_pointcloud_to_mesh_ball_pivoting_custom_radii(self, mock_o3d):
        """Тест Ball Pivoting с пользовательскими радиусами."""
        from postprocess.repair import MeshReconstructor
        
        reconstructor = MeshReconstructor()
        points = np.random.rand(100, 3).astype(np.float32)
        custom_radii = [0.05, 0.1, 0.2]
        
        mesh, info = reconstructor.pointcloud_to_mesh(
            points, 
            method='ball_pivoting', 
            radii=custom_radii
        )
        
        # Проверяем, что переданы пользовательские радиусы
        mock_o3d.utility.DoubleVector.assert_called_once_with(custom_radii)
        
        assert info['radii'] == custom_radii
    
    def test_pointcloud_to_mesh_unknown_method(self, mock_o3d):
        """Тест с неизвестным методом реконструкции."""
        from postprocess.repair import MeshReconstructor
        
        reconstructor = MeshReconstructor()
        points = np.random.rand(50, 3).astype(np.float32)
        
        with pytest.raises(ValueError, match="Неизвестный метод"):
            reconstructor.pointcloud_to_mesh(points, method='unknown_method')
    
    def test_pointcloud_to_mesh_empty_points(self, mock_o3d):
        """Тест с пустым облаком точек."""
        from postprocess.repair import MeshReconstructor
        
        reconstructor = MeshReconstructor()
        points = np.empty((0, 3), dtype=np.float32)
        
        mesh, info = reconstructor.pointcloud_to_mesh(points, method='poisson')
        
        # Должно работать даже с пустыми точками
        assert mesh is not None
        assert isinstance(info, dict)
    
    def test_pointcloud_to_mesh_single_point(self, mock_o3d):
        """Тест с одной точкой."""
        from postprocess.repair import MeshReconstructor
        
        reconstructor = MeshReconstructor()
        points = np.array([[0, 0, 0]], dtype=np.float32)
        
        mesh, info = reconstructor.pointcloud_to_mesh(points, method='poisson')
        
        assert mesh is not None
        assert isinstance(info, dict)
    
    def test_poisson_reconstruction_density_filtering(self, mock_o3d):
        """Тест фильтрации по плотности в реконструкции Пуассона."""
        from postprocess.repair import MeshReconstructor
        
        # Настраиваем мок для возврата плотностей
        mock_mesh = Mock()
        mock_mesh.vertices = np.random.rand(100, 3)
        mock_mesh.triangles = np.random.randint(0, 100, (50, 3))
        mock_mesh.remove_vertices_by_mask = Mock()
        
        densities = np.random.rand(100)
        mock_o3d.geometry.TriangleMesh.create_from_point_cloud_poisson.return_value = (
            mock_mesh, densities
        )
        
        reconstructor = MeshReconstructor()
        points = np.random.rand(100, 3).astype(np.float32)
        
        mesh, info = reconstructor.pointcloud_to_mesh(points, method='poisson')
        
        # Проверяем, что вызвался метод удаления вершин
        mock_mesh.remove_vertices_by_mask.assert_called_once()
        
        # Проверяем, что в info есть информация о фильтрации
        assert 'density_threshold' in info
        assert 'removed_vertices' in info
    
    def test_poisson_reconstruction_exception(self, mock_o3d):
        """Тест обработки исключения в реконструкции Пуассона."""
        from postprocess.repair import MeshReconstructor
        
        # Настраиваем мок для выброса исключения
        mock_o3d.geometry.TriangleMesh.create_from_point_cloud_poisson.side_effect = Exception("Poisson error")
        
        reconstructor = MeshReconstructor()
        points = np.random.rand(50, 3).astype(np.float32)
        
        with pytest.raises(Exception, match="Poisson error"):
            reconstructor.pointcloud_to_mesh(points, method='poisson')
    
    def test_ball_pivoting_reconstruction_exception(self, mock_o3d):
        """Тест обработки исключения в Ball Pivoting."""
        from postprocess.repair import MeshReconstructor
        
        # Настраиваем мок для выброса исключения
        mock_o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting.side_effect = Exception("Ball pivoting error")
        
        reconstructor = MeshReconstructor()
        points = np.random.rand(50, 3).astype(np.float32)
        
        with pytest.raises(Exception, match="Ball pivoting error"):
            reconstructor.pointcloud_to_mesh(points, method='ball_pivoting')
    
    @pytest.mark.parametrize("method,expected_calls", [
        ('poisson', 'create_from_point_cloud_poisson'),
        ('ball_pivoting', 'create_from_point_cloud_ball_pivoting')
    ])
    def test_method_selection(self, mock_o3d, method, expected_calls):
        """Параметризованный тест выбора метода реконструкции."""
        from postprocess.repair import MeshReconstructor
        
        if method == 'ball_pivoting':
            # Настраиваем мок для Ball Pivoting
            mock_pcd = mock_o3d.geometry.PointCloud.return_value
            mock_pcd.compute_nearest_neighbor_distance.return_value = np.array([0.1, 0.1, 0.1])
        
        reconstructor = MeshReconstructor()
        points = np.random.rand(50, 3).astype(np.float32)
        
        mesh, info = reconstructor.pointcloud_to_mesh(points, method=method)
        
        # Проверяем, что вызвался правильный метод
        expected_method = getattr(mock_o3d.geometry.TriangleMesh, expected_calls)
        expected_method.assert_called_once()
        
        assert info['method'] == method
    
    def test_normal_estimation_parameters(self, mock_o3d):
        """Тест параметров оценки нормалей."""
        from postprocess.repair import MeshReconstructor
        
        reconstructor = MeshReconstructor()
        points = np.random.rand(100, 3).astype(np.float32)
        
        reconstructor.pointcloud_to_mesh(points, method='poisson')
        
        # Проверяем параметры оценки нормалей
        mock_pcd = mock_o3d.geometry.PointCloud.return_value
        mock_pcd.estimate_normals.assert_called_once()
        
        # Проверяем, что вызвались методы ориентации нормалей
        mock_pcd.orient_normals_towards_camera_location.assert_called_once_with(camera_location=[0, 0, 0])
        mock_pcd.orient_normals_consistent_tangent_plane.assert_called_once_with(k=30)
    
    def test_point_cloud_preprocessing(self, mock_o3d):
        """Тест предобработки облака точек."""
        from postprocess.repair import MeshReconstructor
        
        reconstructor = MeshReconstructor()
        points = np.random.rand(100, 3).astype(np.float32)
        
        reconstructor.pointcloud_to_mesh(points, method='poisson')
        
        # Проверяем создание PointCloud
        mock_o3d.geometry.PointCloud.assert_called_once()
        
        # Проверяем установку точек
        mock_pcd = mock_o3d.geometry.PointCloud.return_value
        mock_o3d.utility.Vector3dVector.assert_called_once_with(points)
    
    @pytest.mark.parametrize("num_points", [10, 100, 1000, 5000])
    def test_various_point_cloud_sizes(self, mock_o3d, num_points):
        """Параметризованный тест для различных размеров облаков точек."""
        from postprocess.repair import MeshReconstructor
        
        reconstructor = MeshReconstructor()
        points = np.random.rand(num_points, 3).astype(np.float32)
        
        mesh, info = reconstructor.pointcloud_to_mesh(points, method='poisson')
        
        assert mesh is not None
        assert isinstance(info, dict)
        assert info['method'] == 'poisson'
    
    def test_ball_pivoting_radii_calculation(self, mock_o3d):
        """Тест вычисления радиусов для Ball Pivoting."""
        from postprocess.repair import MeshReconstructor
        
        # Настраиваем мок для возврата конкретных расстояний
        mock_pcd = mock_o3d.geometry.PointCloud.return_value
        test_distances = np.array([0.1, 0.2, 0.15, 0.12, 0.18])
        mock_pcd.compute_nearest_neighbor_distance.return_value = test_distances
        
        reconstructor = MeshReconstructor()
        points = np.random.rand(50, 3).astype(np.float32)
        
        mesh, info = reconstructor.pointcloud_to_mesh(points, method='ball_pivoting')
        
        # Проверяем, что радиусы вычислились правильно
        expected_avg_dist = np.mean(test_distances)
        expected_radii = [expected_avg_dist, expected_avg_dist * 2, expected_avg_dist * 4]
        
        assert info['radii'] == expected_radii
    
    def test_mesh_info_completeness(self, mock_o3d):
        """Тест полноты информации в возвращаемом info."""
        from postprocess.repair import MeshReconstructor
        
        reconstructor = MeshReconstructor()
        points = np.random.rand(100, 3).astype(np.float32)
        
        # Тест для Poisson
        mesh_poisson, info_poisson = reconstructor.pointcloud_to_mesh(points, method='poisson')
        
        required_poisson_keys = ['method', 'depth', 'original_vertices', 'removed_vertices', 'density_threshold']
        for key in required_poisson_keys:
            assert key in info_poisson
        
        # Тест для Ball Pivoting
        mock_pcd = mock_o3d.geometry.PointCloud.return_value
        mock_pcd.compute_nearest_neighbor_distance.return_value = np.array([0.1, 0.1, 0.1])
        
        mesh_bp, info_bp = reconstructor.pointcloud_to_mesh(points, method='ball_pivoting')
        
        required_bp_keys = ['method', 'radii', 'vertices', 'triangles']
        for key in required_bp_keys:
            assert key in info_bp

class TestMeshReconstructorIntegration:
    """Интеграционные тесты для MeshReconstructor."""
    
    @pytest.mark.integration
    def test_full_reconstruction_pipeline(self, mock_o3d):
        """Тест полного pipeline реконструкции."""
        from postprocess.repair import MeshReconstructor
        
        # Создаем реалистичное облако точек (сфера)
        theta = np.linspace(0, 2*np.pi, 50)
        phi = np.linspace(0, np.pi, 25)
        theta, phi = np.meshgrid(theta, phi)
        
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        
        points = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1).astype(np.float32)
        
        reconstructor = MeshReconstructor()
        
        # Тестируем оба метода
        mesh_poisson, info_poisson = reconstructor.pointcloud_to_mesh(points, method='poisson')
        
        mock_pcd = mock_o3d.geometry.PointCloud.return_value
        mock_pcd.compute_nearest_neighbor_distance.return_value = np.random.rand(len(points)) * 0.1
        
        mesh_bp, info_bp = reconstructor.pointcloud_to_mesh(points, method='ball_pivoting')
        
        # Проверяем результаты
        assert mesh_poisson is not None
        assert mesh_bp is not None
        assert info_poisson['method'] == 'poisson'
        assert info_bp['method'] == 'ball_pivoting'
    
    @pytest.mark.slow
    def test_large_pointcloud_reconstruction(self, mock_o3d):
        """Тест реконструкции большого облака точек."""
        from postprocess.repair import MeshReconstructor
        
        # Создаем большое облако точек
        num_points = 10000
        points = np.random.rand(num_points, 3).astype(np.float32)
        
        reconstructor = MeshReconstructor()
        
        # Настраиваем мок для больших данных
        mock_mesh = Mock()
        mock_mesh.vertices = np.random.rand(num_points, 3)
        mock_mesh.triangles = np.random.randint(0, num_points, (num_points//2, 3))
        mock_mesh.remove_vertices_by_mask = Mock()
        
        mock_o3d.geometry.TriangleMesh.create_from_point_cloud_poisson.return_value = (
            mock_mesh, np.random.rand(num_points)
        )
        
        mesh, info = reconstructor.pointcloud_to_mesh(points, method='poisson')
        
        assert mesh is not None
        assert isinstance(info, dict)
        assert info['original_vertices'] == len(mock_mesh.vertices)
    
    def test_reconstruction_with_noise(self, mock_o3d):
        """Тест реконструкции с зашумленными данными."""
        from postprocess.repair import MeshReconstructor
        
        # Создаем зашумленное облако точек
        clean_points = np.random.rand(100, 3).astype(np.float32)
        noise = np.random.normal(0, 0.01, clean_points.shape).astype(np.float32)
        noisy_points = clean_points + noise
        
        reconstructor = MeshReconstructor()
        
        mesh, info = reconstructor.pointcloud_to_mesh(noisy_points, method='poisson')
        
        assert mesh is not None
        assert isinstance(info, dict)
        assert info['method'] == 'poisson'