#Imports

#File Imports

# CHECK load mesh
# CHECK find holes
# CHECK mesh into pointcloud

import pytest
import numpy as np
import torch
import trimesh
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys
sys.path.insert(0, '/home/ubuntu/upload')

from preprocess.data import MeshData

class TestMeshData:
    """Тесты для класса MeshData."""
    
    def test_init_default_params(self, temp_dir):
        """Тест инициализации с параметрами по умолчанию."""
        # Создаем тестовый .obj файл
        obj_file = temp_dir / "test.obj"
        obj_file.write_text("""
v 0 0 0
v 1 0 0
v 0 1 0
f 1 2 3
""")
        
        dataset = MeshData(root_dir=str(temp_dir))
        
        assert dataset.root_dir == str(temp_dir)
        assert dataset.num_points == 512
        assert dataset.normalize_pointcloud == True
        assert dataset.simulate_damage == True
        assert len(dataset.file_list) == 1
    
    def test_init_custom_params(self, temp_dir):
        """Тест инициализации с пользовательскими параметрами."""
        obj_file = temp_dir / "test.obj"
        obj_file.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3")
        
        dataset = MeshData(
            root_dir=str(temp_dir),
            num_points=1024,
            normalize_pointcloud=False,
            simulate_damage=False
        )
        
        assert dataset.num_points == 1024
        assert dataset.normalize_pointcloud == False
        assert dataset.simulate_damage == False
    
    def test_init_no_obj_files(self, temp_dir):
        """Тест инициализации когда нет .obj файлов."""
        with pytest.raises(FileNotFoundError, match="No .obj files found"):
            MeshData(root_dir=str(temp_dir))
    
    def test_init_nonexistent_directory(self):
        """Тест инициализации с несуществующей директорией."""
        with pytest.raises(FileNotFoundError, match="Directory not found"):
            MeshData(root_dir="/nonexistent/directory")
    
    def test_len(self, temp_dir):
        """Тест метода __len__."""
        # Создаем несколько тестовых файлов
        for i in range(3):
            obj_file = temp_dir / f"test{i}.obj"
            obj_file.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3")
        
        dataset = MeshData(root_dir=str(temp_dir))
        assert len(dataset) == 3
    
    def test_load_obj_files_recursive(self, temp_dir):
        """Тест загрузки .obj файлов из подпапок."""
        # Создаем структуру папок
        subdir = temp_dir / "subdir"
        subdir.mkdir()
        
        # Файлы в корневой папке
        (temp_dir / "root.obj").write_text("v 0 0 0\nf 1 1 1")
        # Файлы в подпапке
        (subdir / "sub.obj").write_text("v 0 0 0\nf 1 1 1")
        # Не .obj файл (должен игнорироваться)
        (temp_dir / "ignore.txt").write_text("ignore me")
        
        dataset = MeshData(root_dir=str(temp_dir))
        assert len(dataset.file_list) == 2
    
    @patch('trimesh.load')
    @patch('trimesh.sample.sample_surface')
    def test_getitem_with_damage_simulation(self, mock_sample, mock_load, temp_dir):
        """Тест __getitem__ с симуляцией повреждений."""
        # Настраиваем моки
        mock_mesh = Mock()
        mock_mesh.vertices = np.random.rand(100, 3)
        mock_load.return_value = mock_mesh
        
        points = np.random.rand(512, 3).astype(np.float32)
        mock_sample.return_value = (points, None)
        
        # Создаем тестовый файл
        obj_file = temp_dir / "test.obj"
        obj_file.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3")
        
        dataset = MeshData(root_dir=str(temp_dir), simulate_damage=True)
        result = dataset[0]
        
        assert result is not None
        assert len(result) == 4  # original, damaged, mask, labels
        
        original, damaged, mask, labels = result
        assert isinstance(original, torch.Tensor)
        assert isinstance(damaged, torch.Tensor)
        assert isinstance(mask, torch.Tensor)
        assert isinstance(labels, torch.Tensor)
        
        assert original.shape == (512, 3)
        assert damaged.shape == (512, 3)
        assert mask.shape == (512,)
        assert labels.shape == (512,)
    
    @patch('trimesh.load')
    def test_getitem_without_damage_simulation(self, mock_load, temp_dir):
        """Тест __getitem__ без симуляции повреждений."""
        # Настраиваем мок
        mock_mesh = Mock()
        mock_mesh.vertices = np.random.rand(100, 3)
        mock_mesh.is_watertight = True
        mock_load.return_value = mock_mesh
        
        obj_file = temp_dir / "test.obj"
        obj_file.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3")
        
        dataset = MeshData(root_dir=str(temp_dir), simulate_damage=False)
        
        with patch.object(dataset, '_mesh_to_pointcloud') as mock_mesh_to_pc:
            mock_mesh_to_pc.return_value = torch.randn(512, 3)
            
            with patch.object(dataset, '_find_mesh_holes') as mock_find_holes:
                mock_find_holes.return_value = []
                
                with patch.object(dataset, '_create_hole_labels') as mock_create_labels:
                    mock_create_labels.return_value = torch.zeros(512)
                    
                    result = dataset[0]
        
        assert result is not None
        original, damaged, mask, labels = result
        
        # Без симуляции повреждений original и damaged должны быть одинаковыми
        assert torch.allclose(original, damaged)
        assert mask is None
    
    @patch('trimesh.load')
    def test_getitem_invalid_mesh(self, mock_load, temp_dir):
        """Тест __getitem__ с невалидным мешем."""
        # Мок возвращает невалидный меш
        mock_load.return_value = None
        
        obj_file = temp_dir / "test.obj"
        obj_file.write_text("invalid mesh data")
        
        dataset = MeshData(root_dir=str(temp_dir))
        result = dataset[0]
        
        # Должен вернуть fallback данные
        assert result is not None
        original, damaged, mask, labels = result
        assert original.shape == (512, 3)
        assert torch.all(original == 0)  # Fallback данные - нули
    
    @patch('trimesh.load')
    def test_getitem_load_exception(self, mock_load, temp_dir):
        """Тест __getitem__ с исключением при загрузке."""
        mock_load.side_effect = Exception("Load error")
        
        obj_file = temp_dir / "test.obj"
        obj_file.write_text("v 0 0 0\nf 1 1 1")
        
        dataset = MeshData(root_dir=str(temp_dir))
        result = dataset[0]
        
        # Должен вернуть fallback данные
        assert result is not None
        original, damaged, mask, labels = result
        assert original.shape == (512, 3)
    
    def test_mesh_to_pointcloud_small_mesh(self):
        """Тест преобразования меша с малым количеством вершин."""
        dataset = MeshData.__new__(MeshData)  # Создаем без инициализации
        dataset.num_points = 100
        
        # Меш с малым количеством вершин
        mock_mesh = Mock()
        mock_mesh.vertices = np.random.rand(10, 3)  # Меньше чем num_points
        
        result = dataset._mesh_to_pointcloud(mock_mesh)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape == (100, 3)
        assert result.dtype == torch.float32
    
    @patch('trimesh.sample.sample_surface')
    def test_mesh_to_pointcloud_large_mesh(self, mock_sample):
        """Тест преобразования меша с большим количеством вершин."""
        dataset = MeshData.__new__(MeshData)
        dataset.num_points = 100
        
        mock_mesh = Mock()
        mock_mesh.vertices = np.random.rand(1000, 3)  # Больше чем num_points
        
        # Настраиваем мок для sample_surface
        sampled_points = np.random.rand(100, 3)
        mock_sample.return_value = (sampled_points, None)
        
        result = dataset._mesh_to_pointcloud(mock_mesh)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape == (100, 3)
        mock_sample.assert_called_once_with(mock_mesh, 100)
    
    def test_find_mesh_holes_watertight(self):
        """Тест поиска дыр в водонепроницаемом меше."""
        dataset = MeshData.__new__(MeshData)
        
        mock_mesh = Mock()
        mock_mesh.is_watertight = True
        
        holes = dataset._find_mesh_holes(mock_mesh)
        
        assert holes == []
    
    @patch('trimesh.graph.edges_unique')
    @patch('trimesh.graph.split')
    @patch('trimesh.graph.get_path_sequence')
    def test_find_mesh_holes_with_holes(self, mock_path_seq, mock_split, mock_edges):
        """Тест поиска дыр в меше с дырами."""
        dataset = MeshData.__new__(MeshData)
        
        mock_mesh = Mock()
        mock_mesh.is_watertight = False
        mock_mesh.faces = np.array([[0, 1, 2], [1, 2, 3]])
        mock_mesh.edges = np.array([[0, 1], [1, 2], [2, 0]])
        mock_mesh.edges_unique = np.array([0, 1, 2])
        
        # Настраиваем моки
        mock_edges.return_value = np.array([[0, 1], [1, 2]])
        mock_split.return_value = [np.array([0, 1, 2])]
        mock_path_seq.return_value = np.array([0, 1, 2])
        
        holes = dataset._find_mesh_holes(mock_mesh)
        
        assert isinstance(holes, list)
        assert len(holes) == 1
    
    def test_create_hole_labels_no_holes(self):
        """Тест создания меток дыр когда дыр нет."""
        dataset = MeshData.__new__(MeshData)
        dataset.num_points = 100
        
        points = np.random.rand(100, 3)
        hole_loops = []
        
        labels = dataset._create_hole_labels(points, hole_loops)
        
        assert isinstance(labels, torch.Tensor)
        assert labels.shape == (100,)
        assert torch.all(labels == 0)
    
    def test_create_hole_labels_with_holes(self):
        """Тест создания меток дыр когда есть дыры."""
        dataset = MeshData.__new__(MeshData)
        dataset.num_points = 10
        
        points = np.array([
            [0, 0, 0], [0.005, 0, 0], [0.1, 0, 0],  # Первые две точки близко к дыре
            [1, 1, 1], [2, 2, 2], [3, 3, 3],
            [4, 4, 4], [5, 5, 5], [6, 6, 6], [7, 7, 7]
        ])
        
        hole_loops = [np.array([[0, 0, 0], [0.01, 0, 0]])]  # Дыра рядом с первыми точками
        
        labels = dataset._create_hole_labels(points, hole_loops)
        
        assert isinstance(labels, torch.Tensor)
        assert labels.shape == (10,)
        # Первые две точки должны быть помечены как дыры
        assert labels[0] == 1.0
        assert labels[1] == 1.0
        assert labels[2] == 0.0  # Третья точка далеко
    
    def test_simulate_holes(self):
        """Тест симуляции дыр в облаке точек."""
        dataset = MeshData.__new__(MeshData)
        
        # Создаем облако точек
        points = torch.randn(100, 3)
        num_points = 100
        
        damaged, mask, labels = dataset._simulate_holes(points, num_points)
        
        assert isinstance(damaged, torch.Tensor)
        assert isinstance(mask, torch.Tensor)
        assert isinstance(labels, torch.Tensor)
        
        assert damaged.shape == (100, 3)
        assert mask.shape == (100,)
        assert labels.shape == (100,)
        
        # Проверяем, что некоторые точки были удалены
        assert torch.sum(~mask) > 0  # Есть удаленные точки
        assert torch.sum(labels) > 0  # Есть метки дыр
        
        # Проверяем, что удаленные точки зануляются
        removed_points = damaged[~mask]
        if len(removed_points) > 0:
            assert torch.all(removed_points == 0)
    
    def test_create_fallback_data(self):
        """Тест создания fallback данных."""
        dataset = MeshData.__new__(MeshData)
        dataset.num_points = 50
        
        fallback = dataset._create_fallback_data()
        
        assert len(fallback) == 4
        points, damaged, mask, labels = fallback
        
        assert points.shape == (50, 3)
        assert damaged.shape == (50, 3)
        assert mask.shape == (50,)
        assert labels.shape == (50,)
        
        assert torch.all(points == 0)
        assert torch.all(damaged == 0)
        assert torch.all(mask == True)
        assert torch.all(labels == 0)
    
    @pytest.mark.parametrize("num_points", [256, 512, 1024, 2048])
    def test_various_num_points(self, temp_dir, num_points):
        """Параметризованный тест для различных количеств точек."""
        obj_file = temp_dir / "test.obj"
        obj_file.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3")
        
        with patch('trimesh.load') as mock_load:
            mock_mesh = Mock()
            mock_mesh.vertices = np.random.rand(1000, 3)
            mock_load.return_value = mock_mesh
            
            with patch('trimesh.sample.sample_surface') as mock_sample:
                points = np.random.rand(num_points, 3).astype(np.float32)
                mock_sample.return_value = (points, None)
                
                dataset = MeshData(root_dir=str(temp_dir), num_points=num_points)
                result = dataset[0]
        
        original, damaged, mask, labels = result
        assert original.shape == (num_points, 3)
        assert damaged.shape == (num_points, 3)
        assert mask.shape == (num_points,)
        assert labels.shape == (num_points,)
    
    def test_normalize_pointcloud_integration(self, temp_dir):
        """Тест интеграции с нормализацией облака точек."""
        obj_file = temp_dir / "test.obj"
        obj_file.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3")
        
        with patch('trimesh.load') as mock_load:
            mock_mesh = Mock()
            mock_mesh.vertices = np.random.rand(100, 3)
            mock_load.return_value = mock_mesh
            
            with patch('trimesh.sample.sample_surface') as mock_sample:
                # Создаем точки с большими координатами
                points = np.random.rand(512, 3).astype(np.float32) * 100
                mock_sample.return_value = (points, None)
                
                # С нормализацией
                dataset_norm = MeshData(root_dir=str(temp_dir), normalize_pointcloud=True)
                result_norm = dataset_norm[0]
                
                # Без нормализации
                dataset_no_norm = MeshData(root_dir=str(temp_dir), normalize_pointcloud=False)
                result_no_norm = dataset_no_norm[0]
        
        # Проверяем, что нормализация применилась
        original_norm = result_norm[0]
        original_no_norm = result_no_norm[0]
        
        # Нормализованные точки должны быть в диапазоне [-1, 1]
        assert torch.max(torch.abs(original_norm)) <= 1.0 + 1e-5
        # Ненормализованные точки должны быть больше
        assert torch.max(torch.abs(original_no_norm)) > 1.0

class TestMeshDataIntegration:
    """Интеграционные тесты для MeshData."""
    
    @pytest.mark.integration
    def test_full_pipeline_with_real_mesh(self, temp_dir):
        """Тест полного pipeline с реальным мешем."""
        # Создаем простой меш куба
        cube_obj = """
# Cube vertices
v -0.5 -0.5 -0.5
v 0.5 -0.5 -0.5
v -0.5 0.5 -0.5
v 0.5 0.5 -0.5
v -0.5 -0.5 0.5
v 0.5 -0.5 0.5
v -0.5 0.5 0.5
v 0.5 0.5 0.5

# Cube faces
f 1 2 4 3
f 5 6 8 7
f 1 5 7 3
f 2 6 8 4
f 1 2 6 5
f 3 4 8 7
"""
        
        obj_file = temp_dir / "cube.obj"
        obj_file.write_text(cube_obj)
        
        dataset = MeshData(
            root_dir=str(temp_dir),
            num_points=1024,
            normalize_pointcloud=True,
            simulate_damage=True
        )
        
        # Тестируем загрузку данных
        assert len(dataset) == 1
        
        result = dataset[0]
        assert result is not None
        
        original, damaged, mask, labels = result
        
        # Проверяем формы
        assert original.shape == (1024, 3)
        assert damaged.shape == (1024, 3)
        assert mask.shape == (1024,)
        assert labels.shape == (1024,)
        
        # Проверяем типы
        assert original.dtype == torch.float32
        assert damaged.dtype == torch.float32
        assert mask.dtype == torch.bool
        assert labels.dtype == torch.float32
        
        # Проверяем нормализацию
        assert torch.max(torch.abs(original)) <= 1.0 + 1e-5
        
        # Проверяем симуляцию повреждений
        assert torch.sum(~mask) > 0  # Есть удаленные точки
        assert torch.sum(labels) > 0  # Есть метки дыр
    
    @pytest.mark.integration
    def test_dataloader_compatibility(self, temp_dir):
        """Тест совместимости с PyTorch DataLoader."""
        from torch.utils.data import DataLoader
        
        # Создаем несколько тестовых файлов
        for i in range(3):
            obj_file = temp_dir / f"test{i}.obj"
            obj_file.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3")
        
        with patch('trimesh.load') as mock_load:
            mock_mesh = Mock()
            mock_mesh.vertices = np.random.rand(100, 3)
            mock_load.return_value = mock_mesh
            
            with patch('trimesh.sample.sample_surface') as mock_sample:
                points = np.random.rand(512, 3).astype(np.float32)
                mock_sample.return_value = (points, None)
                
                dataset = MeshData(root_dir=str(temp_dir), num_points=512)
                dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
                
                # Тестируем один батч
                batch = next(iter(dataloader))
                
                assert len(batch) == 4  # original, damaged, mask, labels
                original, damaged, mask, labels = batch
                
                assert original.shape == (2, 512, 3)  # batch_size=2
                assert damaged.shape == (2, 512, 3)
                assert mask.shape == (2, 512)
                assert labels.shape == (2, 512)