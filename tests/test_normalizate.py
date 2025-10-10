#Imports

#File Imports

# CHECK normalize pointcloud

# !!! check is norm_pointcloud clear_3dmodel is similar to norm_pointcloud crush_3dmodel!!!

"""
Тесты для модуля normalize.py - нормализация облаков точек.
"""

import pytest
import numpy as np
import torch
import sys
sys.path.insert(0, '/home/ubuntu/upload')

from preprocess.normalize import norm_pointcloud

class TestNormPointcloud:
    """Тесты для функции norm_pointcloud."""
    
    def test_norm_pointcloud_basic(self):
        """Тест базовой нормализации облака точек."""
        # Создаем простое облако точек
        points = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ], dtype=np.float32)
        
        result = norm_pointcloud(points)
        
        # Проверяем тип результата
        assert isinstance(result, torch.Tensor)
        assert result.dtype == torch.float32
        assert result.shape == (3, 3)
        
        # Проверяем, что результат центрирован (среднее близко к нулю)
        mean = torch.mean(result, dim=0)
        assert torch.allclose(mean, torch.zeros(3), atol=1e-6)
        
        # Проверяем, что максимальное абсолютное значение равно 1
        max_abs_val = torch.max(torch.abs(result))
        assert torch.allclose(max_abs_val, torch.tensor(1.0), atol=1e-6)
    
    def test_norm_pointcloud_single_point(self):
        """Тест нормализации одной точки."""
        points = np.array([[5.0, 10.0, -3.0]], dtype=np.float32)
        result = norm_pointcloud(points)
        
        # Одна точка после центрирования становится нулевой
        expected = torch.zeros(1, 3)
        assert torch.allclose(result, expected, atol=1e-6)
    
    def test_norm_pointcloud_zero_centered(self):
        """Тест нормализации уже центрированного облака."""
        points = np.array([
            [-1.0, -1.0, -1.0],
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0]
        ], dtype=np.float32)
        
        result = norm_pointcloud(points)
        
        # Проверяем центрирование
        mean = torch.mean(result, dim=0)
        assert torch.allclose(mean, torch.zeros(3), atol=1e-6)
        
        # Проверяем масштабирование
        max_abs_val = torch.max(torch.abs(result))
        assert torch.allclose(max_abs_val, torch.tensor(1.0), atol=1e-6)
    
    def test_norm_pointcloud_cube_vertices(self, sample_pointcloud):
        """Тест нормализации вершин куба."""
        result = norm_pointcloud(sample_pointcloud)
        
        # Проверяем форму
        assert result.shape == sample_pointcloud.shape
        
        # Проверяем центрирование
        mean = torch.mean(result, dim=0)
        assert torch.allclose(mean, torch.zeros(3), atol=1e-5)
        
        # Проверяем масштабирование
        max_abs_val = torch.max(torch.abs(result))
        assert torch.allclose(max_abs_val, torch.tensor(1.0), atol=1e-5)
    
    def test_norm_pointcloud_asymmetric(self):
        """Тест нормализации асимметричного облака точек."""
        points = np.array([
            [0.0, 0.0, 0.0],
            [10.0, 5.0, 2.0],
            [20.0, 10.0, 4.0]
        ], dtype=np.float32)
        
        result = norm_pointcloud(points)
        
        # Проверяем центрирование
        mean = torch.mean(result, dim=0)
        assert torch.allclose(mean, torch.zeros(3), atol=1e-6)
        
        # Максимальное абсолютное значение должно быть 1
        max_abs_val = torch.max(torch.abs(result))
        assert torch.allclose(max_abs_val, torch.tensor(1.0), atol=1e-6)
    
    def test_norm_pointcloud_negative_values(self):
        """Тест нормализации с отрицательными значениями."""
        points = np.array([
            [-5.0, -10.0, -15.0],
            [0.0, 0.0, 0.0],
            [5.0, 10.0, 15.0]
        ], dtype=np.float32)
        
        result = norm_pointcloud(points)
        
        # Проверяем центрирование
        mean = torch.mean(result, dim=0)
        assert torch.allclose(mean, torch.zeros(3), atol=1e-6)
        
        # Проверяем масштабирование
        max_abs_val = torch.max(torch.abs(result))
        assert torch.allclose(max_abs_val, torch.tensor(1.0), atol=1e-6)
    
    def test_norm_pointcloud_large_scale(self):
        """Тест нормализации облака с большими координатами."""
        points = np.array([
            [1000.0, 2000.0, 3000.0],
            [1001.0, 2001.0, 3001.0],
            [1002.0, 2002.0, 3002.0]
        ], dtype=np.float32)
        
        result = norm_pointcloud(points)
        
        # Проверяем центрирование
        mean = torch.mean(result, dim=0)
        assert torch.allclose(mean, torch.zeros(3), atol=1e-5)
        
        # Проверяем масштабирование
        max_abs_val = torch.max(torch.abs(result))
        assert torch.allclose(max_abs_val, torch.tensor(1.0), atol=1e-5)
    
    def test_norm_pointcloud_small_scale(self):
        """Тест нормализации облака с малыми координатами."""
        points = np.array([
            [0.001, 0.002, 0.003],
            [0.004, 0.005, 0.006],
            [0.007, 0.008, 0.009]
        ], dtype=np.float32)
        
        result = norm_pointcloud(points)
        
        # Проверяем центрирование
        mean = torch.mean(result, dim=0)
        assert torch.allclose(mean, torch.zeros(3), atol=1e-6)
        
        # Проверяем масштабирование
        max_abs_val = torch.max(torch.abs(result))
        assert torch.allclose(max_abs_val, torch.tensor(1.0), atol=1e-6)
    
    def test_norm_pointcloud_identical_points(self):
        """Тест нормализации идентичных точек."""
        points = np.array([
            [5.0, 5.0, 5.0],
            [5.0, 5.0, 5.0],
            [5.0, 5.0, 5.0]
        ], dtype=np.float32)
        
        result = norm_pointcloud(points)
        
        # Все точки должны стать нулевыми после центрирования
        expected = torch.zeros(3, 3)
        assert torch.allclose(result, expected, atol=1e-6)
    
    def test_norm_pointcloud_zero_points(self):
        """Тест нормализации нулевых точек."""
        points = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0]
        ], dtype=np.float32)
        
        result = norm_pointcloud(points)
        
        # Все точки должны остаться нулевыми
        expected = torch.zeros(3, 3)
        assert torch.allclose(result, expected, atol=1e-6)
    
    def test_norm_pointcloud_input_types(self):
        """Тест различных типов входных данных."""
        # Тест с float64
        points_f64 = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
        result_f64 = norm_pointcloud(points_f64)
        assert result_f64.dtype == torch.float32
        
        # Тест с int32
        points_int = np.array([[1, 2, 3]], dtype=np.int32)
        result_int = norm_pointcloud(points_int)
        assert result_int.dtype == torch.float32
    
    def test_norm_pointcloud_empty_array(self):
        """Тест нормализации пустого массива."""
        points = np.empty((0, 3), dtype=np.float32)
        result = norm_pointcloud(points)
        
        assert result.shape == (0, 3)
        assert result.dtype == torch.float32
    
    def test_norm_pointcloud_preserves_relative_positions(self):
        """Тест сохранения относительных позиций точек."""
        points = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        
        result = norm_pointcloud(points)
        
        # Проверяем, что относительные расстояния сохраняются пропорционально
        original_distances = []
        normalized_distances = []
        
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                orig_dist = np.linalg.norm(points[i] - points[j])
                norm_dist = torch.norm(result[i] - result[j]).item()
                original_distances.append(orig_dist)
                normalized_distances.append(norm_dist)
        
        # Проверяем пропорциональность расстояний
        if len(original_distances) > 1:
            orig_ratios = [original_distances[i] / original_distances[0] 
                          for i in range(1, len(original_distances))]
            norm_ratios = [normalized_distances[i] / normalized_distances[0] 
                          for i in range(1, len(normalized_distances))]
            
            for orig_ratio, norm_ratio in zip(orig_ratios, norm_ratios):
                assert abs(orig_ratio - norm_ratio) < 1e-5
    
    @pytest.mark.parametrize("num_points", [10, 100, 1000])
    def test_norm_pointcloud_various_sizes(self, num_points):
        """Параметризованный тест для различных размеров облаков точек."""
        np.random.seed(42)
        points = np.random.randn(num_points, 3).astype(np.float32)
        
        result = norm_pointcloud(points)
        
        assert result.shape == (num_points, 3)
        assert result.dtype == torch.float32
        
        # Проверяем центрирование
        mean = torch.mean(result, dim=0)
        assert torch.allclose(mean, torch.zeros(3), atol=1e-5)
        
        # Проверяем масштабирование
        max_abs_val = torch.max(torch.abs(result))
        assert torch.allclose(max_abs_val, torch.tensor(1.0), atol=1e-5)
    
    def test_norm_pointcloud_deterministic(self):
        """Тест детерминированности функции."""
        points = np.random.rand(100, 3).astype(np.float32)
        
        result1 = norm_pointcloud(points)
        result2 = norm_pointcloud(points)
        
        assert torch.allclose(result1, result2)
    
    def test_norm_pointcloud_numerical_stability(self):
        """Тест численной стабильности при экстремальных значениях."""
        # Тест с очень большими значениями
        large_points = np.array([
            [1e6, 1e6, 1e6],
            [1e6 + 1, 1e6 + 1, 1e6 + 1]
        ], dtype=np.float32)
        
        result_large = norm_pointcloud(large_points)
        assert torch.isfinite(result_large).all()
        
        # Тест с очень малыми значениями
        small_points = np.array([
            [1e-6, 1e-6, 1e-6],
            [2e-6, 2e-6, 2e-6]
        ], dtype=np.float32)
        
        result_small = norm_pointcloud(small_points)
        assert torch.isfinite(result_small).all()