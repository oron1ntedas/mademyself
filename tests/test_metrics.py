#Imports

#File Imports

# CHECK init metrics
# CHECK metrics culculation
# CHECK metrics visualization

"""
Тесты для модуля metrics.py - система метрик для оценки качества восстановления.
"""

import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Используем non-GUI backend для тестов
import matplotlib.pyplot as plt
from unittest.mock import Mock, patch, MagicMock
import sys
sys.path.insert(0, '/home/ubuntu/upload')

from metrics import (
    PointCloudMetrics, GeometricMetrics, QualityEvaluator, 
    VisualizationTools
)

class TestPointCloudMetrics:
    """Тесты для класса PointCloudMetrics."""
    
    def test_chamfer_distance_identical_clouds(self):
        """Тест Chamfer Distance для идентичных облаков точек."""
        pc1 = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=np.float32)
        pc2 = pc1.copy()
        
        distance = PointCloudMetrics.chamfer_distance(pc1, pc2)
        
        assert distance == pytest.approx(0.0, abs=1e-6)
    
    def test_chamfer_distance_different_clouds(self):
        """Тест Chamfer Distance для различных облаков точек."""
        pc1 = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32)
        pc2 = np.array([[0, 1, 0], [1, 1, 0], [2, 1, 0]], dtype=np.float32)
        
        distance = PointCloudMetrics.chamfer_distance(pc1, pc2)
        
        # Расстояние должно быть равно 2.0 (по 1.0 в каждую сторону)
        assert distance == pytest.approx(2.0, abs=1e-6)
    
    def test_chamfer_distance_single_points(self):
        """Тест Chamfer Distance для одиночных точек."""
        pc1 = np.array([[0, 0, 0]], dtype=np.float32)
        pc2 = np.array([[3, 4, 0]], dtype=np.float32)
        
        distance = PointCloudMetrics.chamfer_distance(pc1, pc2)
        
        # Расстояние должно быть 10.0 (5.0 в каждую сторону)
        expected_distance = 2 * np.sqrt(3**2 + 4**2)
        assert distance == pytest.approx(expected_distance, abs=1e-6)
    
    def test_chamfer_distance_different_sizes(self):
        """Тест Chamfer Distance для облаков разного размера."""
        pc1 = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
        pc2 = np.array([[0, 0, 0]], dtype=np.float32)
        
        distance = PointCloudMetrics.chamfer_distance(pc1, pc2)
        
        # Проверяем, что функция работает с разными размерами
        assert distance >= 0
        assert isinstance(distance, float)
    
    def test_earth_movers_distance_identical_clouds(self):
        """Тест Earth Mover's Distance для идентичных облаков."""
        pc1 = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)
        pc2 = pc1.copy()
        
        distance = PointCloudMetrics.earth_movers_distance(pc1, pc2)
        
        assert distance == pytest.approx(0.0, abs=1e-6)
    
    def test_earth_movers_distance_large_clouds(self):
        """Тест EMD для больших облаков точек (должен использовать сэмплинг)."""
        np.random.seed(42)
        pc1 = np.random.randn(1500, 3).astype(np.float32)
        pc2 = np.random.randn(1500, 3).astype(np.float32)
        
        distance = PointCloudMetrics.earth_movers_distance(pc1, pc2)
        
        assert distance >= 0
        assert isinstance(distance, float)
    
    def test_hausdorff_distance_identical_clouds(self):
        """Тест Hausdorff Distance для идентичных облаков."""
        pc1 = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)
        pc2 = pc1.copy()
        
        distance = PointCloudMetrics.hausdorff_distance(pc1, pc2)
        
        assert distance == pytest.approx(0.0, abs=1e-6)
    
    def test_hausdorff_distance_different_clouds(self):
        """Тест Hausdorff Distance для различных облаков."""
        pc1 = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
        pc2 = np.array([[0, 0, 0], [0, 2, 0]], dtype=np.float32)
        
        distance = PointCloudMetrics.hausdorff_distance(pc1, pc2)
        
        # Максимальное расстояние должно быть 2.0
        assert distance == pytest.approx(2.0, abs=1e-6)
    
    def test_coverage_score_perfect_coverage(self):
        """Тест Coverage Score для полного покрытия."""
        partial = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
        completed = partial.copy()
        
        score = PointCloudMetrics.coverage_score(partial, completed)
        
        assert score == pytest.approx(1.0, abs=1e-6)
    
    def test_coverage_score_no_coverage(self):
        """Тест Coverage Score для отсутствия покрытия."""
        partial = np.array([[0, 0, 0]], dtype=np.float32)
        completed = np.array([[10, 10, 10]], dtype=np.float32)
        
        score = PointCloudMetrics.coverage_score(partial, completed, threshold=0.1)
        
        assert score == pytest.approx(0.0, abs=1e-6)
    
    def test_coverage_score_partial_coverage(self):
        """Тест Coverage Score для частичного покрытия."""
        partial = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
        completed = np.array([[0, 0, 0], [10, 10, 10]], dtype=np.float32)
        
        score = PointCloudMetrics.coverage_score(partial, completed, threshold=0.1)
        
        assert score == pytest.approx(0.5, abs=1e-6)
    
    @patch('metrics.NearestNeighbors')
    def test_density_uniformity(self, mock_nn):
        """Тест оценки равномерности распределения точек."""
        # Настраиваем мок
        mock_instance = Mock()
        mock_instance.kneighbors.return_value = (
            np.array([[0, 1, 1.5, 2, 2.5]]),  # distances
            np.array([[0, 1, 2, 3, 4]])       # indices
        )
        mock_nn.return_value = mock_instance
        
        points = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32)
        
        uniformity = PointCloudMetrics.density_uniformity(points)
        
        assert 0 <= uniformity <= 1
        assert isinstance(uniformity, float)

class TestGeometricMetrics:
    """Тесты для класса GeometricMetrics."""
    
    def test_volume_similarity_identical(self):
        """Тест сравнения объемов для идентичных облаков."""
        pc1 = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)
        pc2 = pc1.copy()
        
        similarity = GeometricMetrics.volume_similarity(pc1, pc2)
        
        assert similarity == pytest.approx(1.0, abs=1e-6)
    
    def test_volume_similarity_different_scales(self):
        """Тест сравнения объемов для облаков разного масштаба."""
        pc1 = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)
        pc2 = np.array([[0, 0, 0], [2, 2, 2]], dtype=np.float32)
        
        similarity = GeometricMetrics.volume_similarity(pc1, pc2)
        
        # Объем второго облака в 8 раз больше, поэтому similarity = 1/8
        assert similarity == pytest.approx(1/8, abs=1e-6)
    
    def test_volume_similarity_zero_volume(self):
        """Тест сравнения объемов когда одно облако имеет нулевой объем."""
        pc1 = np.array([[0, 0, 0], [0, 0, 0]], dtype=np.float32)  # Нулевой объем
        pc2 = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)
        
        similarity = GeometricMetrics.volume_similarity(pc1, pc2)
        
        assert similarity == pytest.approx(0.0, abs=1e-6)
    
    def test_volume_similarity_both_zero_volume(self):
        """Тест сравнения объемов когда оба облака имеют нулевой объем."""
        pc1 = np.array([[0, 0, 0], [0, 0, 0]], dtype=np.float32)
        pc2 = np.array([[1, 1, 1], [1, 1, 1]], dtype=np.float32)
        
        similarity = GeometricMetrics.volume_similarity(pc1, pc2)
        
        assert similarity == pytest.approx(1.0, abs=1e-6)
    
    def test_surface_area_ratio_mock_mesh(self):
        """Тест сравнения площадей поверхности с мок-мешами."""
        # Создаем мок-меши
        mesh1 = Mock()
        mesh1.vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        mesh1.triangles = np.array([[0, 1, 2]])
        
        mesh2 = Mock()
        mesh2.vertices = np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0]])
        mesh2.triangles = np.array([[0, 1, 2]])
        
        ratio = GeometricMetrics.surface_area_ratio(mesh1, mesh2)
        
        # Площадь второго треугольника в 4 раза больше
        assert ratio == pytest.approx(1/4, abs=1e-6)
    
    def test_surface_area_ratio_empty_mesh(self):
        """Тест сравнения площадей для пустого меша."""
        mesh1 = Mock()
        mesh1.vertices = np.array([])
        mesh1.triangles = np.array([])
        
        mesh2 = Mock()
        mesh2.vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        mesh2.triangles = np.array([[0, 1, 2]])
        
        ratio = GeometricMetrics.surface_area_ratio(mesh1, mesh2)
        
        assert ratio == pytest.approx(0.0, abs=1e-6)
    
    def test_centroid_distance_identical(self):
        """Тест расстояния между центроидами для идентичных облаков."""
        pc1 = np.array([[0, 0, 0], [2, 2, 2]], dtype=np.float32)
        pc2 = pc1.copy()
        
        distance = GeometricMetrics.centroid_distance(pc1, pc2)
        
        assert distance == pytest.approx(0.0, abs=1e-6)
    
    def test_centroid_distance_different(self):
        """Тест расстояния между центроидами для различных облаков."""
        pc1 = np.array([[0, 0, 0], [2, 0, 0]], dtype=np.float32)  # Центроид: (1, 0, 0)
        pc2 = np.array([[0, 2, 0], [2, 2, 0]], dtype=np.float32)  # Центроид: (1, 2, 0)
        
        distance = GeometricMetrics.centroid_distance(pc1, pc2)
        
        assert distance == pytest.approx(2.0, abs=1e-6)
    
    def test_axis_alignment_error_identical(self):
        """Тест ошибок выравнивания для идентичных облаков."""
        pc1 = np.array([[0, 0, 0], [1, 2, 3]], dtype=np.float32)
        pc2 = pc1.copy()
        
        errors = GeometricMetrics.axis_alignment_error(pc1, pc2)
        
        assert errors['x'] == pytest.approx(0.0, abs=1e-6)
        assert errors['y'] == pytest.approx(0.0, abs=1e-6)
        assert errors['z'] == pytest.approx(0.0, abs=1e-6)
        assert errors['max_error'] == pytest.approx(0.0, abs=1e-6)
        assert errors['total_error'] == pytest.approx(0.0, abs=1e-6)
    
    def test_axis_alignment_error_different(self):
        """Тест ошибок выравнивания для различных облаков."""
        pc1 = np.array([[0, 0, 0], [2, 4, 6]], dtype=np.float32)  # Размах: [2, 4, 6]
        pc2 = np.array([[0, 0, 0], [1, 4, 6]], dtype=np.float32)  # Размах: [1, 4, 6]
        
        errors = GeometricMetrics.axis_alignment_error(pc1, pc2)
        
        assert errors['x'] == pytest.approx(0.5, abs=1e-6)  # |2-1|/2 = 0.5
        assert errors['y'] == pytest.approx(0.0, abs=1e-6)  # |4-4|/4 = 0.0
        assert errors['z'] == pytest.approx(0.0, abs=1e-6)  # |6-6|/6 = 0.0
        assert errors['max_error'] == pytest.approx(0.5, abs=1e-6)
        assert errors['total_error'] == pytest.approx(0.5, abs=1e-6)
    
    def test_axis_alignment_error_zero_range(self):
        """Тест ошибок выравнивания когда исходное облако имеет нулевой размах."""
        pc1 = np.array([[1, 1, 1], [1, 1, 1]], dtype=np.float32)  # Нулевой размах
        pc2 = np.array([[0, 0, 0], [2, 2, 2]], dtype=np.float32)
        
        errors = GeometricMetrics.axis_alignment_error(pc1, pc2)
        
        assert errors['x'] == pytest.approx(0.0, abs=1e-6)
        assert errors['y'] == pytest.approx(0.0, abs=1e-6)
        assert errors['z'] == pytest.approx(0.0, abs=1e-6)

class TestQualityEvaluator:
    """Тесты для класса QualityEvaluator."""
    
    def test_evaluate_reconstruction_basic(self, metrics_test_data):
        """Тест базовой оценки качества восстановления."""
        evaluator = QualityEvaluator()
        
        report = evaluator.evaluate_reconstruction(
            original=metrics_test_data['original'],
            damaged=metrics_test_data['damaged'],
            reconstructed=metrics_test_data['reconstructed']
        )
        
        # Проверяем структуру отчета
        assert 'point_cloud_metrics' in report
        assert 'geometric_metrics' in report
        assert 'reconstruction_metrics' in report
        assert 'overall_score' in report
        
        # Проверяем наличие основных метрик
        pc_metrics = report['point_cloud_metrics']
        assert 'chamfer_distance_original' in pc_metrics
        assert 'chamfer_distance_damaged' in pc_metrics
        assert 'hausdorff_distance' in pc_metrics
        assert 'coverage_score' in pc_metrics
        assert 'density_uniformity' in pc_metrics
        
        # Проверяем диапазоны значений
        assert 0 <= report['overall_score'] <= 1
        assert pc_metrics['coverage_score'] >= 0
        assert pc_metrics['density_uniformity'] >= 0
    
    def test_calculate_recovery_rate_perfect(self):
        """Тест расчета степени восстановления при идеальном восстановлении."""
        evaluator = QualityEvaluator()
        
        original = np.random.randn(100, 3)
        damaged = original[:60]  # Потеряли 40 точек
        reconstructed = np.vstack([damaged, np.random.randn(40, 3)])  # Добавили 40 точек
        
        rate = evaluator._calculate_recovery_rate(original, damaged, reconstructed)
        
        assert rate == pytest.approx(1.0, abs=1e-6)
    
    def test_calculate_recovery_rate_partial(self):
        """Тест расчета степени восстановления при частичном восстановлении."""
        evaluator = QualityEvaluator()
        
        original = np.random.randn(100, 3)
        damaged = original[:60]  # Потеряли 40 точек
        reconstructed = np.vstack([damaged, np.random.randn(20, 3)])  # Добавили только 20 точек
        
        rate = evaluator._calculate_recovery_rate(original, damaged, reconstructed)
        
        assert rate == pytest.approx(0.5, abs=1e-6)
    
    def test_calculate_recovery_rate_no_loss(self):
        """Тест расчета степени восстановления когда потерь не было."""
        evaluator = QualityEvaluator()
        
        original = np.random.randn(100, 3)
        damaged = original.copy()  # Потерь нет
        reconstructed = original.copy()
        
        rate = evaluator._calculate_recovery_rate(original, damaged, reconstructed)
        
        assert rate == pytest.approx(1.0, abs=1e-6)
    
    def test_calculate_quality_improvement(self):
        """Тест расчета улучшения качества."""
        evaluator = QualityEvaluator()
        
        original = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32)
        damaged = np.array([[0, 0, 0], [2, 0, 0]], dtype=np.float32)  # Потеряли среднюю точку
        reconstructed = np.array([[0, 0, 0], [1, 0.1, 0], [2, 0, 0]], dtype=np.float32)  # Восстановили с небольшой ошибкой
        
        improvement = evaluator._calculate_quality_improvement(original, damaged, reconstructed)
        
        assert 0 <= improvement <= 1
        assert isinstance(improvement, float)
    
    def test_calculate_overall_score_range(self, metrics_test_data):
        """Тест диапазона общего балла качества."""
        evaluator = QualityEvaluator()
        
        report = {
            'point_cloud_metrics': {
                'chamfer_distance_original': 0.1,
                'coverage_score': 0.8
            },
            'geometric_metrics': {
                'volume_similarity': 0.9,
                'axis_alignment_errors': {'max_error': 0.1}
            },
            'reconstruction_metrics': {
                'recovery_rate': 0.7
            }
        }
        
        score = evaluator._calculate_overall_score(report)
        
        assert 0 <= score <= 1
        assert isinstance(score, float)
    
    @patch('metrics.get_topology_features')
    def test_evaluate_topology_preservation(self, mock_topo):
        """Тест оценки сохранения топологии."""
        evaluator = QualityEvaluator()
        
        # Настраиваем мок
        mock_topo.return_value = {
            'vertices': 8,
            'edges': 12,
            'faces': 6,
            'euler_characteristic': 2
        }
        
        mesh1 = Mock()
        mesh2 = Mock()
        
        result = evaluator._evaluate_topology_preservation(mesh1, mesh2)
        
        assert 'original_topology' in result
        assert 'reconstructed_topology' in result
        assert 'euler_char_preserved' in result

class TestVisualizationTools:
    """Тесты для класса VisualizationTools."""
    
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.savefig')
    def test_plot_comparison(self, mock_savefig, mock_show, metrics_test_data):
        """Тест визуального сравнения облаков точек."""
        VisualizationTools.plot_comparison(
            original=metrics_test_data['original'],
            damaged=metrics_test_data['damaged'],
            reconstructed=metrics_test_data['reconstructed']
        )
        
        # Проверяем, что show был вызван
        mock_show.assert_called_once()
    
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.savefig')
    def test_plot_comparison_with_save(self, mock_savefig, mock_show, metrics_test_data, temp_dir):
        """Тест сохранения визуального сравнения."""
        save_path = temp_dir / "comparison.png"
        
        VisualizationTools.plot_comparison(
            original=metrics_test_data['original'],
            damaged=metrics_test_data['damaged'],
            reconstructed=metrics_test_data['reconstructed'],
            save_path=str(save_path)
        )
        
        # Проверяем, что savefig был вызван с правильным путем
        mock_savefig.assert_called_once_with(str(save_path), dpi=300, bbox_inches='tight')
        mock_show.assert_not_called()
    
    @patch('matplotlib.pyplot.savefig')
    def test_plot_metrics_dashboard(self, mock_savefig, temp_dir):
        """Тест создания dashboard с метриками."""
        evaluation_report = {
            'point_cloud_metrics': {
                'chamfer_distance_original': 0.1,
                'coverage_score': 0.8,
                'hausdorff_distance': 0.2
            },
            'geometric_metrics': {
                'volume_similarity': 0.9,
                'centroid_distance': 0.05
            },
            'reconstruction_metrics': {
                'recovery_rate': 0.7,
                'quality_improvement': 0.6
            },
            'overall_score': 0.75
        }
        
        save_path = temp_dir / "dashboard.png"
        
        VisualizationTools.plot_metrics_dashboard(
            evaluation_report,
            save_path=str(save_path)
        )
        
        # Проверяем, что savefig был вызван
        mock_savefig.assert_called_once()

class TestIntegration:
    """Интеграционные тесты для модуля metrics."""
    
    @pytest.mark.integration
    def test_full_evaluation_pipeline(self, metrics_test_data, temp_dir):
        """Тест полного pipeline оценки качества."""
        evaluator = QualityEvaluator()
        
        # Выполняем полную оценку
        report = evaluator.evaluate_reconstruction(
            original=metrics_test_data['original'],
            damaged=metrics_test_data['damaged'],
            reconstructed=metrics_test_data['reconstructed']
        )
        
        # Проверяем, что все компоненты работают
        assert isinstance(report, dict)
        assert 'overall_score' in report
        assert 0 <= report['overall_score'] <= 1
        
        # Проверяем наличие всех ожидаемых метрик
        required_sections = [
            'point_cloud_metrics',
            'geometric_metrics', 
            'reconstruction_metrics'
        ]
        
        for section in required_sections:
            assert section in report
            assert isinstance(report[section], dict)
    
    @pytest.mark.slow
    def test_large_pointcloud_evaluation(self):
        """Тест оценки качества для больших облаков точек."""
        np.random.seed(42)
        
        # Создаем большие облака точек
        original = np.random.randn(5000, 3).astype(np.float32)
        damaged = original[:3000]  # Удаляем 40% точек
        reconstructed = np.vstack([
            damaged,
            original[3000:] + np.random.normal(0, 0.1, (2000, 3))
        ]).astype(np.float32)
        
        evaluator = QualityEvaluator()
        report = evaluator.evaluate_reconstruction(original, damaged, reconstructed)
        
        assert isinstance(report['overall_score'], float)
        assert 0 <= report['overall_score'] <= 1