#Imports
import numpy as np
import logging
from typing import Dict, Optional
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from pathlib import Path

logger = logging.getLogger(__name__)

class PointCloudMetrics:
    """Метрики для облаков точек"""
    
    @staticmethod
    def chamfer_distance(pc1: np.ndarray, pc2: np.ndarray) -> float:
        """
        Chamfer Distance между двумя облаками точек
        
        Args:
            pc1, pc2: облака точек [N, 3] и [M, 3]
            
        Returns:
            chamfer_dist: симметричное расстояние Chamfer
        """
        # ТВОЯ ЗАДАЧА: Реализуй эффективный Chamfer Distance
        # Формула: 1/|P1| * sum(min_dist(p, P2)) + 1/|P2| * sum(min_dist(q, P1))
        
        # Расстояния от каждой точки pc1 до ближайшей в pc2
        dists_1_to_2 = cdist(pc1, pc2)
        min_dists_1_to_2 = np.min(dists_1_to_2, axis=1)
        
        # Расстояния от каждой точки pc2 до ближайшей в pc1
        dists_2_to_1 = cdist(pc2, pc1)
        min_dists_2_to_1 = np.min(dists_2_to_1, axis=1)
        
        # Симметричный Chamfer Distance
        chamfer = np.mean(min_dists_1_to_2) + np.mean(min_dists_2_to_1)
        
        return chamfer
    
    @staticmethod
    def earth_movers_distance(pc1: np.ndarray, pc2: np.ndarray) -> float:
        """
        Earth Mover's Distance (Wasserstein distance)
        Приближенная версия для небольших облаков точек
        """
        # ТВОЯ ЗАДАЧА: Реализуй упрощенную версию EMD
        # Для точной версии потребуется scipy.stats.wasserstein_distance или POT библиотека
        
        if len(pc1) > 1000 or len(pc2) > 1000:
            # Для больших облаков используем сэмплинг
            sample_size = 500
            if len(pc1) > sample_size:
                indices1 = np.random.choice(len(pc1), sample_size, replace=False)
                pc1 = pc1[indices1]
            if len(pc2) > sample_size:
                indices2 = np.random.choice(len(pc2), sample_size, replace=False)
                pc2 = pc2[indices2]
        
        # Упрощенная версия: средняя стоимость перемещения
        dist_matrix = cdist(pc1, pc2)
        
        # Минимальные расстояния (приближение к оптимальному транспорту)
        assignment_cost = np.sum(np.min(dist_matrix, axis=1)) / len(pc1)
        
        return assignment_cost
    
    @staticmethod
    def hausdorff_distance(pc1: np.ndarray, pc2: np.ndarray) -> float:
        """Hausdorff Distance"""
        # ТВОЯ ЗАДАЧА: Реализуй Hausdorff Distance
        # max(max_min_dist(P1, P2), max_min_dist(P2, P1))
        
        dists_1_to_2 = cdist(pc1, pc2)
        dists_2_to_1 = cdist(pc2, pc1)
        
        max_min_1_to_2 = np.max(np.min(dists_1_to_2, axis=1))
        max_min_2_to_1 = np.max(np.min(dists_2_to_1, axis=1))
        
        return max(max_min_1_to_2, max_min_2_to_1)
    
    @staticmethod
    def coverage_score(partial: np.ndarray, completed: np.ndarray, threshold: float = 0.05) -> float:
        """
        Оценка покрытия: какая доля исходных точек покрыта восстановленными
        """
        dists = cdist(partial, completed)
        min_dists = np.min(dists, axis=1)
        covered_points = np.sum(min_dists < threshold)
        
        return covered_points / len(partial)
    
    @staticmethod
    def density_uniformity(points: np.ndarray, k_neighbors: int = 10) -> float:
        """
        Оценка равномерности распределения точек
        """
        from sklearn.neighbors import NearestNeighbors
        
        nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1).fit(points)
        distances, _ = nbrs.kneighbors(points)
        
        # Исключаем расстояние до самой точки (0)
        neighbor_distances = distances[:, 1:]
        avg_distances = np.mean(neighbor_distances, axis=1)
        
        # Коэффициент вариации как мера неравномерности
        uniformity = 1.0 - (np.std(avg_distances) / np.mean(avg_distances))
        
        return max(0, uniformity)  # ограничиваем снизу нулем

class GeometricMetrics:
    """Геометрические метрики для 3D моделей"""
    
    @staticmethod
    def volume_similarity(pc1: np.ndarray, pc2: np.ndarray) -> float:
        """Сравнение объемов bounding box"""
        vol1 = np.prod(np.ptp(pc1, axis=0))  # объем bounding box
        vol2 = np.prod(np.ptp(pc2, axis=0))
        
        if vol1 == 0 and vol2 == 0:
            return 1.0
        if vol1 == 0 or vol2 == 0:
            return 0.0
        
        return min(vol1, vol2) / max(vol1, vol2)
    
    @staticmethod
    def surface_area_ratio(mesh1, mesh2) -> float:
        """Сравнение площадей поверхности (для mesh)"""
        # ТВОЯ ЗАДАЧА: Реализуй вычисление площади поверхности
        # Площадь треугольника = 0.5 * ||cross_product||
        
        def mesh_surface_area(mesh):
            if not hasattr(mesh, 'triangles') or len(mesh.triangles) == 0:
                return 0.0
            
            vertices = np.asarray(mesh.vertices)
            triangles = np.asarray(mesh.triangles)
            
            total_area = 0.0
            for tri in triangles:
                v1, v2, v3 = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]
                area = 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1))
                total_area += area
            
            return total_area
        
        area1 = mesh_surface_area(mesh1)
        area2 = mesh_surface_area(mesh2)
        
        if area1 == 0 and area2 == 0:
            return 1.0
        if area1 == 0 or area2 == 0:
            return 0.0
        
        return min(area1, area2) / max(area1, area2)
    
    @staticmethod
    def centroid_distance(pc1: np.ndarray, pc2: np.ndarray) -> float:
        """Расстояние между центроидами"""
        centroid1 = np.mean(pc1, axis=0)
        centroid2 = np.mean(pc2, axis=0)
        
        return np.linalg.norm(centroid1 - centroid2)
    
    @staticmethod
    def axis_alignment_error(original: np.ndarray, reconstructed: np.ndarray) -> Dict[str, float]:
        """Ошибки выравнивания по осям (твоя основная проблема)"""
        orig_ranges = np.ptp(original, axis=0)  # размах по каждой оси
        recon_ranges = np.ptp(reconstructed, axis=0)
        
        # Нормализуем ошибки по размеру исходной модели
        axis_errors = {}
        for i, axis in enumerate(['x', 'y', 'z']):
            if orig_ranges[i] > 0:
                relative_error = abs(orig_ranges[i] - recon_ranges[i]) / orig_ranges[i]
                axis_errors[axis] = relative_error
            else:
                axis_errors[axis] = 0.0
        
        axis_errors['max_error'] = max(axis_errors['x'], axis_errors['y'], axis_errors['z'])
        axis_errors['total_error'] = axis_errors['x'] + axis_errors['y'] + axis_errors['z']
        
        return axis_errors

class QualityEvaluator:
    """Комплексная система оценки качества"""
    
    def __init__(self):
        self.results = {}
    
    def evaluate_reconstruction(self, 
                              original: np.ndarray,
                              damaged: np.ndarray,
                              reconstructed: np.ndarray,
                              mesh_original=None,
                              mesh_reconstructed=None) -> Dict:
        """
        Полная оценка качества реконструкции
        
        Args:
            original: исходное облако точек
            damaged: поврежденное облако точек
            reconstructed: восстановленное облако точек
            mesh_original, mesh_reconstructed: соответствующие mesh (опционально)
            
        Returns:
            evaluation_report: детальный отчет о качестве
        """
        logger.info("Начало комплексной оценки качества...")
        
        report = {
            'point_cloud_metrics': {},
            'geometric_metrics': {},
            'reconstruction_metrics': {},
            'overall_score': 0.0
        }
        
        # Метрики облаков точек
        pc_metrics = PointCloudMetrics()
        report['point_cloud_metrics'] = {
            'chamfer_distance_original': pc_metrics.chamfer_distance(original, reconstructed),
            'chamfer_distance_damaged': pc_metrics.chamfer_distance(damaged, reconstructed),
            'hausdorff_distance': pc_metrics.hausdorff_distance(original, reconstructed),
            'coverage_score': pc_metrics.coverage_score(damaged, reconstructed),
            'density_uniformity': pc_metrics.density_uniformity(reconstructed)
        }
        
        # ТВОЯ ЗАДАЧА: Добавь EMD, если нужно точное сравнение распределений
        try:
            report['point_cloud_metrics']['earth_movers_distance'] = pc_metrics.earth_movers_distance(original, reconstructed)
        except Exception as e:
            logger.warning(f"Не удалось вычислить EMD: {e}")
            report['point_cloud_metrics']['earth_movers_distance'] = None
        
        # Геометрические метрики
        geom_metrics = GeometricMetrics()
        report['geometric_metrics'] = {
            'volume_similarity': geom_metrics.volume_similarity(original, reconstructed),
            'centroid_distance': geom_metrics.centroid_distance(original, reconstructed),
            'axis_alignment_errors': geom_metrics.axis_alignment_error(original, reconstructed)
        }
        
        # Метрики восстановления
        report['reconstruction_metrics'] = {
            'points_added': len(reconstructed) - len(damaged),
            'completion_ratio': len(reconstructed) / len(original) if len(original) > 0 else 0,
            'recovery_rate': self._calculate_recovery_rate(original, damaged, reconstructed),
            'quality_improvement': self._calculate_quality_improvement(original, damaged, reconstructed)
        }
        
        # Если есть mesh - добавляем mesh-метрики
        if mesh_original and mesh_reconstructed:
            report['mesh_metrics'] = {
                'surface_area_ratio': geom_metrics.surface_area_ratio(mesh_original, mesh_reconstructed),
                'topology_preservation': self._evaluate_topology_preservation(mesh_original, mesh_reconstructed)
            }
        
        # Общая оценка
        report['overall_score'] = self._calculate_overall_score(report)
        
        logger.info(f"Оценка завершена. Общий балл: {report['overall_score']:.3f}")
        return report
    
    def _calculate_recovery_rate(self, original: np.ndarray, damaged: np.ndarray, reconstructed: np.ndarray) -> float:
        """Оценка степени восстановления"""
        # Точки, которые были потеряны
        missing_points_count = len(original) - len(damaged)
        if missing_points_count <= 0:
            return 1.0  # ничего не было потеряно
        
        # Точки, которые были добавлены
        added_points_count = len(reconstructed) - len(damaged)
        
        return min(1.0, added_points_count / missing_points_count)
    
    def _calculate_quality_improvement(self, original: np.ndarray, damaged: np.ndarray, reconstructed: np.ndarray) -> float:
        """Улучшение качества по сравнению с поврежденной моделью"""
        pc_metrics = PointCloudMetrics()
        
        # Качество поврежденной модели относительно оригинала
        damaged_quality = pc_metrics.chamfer_distance(original, damaged)
        
        # Качество восстановленной модели относительно оригинала
        reconstructed_quality = pc_metrics.chamfer_distance(original, reconstructed)
        
        if damaged_quality == 0:
            return 1.0 if reconstructed_quality == 0 else 0.0
        
        # Улучшение в процентах
        improvement = (damaged_quality - reconstructed_quality) / damaged_quality
        return max(0.0, improvement)  # не может быть отрицательным
    
    def _evaluate_topology_preservation(self, mesh1, mesh2) -> Dict:
        """Оценка сохранения топологии"""
        # ТВОЯ ЗАДАЧА: Реализуй сравнение топологических характеристик
        # Euler characteristic, genus, количество компонент связности
        
        def get_topology_features(mesh):
            vertices = len(mesh.vertices)
            edges = len(mesh.get_non_manifold_edges()) if hasattr(mesh, 'get_non_manifold_edges') else 0
            faces = len(mesh.triangles)
            
            # Эйлерова характеристика: V - E + F
            euler_char = vertices - edges + faces
            
            return {
                'vertices': vertices,
                'edges': edges,
                'faces': faces,
                'euler_characteristic': euler_char
            }
        
        topo1 = get_topology_features(mesh1)
        topo2 = get_topology_features(mesh2)
        
        return {
            'original_topology': topo1,
            'reconstructed_topology': topo2,
            'euler_char_preserved': abs(topo1['euler_characteristic'] - topo2['euler_characteristic']) < 2
        }
    
    def _calculate_overall_score(self, report: Dict) -> float:
        """Вычисление общего балла качества"""
        # ТВОЯ ЗАДАЧА: Настрой веса для разных метрик в зависимости от приоритетов
        
        scores = []
        weights = []
        
        # Основные метрики с весами
        pc_metrics = report['point_cloud_metrics']
        
        # Chamfer distance (инвертируем, чтобы больше = лучше)
        if pc_metrics['chamfer_distance_original'] > 0:
            cd_score = 1.0 / (1.0 + pc_metrics['chamfer_distance_original'])
            scores.append(cd_score)
            weights.append(3.0)  # высокий вес
        
        # Coverage score
        scores.append(pc_metrics['coverage_score'])
        weights.append(2.0)
        
        # Volume similarity
        geom_metrics = report['geometric_metrics']
        scores.append(geom_metrics['volume_similarity'])
        weights.append(1.5)
        
        # Axis alignment (важно для твоей задачи)
        axis_errors = geom_metrics['axis_alignment_errors']
        axis_score = 1.0 - min(1.0, axis_errors['max_error'])  # инвертируем ошибку
        scores.append(axis_score)
        weights.append(2.5)  # высокий вес для твоей проблемы
        
        # Recovery rate
        recon_metrics = report['reconstruction_metrics']
        scores.append(recon_metrics['recovery_rate'])
        weights.append(1.0)
        
        # Взвешенное среднее
        if len(scores) > 0:
            weighted_sum = sum(s * w for s, w in zip(scores, weights))
            total_weight = sum(weights)
            overall_score = weighted_sum / total_weight
        else:
            overall_score = 0.0
        
        return overall_score

class VisualizationTools:
    """Инструменты визуализации результатов"""
    
    @staticmethod
    def plot_comparison(original: np.ndarray, damaged: np.ndarray, reconstructed: np.ndarray,
                       save_path: Optional[str] = None):
        """Визуальное сравнение облаков точек"""
        # ТВОЯ ЗАДАЧА: Создай наглядную визуализацию для отчета
        
        fig = plt.figure(figsize=(15, 5))
        
        # Исходное облако
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.scatter(original[:, 0], original[:, 1], original[:, 2], c='blue', s=1, alpha=0.6)
        ax1.set_title('Original')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        
        # Поврежденное облако
        ax2 = fig.add_subplot(132, projection='3d')
        ax2.scatter(damaged[:, 0], damaged[:, 1], damaged[:, 2], c='red', s=1, alpha=0.6)
        ax2.set_title('Damaged')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        
        # Восстановленное облако
        ax3 = fig.add_subplot(133, projection='3d')
        ax3.scatter(reconstructed[:, 0], reconstructed[:, 1], reconstructed[:, 2], c='green', s=1, alpha=0.6)
        ax3.set_title('Reconstructed')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_zlabel('Z')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    @staticmethod
    def plot_metrics_dashboard(evaluation_report: Dict, save_path: Optional[str] = None):
        """Dashboard с метриками качества"""
        # ТВОЯ ЗАДАЧА: Создай информативный dashboard
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # График 1: Основные метрики
        pc_metrics = evaluation_report['point_cloud_metrics']
        metrics_names = ['Chamfer Dist', 'Coverage', 'Density Uniformity']
        metrics_values = [
            1.0 / (1.0 + pc_metrics['chamfer_distance_original']),  # инвертируем для отображения
            pc_metrics['coverage_score'],
            pc_metrics['density_uniformity']
        ]
        
        axes[0, 0].bar(metrics_names, metrics_values, color=['skyblue', 'lightgreen', 'orange'])
        axes[0, 0].set_title('Point Cloud Metrics')
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # График 2: Ошибки по осям
        axis_errors = evaluation_report['geometric_metrics']['axis_alignment_errors']
        axis_names = ['X', 'Y', 'Z']
        error_values = [axis_errors['x'], axis_errors['y'], axis_errors['z']]
        
        colors = ['red' if e > 0.1 else 'green' for e in error_values]  # красный для больших ошибок
        axes[0, 1].bar(axis_names, error_values, color=colors)
        axes[0, 1].set_title('Axis Alignment Errors')
        axes[0, 1].set_ylabel('Relative Error')
        
        # График 3: Reconstruction metrics
        recon_metrics = evaluation_report['reconstruction_metrics']
        recon_names = ['Recovery Rate', 'Quality Improvement', 'Completion Ratio']
        recon_values = [
            recon_metrics['recovery_rate'],
            recon_metrics['quality_improvement'],
            min(1.0, recon_metrics['completion_ratio'])  # ограничиваем для отображения
        ]
        
        axes[1, 0].bar(recon_names, recon_values, color=['purple', 'brown', 'pink'])
        axes[1, 0].set_title('Reconstruction Quality')
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # График 4: Общий скор
        overall_score = evaluation_report['overall_score']
        axes[1, 1].pie([overall_score, 1 - overall_score], 
                      labels=['Quality Score', 'Room for Improvement'],
                      colors=['lightcoral', 'lightgray'],
                      startangle=90)
        axes[1, 1].set_title(f'Overall Score: {overall_score:.3f}')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

# Интеграционные функции
def comprehensive_evaluation(original: np.ndarray, damaged: np.ndarray, reconstructed: np.ndarray,
                           mesh_original=None, mesh_reconstructed=None,
                           save_visualizations: bool = True, output_dir: str = "evaluation_results") -> Dict:
    """
    Полная оценка качества с визуализацией
    
    Returns:
        evaluation_report: полный отчет с метриками и визуализациями
    """
    # Создаем директорию для результатов
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Оценка качества
    evaluator = QualityEvaluator()
    report = evaluator.evaluate_reconstruction(
        original, damaged, reconstructed, 
        mesh_original, mesh_reconstructed
    )
    
    # Визуализация
    if save_visualizations:
        viz = VisualizationTools()
        
        # Сравнение облаков точек
        viz.plot_comparison(
            original, damaged, reconstructed,
            save_path=str(output_path / "point_clouds_comparison.png")
        )
        
        # Dashboard с метриками
        viz.plot_metrics_dashboard(
            report,
            save_path=str(output_path / "metrics_dashboard.png")
        )
    
    # Сохраняем отчет в файл
    import json
    
    # Подготавливаем отчет для JSON (убираем объекты, которые нельзя сериализовать)
    json_report = {}
    for key, value in report.items():
        if isinstance(value, dict):
            json_report[key] = {k: float(v) if isinstance(v, (np.float32, np.float64)) else v 
                              for k, v in value.items() if not isinstance(v, (np.ndarray,))}
        else:
            json_report[key] = float(value) if isinstance(value, (np.float32, np.float64)) else value
    
    with open(output_path / "evaluation_report.json", "w") as f:
        json.dump(json_report, f, indent=2)
    
    logger.info(f"Результаты оценки сохранены в {output_path}")
    
    return report

# Пример использования
if __name__ == "__main__":
    # Создание тестовых данных
    original = np.random.randn(1000, 3).astype(np.float32)
    
    # Имитируем повреждение (удаляем 70% точек)
    keep_indices = np.random.choice(len(original), int(0.3 * len(original)), replace=False)
    damaged = original[keep_indices]
    
    # Имитируем восстановление (добавляем точки обратно с шумом)
    reconstructed = np.vstack([
        damaged,
        original[np.setdiff1d(range(len(original)), keep_indices)] + np.random.normal(0, 0.1, (len(original) - len(damaged), 3))
    ])
    
    try:
        # Полная оценка
        report = comprehensive_evaluation(original, damaged, reconstructed)
        
        print(f"Общий балл качества: {report['overall_score']:.3f}")
        print(f"Chamfer Distance: {report['point_cloud_metrics']['chamfer_distance_original']:.6f}")
        print(f"Coverage Score: {report['point_cloud_metrics']['coverage_score']:.3f}")
        print(f"Ошибки по осям: {report['geometric_metrics']['axis_alignment_errors']}")
        
    except Exception as e:
        print(f"Ошибка оценки: {e}")

