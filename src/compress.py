# Imports
import numpy as np
import DracoPy as draco  
import logging
from typing import Optional, Dict

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DracoCodec:
    """
    Класс для сжатия и разжатия облаков точек с использованием DracoPy.
    """
    def __init__(self, quantization_bits: int = 14,
                 compression_level: int = 1):
        # DracoPy использует параметры напрямую в функции encode.
        # Сохраняем их как атрибуты класса.
        self.quantization_bits = quantization_bits
        self.compression_level = compression_level
        self.compression_stats: Dict[str, float] = {}
        
    def compress_pointcloud(self, points: np.ndarray) -> Optional[bytes]:
        """
        Сжимает облако точек (N, 3) в бинарный формат Draco.
        """
        if points.ndim != 2 or points.shape[1] != 3:
            logger.error("Некорректный формат входных данных. Ожидается (N, 3).")
            return None
        
        try:
            # В DracoPy нет необходимости создавать объект PointCloud 
            # и вызывать add_point_attribute.
            # Функция encode принимает массив точек напрямую.
            # Если faces опущен, DracoPy кодирует облако точек.
            compressed_data = draco.encode(
                points, 
                quantization_bits=self.quantization_bits,  # Уровень квантования для позиции
                compression_level=self.compression_level   # Уровень сжатия (0-10)
            )
            
            if compressed_data:
                # np.ndarray.nbytes возвращает размер массива в байтах.
                original_size = points.nbytes
                compressed_size = len(compressed_data)
                compression_ratio = original_size / compressed_size if compressed_size > 0 else 0
                
                self.compression_stats = {
                    'original_size': float(original_size),
                    'compressed_size': float(compressed_size),
                    'compression_ratio': float(compression_ratio),
                    'original_points': float(len(points))
                }
                logger.info(f"Сжатие успешно. Коэффициент сжатия: {compression_ratio:.2f}")
            
            return compressed_data
            
        except Exception as e:
            logger.error(f"Ошибка сжатия: {e}")
            return None

    def decompress_pointcloud(self, compressed_data: bytes) -> Optional[np.ndarray]:
        """
        Разжимает бинарные данные Draco обратно в облако точек (N, 3) NumPy.
        """
        try:
            # DracoPy.decode принимает бинарные данные.
            decoded_mesh_or_pc = draco.decode(compressed_data)
            
            # В случае облака точек, декодированный объект содержит точки в атрибуте .points
            if hasattr(decoded_mesh_or_pc, 'points'):
                points = decoded_mesh_or_pc.points
                return points
            
            logger.error("Декодирование не вернуло ожидаемое облако точек.")
            return None
            
        except Exception as e:
            logger.error(f"Ошибка разжатия: {e}")
            return None

# Пример использования (показан на английском для удобства)
if __name__ == '__main__':
    # 1. Создание тестового облака точек
    num_points = 1000
    test_points = np.random.rand(num_points, 3).astype(np.float32)
    
    # 2. Инициализация кодека
    codec = DracoCodec(quantization_bits=11, compression_level=7)
    
    # 3. Сжатие
    compressed_data = codec.compress_pointcloud(test_points)
    
    if compressed_data:
        print(f"\nОригинальный размер (байты): {codec.compression_stats['original_size']:.0f}")
        print(f"Сжатый размер (байты): {codec.compression_stats['compressed_size']:.0f}")
        print(f"Коэффициент сжатия: {codec.compression_stats['compression_ratio']:.2f}:1")
        
        # 4. Разжатие
        decompressed_points = codec.decompress_pointcloud(compressed_data)
        
        # 5. Проверка результата
        if decompressed_points is not None:
            # Из-за квантования (lossy compression) точки не будут абсолютно идентичными
            # Используем проверку формы и размера
            is_shape_equal = test_points.shape == decompressed_points.shape
            print(f"\nФорма оригинального облака: {test_points.shape}")
            print(f"Форма разжатого облака: {decompressed_points.shape}")
            print(f"Формы совпадают: {is_shape_equal}")
            
            # Проверка, что точки не полностью совпадают (т.е. произошло сжатие с потерями)
            # Если точки Float, они будут немного отличаться из-за квантования
            max_diff = np.max(np.abs(test_points - decompressed_points))
            print(f"Максимальное различие координат (Float): {max_diff:.6f}")