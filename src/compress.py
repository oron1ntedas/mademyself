# Imports
import numpy as np
import pydraco as draco
import logging
from typing import Optional, Dict

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DracoCodec:
    def __init__(self, quantization_bits: int = 14,
                 compression_level: int = 1):

        self.quantization_bits = quantization_bits
        self.compression_level = compression_level
        self.compression_stats: Dict[str, float] = {}
        
    def compress_pointcloud(self, points: np.ndarray) -> Optional[bytes]:

        if points.ndim != 2 or points.shape[1] != 3:
            logger.error("Некорректный формат входных данных. Ожидается (N, 3).")
            return None
        
        try:
            draco_pc = draco.PointCloud()
            draco_pc.add_point_attribute(draco.POSITION, points)
            
            options = draco.EncoderOptions()
            options.set_float("quantization_bits_position", self.quantization_bits)
            options.set_int("compression_level", self.compression_level)
            
            encoder = draco.Encoder()
            compressed_data = encoder.encode_point_cloud(draco_pc, options)
            
            if compressed_data:
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

        try:

            decoder = draco.Decoder()
            draco_pc = decoder.decode_point_cloud_from_buffer(compressed_data)
            
            if draco_pc:
                points = draco_pc.point_attribute(draco.POSITION).to_numpy()
                return points
            
        except Exception as e:
            logger.error(f"Ошибка разжатия: {e}")
            return None