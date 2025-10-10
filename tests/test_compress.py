#Imports

#File Imports

# CHECK compress norm_pointcloud
# CHECK decompress norm_pointcloud

import pytest
import numpy as np
from unittest.mock import Mock, patch
import sys
sys.path.insert(0, '/home/ubuntu/upload')

# Мокируем DracoPy, так как он может быть не установлен
@pytest.fixture
def mock_draco():
    with patch('compress.draco') as mock:
        # Настраиваем мок для успешного сжатия
        mock.encode.return_value = b'compressed_data_mock'
        
        # Настраиваем мок для успешного разжатия
        mock_decoded = Mock()
        mock_decoded.points = np.random.rand(1000, 3).astype(np.float32)
        mock.decode.return_value = mock_decoded
        
        yield mock

class TestDracoCodec:
    """Тесты для класса DracoCodec."""
    
    def test_init_default_params(self, mock_draco):
        """Тест инициализации с параметрами по умолчанию."""
        from compress import DracoCodec
        
        codec = DracoCodec()
        assert codec.quantization_bits == 14
        assert codec.compression_level == 1
        assert codec.compression_stats == {}
    
    def test_init_custom_params(self, mock_draco):
        """Тест инициализации с пользовательскими параметрами."""
        from compress import DracoCodec
        
        codec = DracoCodec(quantization_bits=11, compression_level=7)
        assert codec.quantization_bits == 11
        assert codec.compression_level == 7
    
    def test_compress_pointcloud_valid_input(self, mock_draco, compression_test_data):
        """Тест успешного сжатия валидного облака точек."""
        from compress import DracoCodec
        
        codec = DracoCodec()
        result = codec.compress_pointcloud(compression_test_data)
        
        assert result == b'compressed_data_mock'
        mock_draco.encode.assert_called_once_with(
            compression_test_data,
            quantization_bits=14,
            compression_level=1
        )
        
        # Проверяем статистики сжатия
        assert 'original_size' in codec.compression_stats
        assert 'compressed_size' in codec.compression_stats
        assert 'compression_ratio' in codec.compression_stats
        assert 'original_points' in codec.compression_stats
    
    def test_compress_pointcloud_invalid_shape(self, mock_draco):
        """Тест сжатия с некорректной формой данных."""
        from compress import DracoCodec
        
        codec = DracoCodec()
        
        # Тест с 1D массивом
        invalid_data_1d = np.array([1, 2, 3])
        result = codec.compress_pointcloud(invalid_data_1d)
        assert result is None
        
        # Тест с неправильным количеством координат
        invalid_data_2d = np.random.rand(100, 2)
        result = codec.compress_pointcloud(invalid_data_2d)
        assert result is None
    
    def test_compress_pointcloud_empty_input(self, mock_draco):
        """Тест сжатия пустого облака точек."""
        from compress import DracoCodec
        
        codec = DracoCodec()
        empty_data = np.empty((0, 3))
        result = codec.compress_pointcloud(empty_data)
        
        # Должно обработать пустые данные
        assert result == b'compressed_data_mock'
    
    def test_compress_pointcloud_draco_error(self, mock_draco):
        """Тест обработки ошибки DracoPy при сжатии."""
        from compress import DracoCodec
        
        mock_draco.encode.side_effect = Exception("DracoPy encode error")
        
        codec = DracoCodec()
        valid_data = np.random.rand(100, 3).astype(np.float32)
        result = codec.compress_pointcloud(valid_data)
        
        assert result is None
    
    def test_decompress_pointcloud_valid_input(self, mock_draco):
        """Тест успешного разжатия валидных данных."""
        from compress import DracoCodec
        
        codec = DracoCodec()
        compressed_data = b'compressed_data_mock'
        result = codec.decompress_pointcloud(compressed_data)
        
        assert result is not None
        assert result.shape[1] == 3  # Проверяем, что получили 3D точки
        mock_draco.decode.assert_called_once_with(compressed_data)
    
    def test_decompress_pointcloud_invalid_data(self, mock_draco):
        """Тест разжатия невалидных данных."""
        from compress import DracoCodec
        
        mock_draco.decode.side_effect = Exception("DracoPy decode error")
        
        codec = DracoCodec()
        invalid_data = b'invalid_compressed_data'
        result = codec.decompress_pointcloud(invalid_data)
        
        assert result is None
    
    def test_decompress_pointcloud_no_points_attribute(self, mock_draco):
        """Тест разжатия когда декодированный объект не имеет атрибута points."""
        from compress import DracoCodec
        
        mock_decoded = Mock()
        del mock_decoded.points  # Удаляем атрибут points
        mock_draco.decode.return_value = mock_decoded
        
        codec = DracoCodec()
        result = codec.decompress_pointcloud(b'test_data')
        
        assert result is None
    
    def test_compression_decompression_cycle(self, mock_draco, compression_test_data):
        """Тест полного цикла сжатие -> разжатие."""
        from compress import DracoCodec
        
        # Настраиваем мок так, чтобы разжатие возвращало исходные данные
        mock_decoded = Mock()
        mock_decoded.points = compression_test_data
        mock_draco.decode.return_value = mock_decoded
        
        codec = DracoCodec()
        
        # Сжимаем
        compressed = codec.compress_pointcloud(compression_test_data)
        assert compressed is not None
        
        # Разжимаем
        decompressed = codec.decompress_pointcloud(compressed)
        assert decompressed is not None
        
        # Проверяем форму (точные значения могут отличаться из-за квантования)
        assert decompressed.shape == compression_test_data.shape
    
    def test_compression_stats_calculation(self, mock_draco, compression_test_data):
        """Тест правильности расчета статистик сжатия."""
        from compress import DracoCodec
        
        codec = DracoCodec()
        compressed = codec.compress_pointcloud(compression_test_data)
        
        stats = codec.compression_stats
        expected_original_size = compression_test_data.nbytes
        expected_compressed_size = len(b'compressed_data_mock')
        expected_ratio = expected_original_size / expected_compressed_size
        
        assert stats['original_size'] == expected_original_size
        assert stats['compressed_size'] == expected_compressed_size
        assert stats['compression_ratio'] == expected_ratio
        assert stats['original_points'] == len(compression_test_data)
    
    def test_compression_with_different_parameters(self, mock_draco, compression_test_data):
        """Тест сжатия с различными параметрами."""
        from compress import DracoCodec
        
        # Тест с высоким уровнем сжатия
        codec_high = DracoCodec(quantization_bits=8, compression_level=10)
        result_high = codec_high.compress_pointcloud(compression_test_data)
        
        mock_draco.encode.assert_called_with(
            compression_test_data,
            quantization_bits=8,
            compression_level=10
        )
        
        # Тест с низким уровнем сжатия
        codec_low = DracoCodec(quantization_bits=16, compression_level=0)
        result_low = codec_low.compress_pointcloud(compression_test_data)
        
        mock_draco.encode.assert_called_with(
            compression_test_data,
            quantization_bits=16,
            compression_level=0
        )
    
    @pytest.mark.parametrize("quantization_bits,compression_level", [
        (8, 0),
        (11, 5),
        (14, 7),
        (16, 10)
    ])
    def test_various_compression_settings(self, mock_draco, compression_test_data, 
                                        quantization_bits, compression_level):
        """Параметризованный тест различных настроек сжатия."""
        from compress import DracoCodec
        
        codec = DracoCodec(quantization_bits=quantization_bits, 
                          compression_level=compression_level)
        result = codec.compress_pointcloud(compression_test_data)
        
        assert result == b'compressed_data_mock'
        mock_draco.encode.assert_called_with(
            compression_test_data,
            quantization_bits=quantization_bits,
            compression_level=compression_level
        )

class TestCompressionIntegration:
    """Интеграционные тесты для модуля сжатия."""
    
    @pytest.mark.integration
    def test_main_function_execution(self, mock_draco):
        """Тест выполнения основной функции модуля."""
        from compress import DracoCodec
        
        # Имитируем выполнение основного блока
        num_points = 1000
        test_points = np.random.rand(num_points, 3).astype(np.float32)
        
        codec = DracoCodec(quantization_bits=11, compression_level=7)
        compressed_data = codec.compress_pointcloud(test_points)
        
        assert compressed_data is not None
        assert codec.compression_stats['original_points'] == num_points
        
        # Настраиваем мок для разжатия
        mock_decoded = Mock()
        mock_decoded.points = test_points
        mock_draco.decode.return_value = mock_decoded
        
        decompressed_points = codec.decompress_pointcloud(compressed_data)
        assert decompressed_points is not None
        assert decompressed_points.shape == test_points.shape
    
    @pytest.mark.slow
    def test_large_pointcloud_compression(self, mock_draco):
        """Тест сжатия большого облака точек."""
        from compress import DracoCodec
        
        # Создаем большое облако точек
        large_pointcloud = np.random.rand(100000, 3).astype(np.float32)
        
        codec = DracoCodec()
        result = codec.compress_pointcloud(large_pointcloud)
        
        assert result is not None
        assert codec.compression_stats['original_points'] == 100000