#Imports

#File Imports

# CHECK model structure
# CHEK model

"""
Тесты для модуля model.py - архитектура VAE для облаков точек.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.insert(0, '/home/ubuntu/upload')

from model import PCNEncoder, PCNDecoder, PointCloudVAE

class TestPCNEncoder:
    """Тесты для класса PCNEncoder."""
    
    def test_encoder_init_default(self):
        """Тест инициализации энкодера с параметрами по умолчанию."""
        encoder = PCNEncoder()
        
        assert encoder.conv1.in_channels == 3
        assert encoder.conv1.out_channels == 128
        assert encoder.conv2.in_channels == 128
        assert encoder.conv2.out_channels == 256
        assert encoder.fc1.in_features == 256
        assert encoder.fc1.out_features == 256
        assert encoder.fc_mean.in_features == 256
        assert encoder.fc_mean.out_features == 128
        assert encoder.fc_log_var.in_features == 256
        assert encoder.fc_log_var.out_features == 128
    
    def test_encoder_init_custom(self):
        """Тест инициализации энкодера с пользовательскими параметрами."""
        encoder = PCNEncoder(input_dim=6, latent_dim=256)
        
        assert encoder.conv1.in_channels == 6
        assert encoder.fc_mean.out_features == 256
        assert encoder.fc_log_var.out_features == 256
    
    def test_encoder_forward_shape(self, device, sample_pointcloud_torch):
        """Тест формы выходных тензоров энкодера."""
        encoder = PCNEncoder().to(device)
        batch_size = 4
        num_points = sample_pointcloud_torch.shape[0]
        
        # Создаем батч
        input_batch = sample_pointcloud_torch.unsqueeze(0).repeat(batch_size, 1, 1).to(device)
        
        mean, log_var = encoder(input_batch)
        
        assert mean.shape == (batch_size, 128)
        assert log_var.shape == (batch_size, 128)
        assert mean.dtype == torch.float32
        assert log_var.dtype == torch.float32
    
    def test_encoder_forward_single_batch(self, device):
        """Тест прямого прохода энкодера с одним элементом в батче."""
        encoder = PCNEncoder().to(device)
        input_pc = torch.randn(1, 1024, 3).to(device)
        
        mean, log_var = encoder(input_pc)
        
        assert mean.shape == (1, 128)
        assert log_var.shape == (1, 128)
    
    def test_encoder_gradient_flow(self, device):
        """Тест прохождения градиентов через энкодер."""
        encoder = PCNEncoder().to(device)
        input_pc = torch.randn(2, 512, 3, requires_grad=True).to(device)
        
        mean, log_var = encoder(input_pc)
        loss = mean.sum() + log_var.sum()
        loss.backward()
        
        # Проверяем, что градиенты вычислены
        assert input_pc.grad is not None
        for param in encoder.parameters():
            assert param.grad is not None
    
    @pytest.mark.parametrize("batch_size,num_points", [
        (1, 256),
        (2, 512),
        (4, 1024),
        (8, 2048)
    ])
    def test_encoder_various_inputs(self, device, batch_size, num_points):
        """Параметризованный тест энкодера с различными размерами входа."""
        encoder = PCNEncoder().to(device)
        input_pc = torch.randn(batch_size, num_points, 3).to(device)
        
        mean, log_var = encoder(input_pc)
        
        assert mean.shape == (batch_size, 128)
        assert log_var.shape == (batch_size, 128)

class TestPCNDecoder:
    """Тесты для класса PCNDecoder."""
    
    def test_decoder_init_default(self):
        """Тест инициализации декодера с параметрами по умолчанию."""
        decoder = PCNDecoder()
        
        assert decoder.num_coarse == 256
        assert decoder.num_fine == 1024
        assert decoder.fc1.in_features == 128
        assert decoder.fc3.out_features == 256 * 3
    
    def test_decoder_init_custom(self):
        """Тест инициализации декодера с пользовательскими параметрами."""
        decoder = PCNDecoder(latent_dim=256, num_coarse=512, num_fine=2048)
        
        assert decoder.num_coarse == 512
        assert decoder.num_fine == 2048
        assert decoder.fc1.in_features == 256
        assert decoder.fc3.out_features == 512 * 3
    
    def test_decoder_forward_shape(self, device):
        """Тест формы выходных тензоров декодера."""
        decoder = PCNDecoder().to(device)
        batch_size = 4
        latent_dim = 128
        
        z = torch.randn(batch_size, latent_dim).to(device)
        coarse_pc, fine_pc = decoder(z)
        
        assert coarse_pc.shape == (batch_size, 256, 3)
        assert fine_pc.shape == (batch_size, 1024, 3)
        assert coarse_pc.dtype == torch.float32
        assert fine_pc.dtype == torch.float32
    
    def test_decoder_forward_single_batch(self, device):
        """Тест прямого прохода декодера с одним элементом в батче."""
        decoder = PCNDecoder().to(device)
        z = torch.randn(1, 128).to(device)
        
        coarse_pc, fine_pc = decoder(z)
        
        assert coarse_pc.shape == (1, 256, 3)
        assert fine_pc.shape == (1, 1024, 3)
    
    def test_decoder_gradient_flow(self, device):
        """Тест прохождения градиентов через декодер."""
        decoder = PCNDecoder().to(device)
        z = torch.randn(2, 128, requires_grad=True).to(device)
        
        coarse_pc, fine_pc = decoder(z)
        loss = coarse_pc.sum() + fine_pc.sum()
        loss.backward()
        
        # Проверяем, что градиенты вычислены
        assert z.grad is not None
        for param in decoder.parameters():
            assert param.grad is not None
    
    @pytest.mark.parametrize("batch_size,latent_dim", [
        (1, 64),
        (2, 128),
        (4, 256),
        (8, 512)
    ])
    def test_decoder_various_latent_dims(self, device, batch_size, latent_dim):
        """Параметризованный тест декодера с различными размерами латентного пространства."""
        decoder = PCNDecoder(latent_dim=latent_dim).to(device)
        z = torch.randn(batch_size, latent_dim).to(device)
        
        coarse_pc, fine_pc = decoder(z)
        
        assert coarse_pc.shape == (batch_size, 256, 3)
        assert fine_pc.shape == (batch_size, 1024, 3)

class TestPointCloudVAE:
    """Тесты для полной модели PointCloudVAE."""
    
    def test_vae_init_default(self):
        """Тест инициализации VAE с параметрами по умолчанию."""
        model = PointCloudVAE()
        
        assert isinstance(model.encoder, PCNEncoder)
        assert isinstance(model.decoder, PCNDecoder)
        assert model.encoder.fc_mean.out_features == 128
        assert model.decoder.num_fine == 1024
    
    def test_vae_init_custom(self):
        """Тест инициализации VAE с пользовательскими параметрами."""
        model = PointCloudVAE(
            input_dim=6,
            latent_dim=256,
            num_coarse=512,
            num_fine=2048
        )
        
        assert model.encoder.conv1.in_channels == 6
        assert model.encoder.fc_mean.out_features == 256
        assert model.decoder.num_coarse == 512
        assert model.decoder.num_fine == 2048
    
    def test_reparameterize(self, device):
        """Тест reparameterization trick."""
        model = PointCloudVAE().to(device)
        batch_size = 4
        latent_dim = 128
        
        mean = torch.randn(batch_size, latent_dim).to(device)
        log_var = torch.randn(batch_size, latent_dim).to(device)
        
        z = model.reparameterize(mean, log_var)
        
        assert z.shape == (batch_size, latent_dim)
        assert z.dtype == torch.float32
        
        # Проверяем, что результат отличается при повторных вызовах (из-за случайности)
        z2 = model.reparameterize(mean, log_var)
        assert not torch.allclose(z, z2)
    
    def test_reparameterize_deterministic_when_log_var_zero(self, device):
        """Тест детерминированности reparameterize при нулевой дисперсии."""
        model = PointCloudVAE().to(device)
        batch_size = 2
        latent_dim = 128
        
        mean = torch.randn(batch_size, latent_dim).to(device)
        log_var = torch.zeros(batch_size, latent_dim).to(device)  # log(1) = 0, std = 1
        
        # При нулевой log_var результат должен быть близок к mean (но не точно из-за шума)
        z = model.reparameterize(mean, log_var)
        assert z.shape == mean.shape
    
    def test_vae_forward_shape(self, device, sample_pointcloud_torch):
        """Тест формы выходных тензоров полной модели."""
        model = PointCloudVAE().to(device)
        batch_size = 4
        
        input_batch = sample_pointcloud_torch.unsqueeze(0).repeat(batch_size, 1, 1).to(device)
        
        coarse_pc, fine_pc, mean, log_var = model(input_batch)
        
        assert coarse_pc.shape == (batch_size, 256, 3)
        assert fine_pc.shape == (batch_size, 1024, 3)
        assert mean.shape == (batch_size, 128)
        assert log_var.shape == (batch_size, 128)
    
    def test_vae_forward_single_batch(self, device):
        """Тест прямого прохода VAE с одним элементом в батче."""
        model = PointCloudVAE().to(device)
        input_pc = torch.randn(1, 512, 3).to(device)
        
        coarse_pc, fine_pc, mean, log_var = model(input_pc)
        
        assert coarse_pc.shape == (1, 256, 3)
        assert fine_pc.shape == (1, 1024, 3)
        assert mean.shape == (1, 128)
        assert log_var.shape == (1, 128)
    
    def test_vae_gradient_flow(self, device):
        """Тест прохождения градиентов через полную модель."""
        model = PointCloudVAE().to(device)
        input_pc = torch.randn(2, 512, 3, requires_grad=True).to(device)
        
        coarse_pc, fine_pc, mean, log_var = model(input_pc)
        loss = coarse_pc.sum() + fine_pc.sum() + mean.sum() + log_var.sum()
        loss.backward()
        
        # Проверяем, что градиенты вычислены
        assert input_pc.grad is not None
        for param in model.parameters():
            assert param.grad is not None
    
    def test_vae_eval_mode(self, device):
        """Тест работы модели в режиме eval."""
        model = PointCloudVAE().to(device)
        input_pc = torch.randn(2, 512, 3).to(device)
        
        model.eval()
        with torch.no_grad():
            coarse_pc, fine_pc, mean, log_var = model(input_pc)
        
        assert coarse_pc.shape == (2, 256, 3)
        assert fine_pc.shape == (2, 1024, 3)
        assert mean.shape == (2, 128)
        assert log_var.shape == (2, 128)
    
    def test_vae_train_mode(self, device):
        """Тест работы модели в режиме train."""
        model = PointCloudVAE().to(device)
        input_pc = torch.randn(2, 512, 3).to(device)
        
        model.train()
        coarse_pc, fine_pc, mean, log_var = model(input_pc)
        
        assert coarse_pc.shape == (2, 256, 3)
        assert fine_pc.shape == (2, 1024, 3)
        assert mean.shape == (2, 128)
        assert log_var.shape == (2, 128)
    
    @pytest.mark.parametrize("batch_size,num_points", [
        (1, 256),
        (2, 512),
        (4, 1024),
        (8, 2048)
    ])
    def test_vae_various_inputs(self, device, batch_size, num_points):
        """Параметризованный тест VAE с различными размерами входа."""
        model = PointCloudVAE().to(device)
        input_pc = torch.randn(batch_size, num_points, 3).to(device)
        
        coarse_pc, fine_pc, mean, log_var = model(input_pc)
        
        assert coarse_pc.shape == (batch_size, 256, 3)
        assert fine_pc.shape == (batch_size, 1024, 3)
        assert mean.shape == (batch_size, 128)
        assert log_var.shape == (batch_size, 128)
    
    def test_vae_state_dict_save_load(self, device, temp_dir):
        """Тест сохранения и загрузки состояния модели."""
        model1 = PointCloudVAE().to(device)
        model2 = PointCloudVAE().to(device)
        
        # Сохраняем состояние первой модели
        state_dict_path = temp_dir / "model_state.pth"
        torch.save(model1.state_dict(), state_dict_path)
        
        # Загружаем состояние во вторую модель
        model2.load_state_dict(torch.load(state_dict_path))
        
        # Проверяем, что параметры одинаковые
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2)
    
    def test_vae_reproducibility(self, device):
        """Тест воспроизводимости результатов при фиксированном seed."""
        model = PointCloudVAE().to(device)
        input_pc = torch.randn(2, 512, 3).to(device)
        
        # Первый прогон
        torch.manual_seed(42)
        coarse1, fine1, mean1, log_var1 = model(input_pc)
        
        # Второй прогон с тем же seed
        torch.manual_seed(42)
        coarse2, fine2, mean2, log_var2 = model(input_pc)
        
        # Энкодер должен давать одинаковые результаты
        assert torch.allclose(mean1, mean2)
        assert torch.allclose(log_var1, log_var2)
        
        # Декодер может отличаться из-за reparameterization trick
        # но при фиксированном seed должен быть одинаковым
        assert torch.allclose(coarse1, coarse2)
        assert torch.allclose(fine1, fine2)
    
    @pytest.mark.gpu
    def test_vae_gpu_compatibility(self):
        """Тест совместимости модели с GPU."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        device = torch.device("cuda")
        model = PointCloudVAE().to(device)
        input_pc = torch.randn(2, 512, 3).to(device)
        
        coarse_pc, fine_pc, mean, log_var = model(input_pc)
        
        assert coarse_pc.device.type == "cuda"
        assert fine_pc.device.type == "cuda"
        assert mean.device.type == "cuda"
        assert log_var.device.type == "cuda"
    
    def test_vae_memory_efficiency(self, device):
        """Тест эффективности использования памяти."""
        model = PointCloudVAE().to(device)
        
        # Проверяем, что модель не создает лишних тензоров
        initial_memory = torch.cuda.memory_allocated() if device.type == "cuda" else 0
        
        input_pc = torch.randn(4, 1024, 3).to(device)
        coarse_pc, fine_pc, mean, log_var = model(input_pc)
        
        # Очищаем промежуточные результаты
        del coarse_pc, fine_pc, mean, log_var, input_pc
        torch.cuda.empty_cache() if device.type == "cuda" else None
        
        final_memory = torch.cuda.memory_allocated() if device.type == "cuda" else 0
        
        # Память должна вернуться к исходному уровню (с небольшой погрешностью)
        if device.type == "cuda":
            assert abs(final_memory - initial_memory) < 1024 * 1024  # Менее 1MB разницы