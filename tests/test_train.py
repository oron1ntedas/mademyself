#Imports

#File Imports

# CHECK train model
# CHECK plot of training + metrics
# CHECK checkpoints

import pytest
import torch

from src.hybrid_model import HybridModel, create_optimizers

class TestHybridModel:
    """Критически важные тесты гибридной модели"""
    
    def test_hybrid_model_creation(self):
        """Тест создания гибридной модели"""
        model = HybridModel(num_points=64, latent_dim=32, timesteps=50)
        
        # Проверяем наличие всех компонентов
        assert hasattr(model, 'vae')
        assert hasattr(model, 'gan_generator')
        assert hasattr(model, 'gan_discriminator')
        assert hasattr(model, 'ddpm')
    
    def test_hybrid_forward_pass(self, sample_damaged_points):
        """Тест forward pass гибридной модели"""
        model = HybridModel(num_points=64, latent_dim=32, timesteps=50)
        model.eval()
        
        with torch.no_grad():
            coarse, refined, latent = model(sample_damaged_points)
        
        # Проверяем корректность выходов
        assert coarse.shape == sample_damaged_points.shape
        assert refined.shape == sample_damaged_points.shape
        assert latent.shape[0] == sample_damaged_points.shape[0]
    
    def test_optimizers_creation(self):
        """Тест создания оптимизаторов"""
        model = HybridModel(num_points=64, latent_dim=32, timesteps=50)
        optimizers = create_optimizers(model)
        
        # Проверяем что все оптимизаторы созданы
        expected_keys = ['vae', 'gan_g', 'gan_d', 'diffusion']
        for key in expected_keys:
            assert key in optimizers
            assert optimizers[key] is not None

