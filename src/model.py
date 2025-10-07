#Imports
import torch
import torch.nn as nn
import torch.nn.functional as F

#File Imports

# model structure
class PCNEncoder(nn.Module):
    def __init__(self, input_dim: int = 3, latent_dim: int = 128):
        super(PCNEncoder, self).__init__()
        # Сверточные слои для извлечения признаков из точек
        self.conv1 = nn.Conv1d(input_dim, 128, 1)
        self.conv2 = nn.Conv1d(128, 256, 1)
        
        # Полносвязные слои для обработки глобального признака
        self.fc1 = nn.Linear(256, 256)
        
        # Два "головы" в конце: одна предсказывает mean, другая - log_var
        self.fc_mean = nn.Linear(256, latent_dim)
        self.fc_log_var = nn.Linear(256, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Прямой проход энкодера. Args: x - Входное облако точек (batch_size, num_points, 3)
        
        # Свертки ожидают формат (batch_size, channels, num_points), поэтому транспонируем
        x = x.transpose(1, 2)
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        # Чтобы получить один глобальный вектор, описывающий всю форму,
        # мы берем максимальное значение по всем точкам (это называется Max Pooling)
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 256) # Убираем лишнюю размерность
        
        x = F.relu(self.fc1(x))
        
        # Получаем наши два выхода
        mean = self.fc_mean(x)
        log_var = self.fc_log_var(x)
        
        return mean, log_var


class PCNDecoder(nn.Module):
    def __init__(self, latent_dim: int = 128, num_coarse: int = 256, num_fine: int = 1024):
        super(PCNDecoder, self).__init__()
        self.num_coarse = num_coarse
        self.num_fine = num_fine
        
        # --- Ветка для создания "наброска" (Coarse) ---
        self.fc1 = nn.Linear(latent_dim, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, self.num_coarse * 3) # Выход: 256 точек * 3 координаты
        
        # --- Ветка для детализации (Fine) ---
        # Мы используем 1D-свертки для "сворачивания" (folding)
        # Вход: латентный вектор + координаты грубой точки + координаты 2D-сетки
        self.conv1 = nn.Conv1d(latent_dim + 3 + 3, 512, 1)
        self.conv2 = nn.Conv1d(512, 512, 1)
        self.conv3 = nn.Conv1d(512, 3, 1) # На выходе 3D-координаты для каждой точки

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Прямой проход декодера. Args: z- Латентный вектор (batch_size, latent_dim)
        batch_size = z.size(0)
        
        # 1. Генерируем "набросок"
        c = F.relu(self.fc1(z))
        c = F.relu(self.fc2(c))
        coarse_pc = self.fc3(c).view(batch_size, self.num_coarse, 3)
        
        # 2. Генерируем детальную модель
        # Создаем 2D-сетку (как лист бумаги, который будем сгибать)
        # Для 1024 точек нам нужна сетка 32x32
        grid_size = int(self.num_fine**0.5)
        grid_x, grid_y = torch.meshgrid(
            torch.linspace(-0.5, 0.5, grid_size),
            torch.linspace(-0.5, 0.5, grid_size),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=-1).view(-1, 2).unsqueeze(0).to(z.device)
        grid_3d = torch.cat([grid, torch.zeros_like(grid[..., :1])], dim=-1) # Добавляем Z=0
        grid_3d = grid_3d.repeat(batch_size, 1, 1) # Повторяем для каждого элемента в батче

        # "Размножаем" точки наброска, чтобы их количество совпало с детальными
        coarse_tiled = coarse_pc.unsqueeze(2).repeat(1, 1, self.num_fine // self.num_coarse, 1).view(batch_size, self.num_fine, 3)
        
        # "Размножаем" латентный вектор для каждой точки
        z_tiled = z.unsqueeze(1).repeat(1, self.num_fine, 1)

        # Собираем всю информацию вместе для каждой будущей точки
        features = torch.cat([coarse_tiled, grid_3d, z_tiled], dim=2)
        
        # Транспонируем для сверток: (batch_size, channels, num_points)
        features = features.transpose(1, 2)
        
        # Прогоняем через свертки, чтобы "свернуть" сетку в 3D-форму
        f = F.relu(self.conv1(features))
        f = F.relu(self.conv2(f))
        fine_pc = self.conv3(f).transpose(1, 2) # Возвращаем в формат (batch, num_points, 3)

        return coarse_pc, fine_pc
# model 

class PointCloudVAE(nn.Module):
    #Полная модель VAE, объединяющая Энкодер и Декодер.
    def __init__(self, input_dim: int = 3, latent_dim: int = 128, 
                 num_coarse: int = 256, num_fine: int = 1024):
        super(PointCloudVAE, self).__init__()
        self.encoder = PCNEncoder(input_dim, latent_dim)
        self.decoder = PCNDecoder(latent_dim, num_coarse, num_fine)

    def reparameterize(self, mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Тот самый "Reparameterization Trick". Позволяет взять случайную
        точку из "облака резюме" обучаемым способом.
        """
        std = torch.exp(0.5 * log_var) # стандартное отклонение
        eps = torch.randn_like(std)   # случайный шум из стандартного нормального распределения
        return mean + eps * std       # смещаем и масштабируем шум

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # 1. Прогоняем через энкодер, чтобы получить параметры "облака"
        mean, log_var = self.encoder(x)
        # 2. Берем случайную точку 'z' из этого облака
        z = self.reparameterize(mean, log_var)
        # 3. Прогоняем 'z' через декодер для восстановления
        coarse_pc, fine_pc = self.decoder(z)
        
        # Возвращаем всё, что нужно для "проверки качества":
        # два восстановленных облака и параметры распределения
        return coarse_pc, fine_pc, mean, log_var