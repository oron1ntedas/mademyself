#Imports
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from pathlib import Path
import matplotlib.pyplot as plt  # Библиотека для графиков
import numpy as np

#File imports
from preprocess.data import MeshData
from model import PointCloudVAE


def chamfer_distance_torch(pc1, pc2):
    """Torch-версия Chamfer Distance для обучения."""
    dist_matrix = torch.cdist(pc1, pc2)
    min_dists_1_to_2, _ = torch.min(dist_matrix, dim=2)
    min_dists_2_to_1, _ = torch.min(dist_matrix, dim=1)
    return torch.mean(min_dists_1_to_2) + torch.mean(min_dists_2_to_1)

def vae_loss_function(original_pc, coarse_pc, fine_pc, mean, log_var):
    """Вычисляет комбинированную VAE-потерю."""
    loss_coarse = chamfer_distance_torch(coarse_pc, original_pc)
    loss_fine = chamfer_distance_torch(fine_pc, original_pc)
    reconstruction_loss = loss_coarse + loss_fine
    kl_divergence = -0.5 * torch.mean(torch.sum(1 + log_var - mean.pow(2) - log_var.exp(), dim=1))
    total_loss = reconstruction_loss + 0.001 * kl_divergence
    return total_loss, reconstruction_loss, kl_divergence

#plot of train + metrics
def plot_training_history(history):
    """Строит и сохраняет график истории обучения."""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['total_loss'], label='Общая ошибка (Total Loss)')
    plt.title('Динамика общей ошибки')
    plt.xlabel('Эпоха')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['recon_loss'], label='Ошибка восстановления (Recon Loss)', color='orange')
    plt.plot(history['kl_loss'], label='KL дивергенция (KL Loss)', color='green')
    plt.title('Компоненты ошибки')
    plt.xlabel('Эпоха')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("training_history.png")
    print("📈 График обучения сохранен в training_history.png")
    plt.close()

# train
def train(model, dataloader, optimizer, device, epochs, checkpoint_dir, start_epoch=0):
    print("🚀 Начинаем тренировку модели...")
    model.train()
    
    history = {
        'total_loss': [],
        'recon_loss': [],
        'kl_loss': []
    }
    
    for epoch in range(start_epoch, epochs):
        epoch_total_loss = 0
        epoch_recon_loss = 0
        epoch_kl_loss = 0
        
        for i, (original, damaged, _, _) in enumerate(dataloader):
            original = original.to(device)
            damaged = damaged.to(device)
            
            optimizer.zero_grad()
            coarse_rec, fine_rec, mean, log_var = model(damaged)
            loss, recon_loss, kl_loss = vae_loss_function(original, coarse_rec, fine_rec, mean, log_var)
            loss.backward()
            optimizer.step()
            
            epoch_total_loss += loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_kl_loss += kl_loss.item()
        
        # Записываем средние значения за эпоху
        avg_total_loss = epoch_total_loss / len(dataloader)
        avg_recon_loss = epoch_recon_loss / len(dataloader)
        avg_kl_loss = epoch_kl_loss / len(dataloader)
        
        history['total_loss'].append(avg_total_loss)
        history['recon_loss'].append(avg_recon_loss)
        history['kl_loss'].append(avg_kl_loss)
        
        print(f"--- 🏁 Эпоха {epoch+1}/{epochs} завершена ---")
        print(f"Avg Loss: {avg_total_loss:.4f} (Recon: {avg_recon_loss:.4f}, KL: {avg_kl_loss:.4f})")
        
# checkpoints
        if (epoch + 1) % 5 == 0 or (epoch + 1) == epochs: # Сохраняем каждые 5 эпох и в конце
            checkpoint_path = Path(checkpoint_dir) / f"checkpoint_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history
            }, checkpoint_path)
            print(f"💾 Чекпоинт сохранен: {checkpoint_path}")

    print("✅ Тренировка окончена.")
    return history


if __name__ == "__main__":
    # --- 1. Настройка ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_POINTS = 1024
    EPOCHS = 25
    LEARNING_RATE = 0.001
    BATCH_SIZE = 4
    CHECKPOINT_DIR = "checkpoints"
    
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    print(f"Используем устройство: {DEVICE}")

    # --- 2. Подготовка данных ---
    # (код для создания тестовых данных остается таким же)
    data_dir = "sample_data"
    os.makedirs(data_dir, exist_ok=True)
    sample_obj_path = Path(data_dir) / "cube.obj"
    if not sample_obj_path.exists():
        print("Создаем тестовый куб cube.obj...")
        with open(sample_obj_path, "w") as f:
            f.write("v -0.5 -0.5 -0.5\nv 0.5 -0.5 -0.5\nv -0.5 0.5 -0.5\nv 0.5 0.5 -0.5\n")
            f.write("v -0.5 -0.5 0.5\nv 0.5 -0.5 0.5\nv -0.5 0.5 0.5\nv 0.5 0.5 0.5\n")
            f.write("f 1 2 4 3\nf 5 6 8 7\nf 1 5 7 3\nf 2 6 8 4\nf 1 2 6 5\nf 3 4 8 7\n")
    
    dataset = MeshData(root_dir=data_dir, num_points=NUM_POINTS, simulate_damage=True)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # --- 3. Инициализация модели и оптимизатора ---
    model = PointCloudVAE(num_fine=NUM_POINTS).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- НОВЫЙ БЛОК: Загрузка чекпоинта для возобновления обучения ---
    start_epoch = 0
    latest_checkpoint = None # Здесь можно указать путь к конкретному чекпоинту
    
    if latest_checkpoint and os.path.exists(latest_checkpoint):
        print(f"🔄 Возобновление обучения с чекпоинта: {latest_checkpoint}")
        checkpoint = torch.load(latest_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        # Можно также загрузить историю, если нужно
        # history = checkpoint['history']
        print(f"Обучение продолжится с эпохи {start_epoch + 1}")
    else:
        print("Начинаем обучение с нуля.")

    training_history = train(model, dataloader, optimizer, DEVICE, EPOCHS, CHECKPOINT_DIR, start_epoch)

    if training_history:
        plot_training_history(training_history)