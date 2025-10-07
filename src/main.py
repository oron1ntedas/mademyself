# Imports
import torch
import argparse
import os
from pathlib import Path
import numpy as np
import trimesh
from torch.utils.data import DataLoader

# File Imports
from postprocess.repair import MeshReconstructor
from model import PointCloudVAE
from train import train, plot_training_history # Импортируем функции из train.py
from metrics import QualityEvaluator, VisualizationTools
from preprocess.data import MeshData
from preprocess.normalize import norm_pointcloud


def main(args):
    """Основная функция, которая запускает нужный режим работы."""
    
    # --- 1. Выбор устройства ---
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Используем устройство: {device}")

    # --- 2. Режим ТРЕНИРОВКИ ---
    if args.action == 'train':
        print("--- РЕЖИМ: Начало новой тренировки ---")
        
        # Настройка данных
        dataset = MeshData(root_dir=args.data_dir, num_points=args.num_points, simulate_damage=True)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        
        # Инициализация модели и оптимизатора
        model = PointCloudVAE(num_fine=args.num_points).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        
        # Запуск обучения
        history = train(
            model=model, 
            dataloader=dataloader, 
            optimizer=optimizer, 
            device=device, 
            epochs=args.epochs, 
            checkpoint_dir=args.checkpoint_dir,
            start_epoch=0
        )
        
        # Построение графика после обучения
        if history:
            plot_training_history(history)

    # --- 3. Режим ПРОДОЛЖЕНИЯ ТРЕНИРОВКИ ---
    elif args.action == 'resume':
        print(f"--- РЕЖИМ: Продолжение тренировки с {args.checkpoint} ---")
        if not os.path.exists(args.checkpoint):
            print(f"Ошибка: Файл чекпоинта не найден по пути {args.checkpoint}")
            return
            
        # Настройка данных
        dataset = MeshData(root_dir=args.data_dir, num_points=args.num_points, simulate_damage=True)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        
        # Инициализация модели и оптимизатора
        model = PointCloudVAE(num_fine=args.num_points).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        
        # Загрузка состояния из чекпоинта
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        
        print(f"Чекпоинт успешно загружен. Обучение продолжится с эпохи {start_epoch + 1}.")
        
        # Запуск обучения с нужной эпохи
        history = train(
            model=model, 
            dataloader=dataloader, 
            optimizer=optimizer, 
            device=device, 
            epochs=args.epochs, 
            checkpoint_dir=args.checkpoint_dir,
            start_epoch=start_epoch
        )
        
        if history:
            plot_training_history(checkpoint.get('history', history)) # Используем старую историю, если есть


    # --- 4. Режим ТЕСТИРОВАНИЯ на одном файле ---
    elif args.action == 'test_one':
        print(f"--- РЕЖИМ: Тестирование на файле {args.file_path} ---")
        if not os.path.exists(args.file_path):
            print(f"Ошибка: Файл не найден по пути {args.file_path}")
            return
        if not os.path.exists(args.model_weights):
            print(f"Ошибка: Файл с весами модели не найден: {args.model_weights}")
            return
            
        # Загрузка модели
        model = PointCloudVAE(num_fine=args.num_points).to(device)
        model.load_state_dict(torch.load(args.model_weights))
        model.eval() # Переключаем модель в режим оценки
        
        # Загрузка и обработка .obj файла
        mesh = trimesh.load(args.file_path, process=False)
        points, _ = trimesh.sample.sample_surface(mesh, args.num_points)
        original_pc = torch.tensor(points, dtype=torch.float32)
        
        # Нормализация и создание "поврежденной" версии для теста
        normalized_pc = norm_pointcloud(original_pc.numpy())
        damaged_pc, _, _ = MeshData._simulate_holes(None, normalized_pc, args.num_points)
        
        # Восстановление
        with torch.no_grad(): # Отключаем расчет градиентов для ускорения
            _, reconstructed_pc, _, _ = model(damaged_pc.unsqueeze(0).to(device))
            reconstructed_pc = reconstructed_pc.squeeze(0).cpu().numpy()
        
        mesh_reconstructor = MeshReconstructor()
        
        # Используем метод Пуассона ('poisson') или 'ball_pivoting'
        # Вход: numpy array (reconstructed_pc)
        reconstructed_o3d_mesh, info = mesh_reconstructor.pointcloud_to_mesh(
            points=reconstructed_pc, 
            method='poisson'
        )
        
        # Преобразование o3d.geometry.TriangleMesh в trimesh для сохранения
        reconstructed_trimesh = trimesh.Trimesh(
            vertices=np.asarray(reconstructed_o3d_mesh.vertices),
            faces=np.asarray(reconstructed_o3d_mesh.triangles)
        )

        # Оценка качества
        evaluator = QualityEvaluator()
        report = evaluator.evaluate_reconstruction(
            original=normalized_pc.numpy(),
            damaged=damaged_pc.numpy(),
            reconstructed=reconstructed_pc
        )
        print("--- Отчет о качестве восстановления ---")
        print(f"Общий балл: {report['overall_score']:.3f}")
        print(f"Chamfer Distance: {report['point_cloud_metrics']['chamfer_distance_original']:.6f}")
        
        # Сохранение результатов для Blender
        output_dir = "test_results"
        os.makedirs(output_dir, exist_ok=True)
        
        # СОХРАНЯЕМ ВОССТАНОВЛЕННЫЙ МЕШ
        reconstructed_trimesh.export(os.path.join(output_dir, "reconstructed_mesh.obj"))
        
        # Можно сохранить и другие для сравнения
        trimesh.Trimesh(vertices=damaged_pc.numpy()).export(os.path.join(output_dir, "damaged.obj"))
        trimesh.Trimesh(vertices=normalized_pc.numpy()).export(os.path.join(output_dir, "original.obj"))

        print(f"Результаты (включая **восстановленный меш**) сохранены в папку '{output_dir}'.")
        
    else:
        print("Неизвестное действие. Используйте 'train', 'resume' или 'test_one'.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Главный скрипт для обучения и тестирования модели восстановления 3D облаков точек.")
    
    # --- Общие аргументы ---
    parser.add_argument('action', choices=['train', 'resume', 'test_one'], help="Действие: 'train' (начать обучение), 'resume' (продолжить), 'test_one' (тест на 1 файле).")
    parser.add_argument('--data_dir', type=str, default='sample_data', help="Путь к папке с данными для обучения.")
    parser.add_argument('--num_points', type=int, default=1024, help="Количество точек в облаке.")
    parser.add_argument('--cpu', action='store_true', help="Использовать CPU вместо GPU.")

    # --- Аргументы для обучения ---
    parser.add_argument('--epochs', type=int, default=50, help="Количество эпох для обучения.")
    parser.add_argument('--batch_size', type=int, default=4, help="Размер батча.")
    parser.add_argument('--lr', type=float, default=0.001, help="Скорость обучения (learning rate).")
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help="Папка для сохранения чекпоинтов.")
    
    # --- Аргументы для продолжения обучения и тестирования ---
    parser.add_argument('--checkpoint', type=str, default='checkpoints/checkpoint_epoch_50.pth', help="Путь к файлу чекпоинта для возобновления обучения.")
    parser.add_argument('--model_weights', type=str, default='pcn_vae_model.pth', help="Путь к сохраненным весам модели для тестирования.")

    # --- Аргументы для тестирования на одном файле ---
    parser.add_argument('--file_path', type=str, help="Путь к .obj файлу для тестирования.")

    args = parser.parse_args()
    main(args)