import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class ClassificationDataset(Dataset):
    """
    Простой Dataset для загрузки изображений и меток из DataFrame.
    DataFrame должен иметь колонки: 'path' и 'target'.
    """
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)  # Сбрасываем индексы
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_path = row['path']
        label = row['target']

        # Открываем изображение и переводим в RGB
        image = Image.open(image_path).convert("RGB")

        # Применяем трансформации (если заданы)
        if self.transform is not None:
            image = self.transform(image)

        return image, label


class Trainer:
    def __init__(self, model: nn.Module, annotation_path: str):
        """
        Args:
            model (nn.Module): Модель для классификации (например, resnet50(...) или vgg16(...)).
            annotation_path (str): Путь к .parquet файлу с колонками 'path', 'target' и 'set'.
        """
        self.model = model
        self.df = pd.read_parquet(annotation_path)

        # Проверим, что нужные колонки существуют
        required_columns = {'path', 'target', 'set'}
        if not required_columns.issubset(self.df.columns):
            raise ValueError("DataFrame must contain 'path', 'target' and 'set' columns.")

    def train(self,
              epochs: int,
              batch_size: int = 32,
              lr: float = 1e-3,
              device: str = 'cuda'):
        """
        Запуск обучения модели.

        Args:
            epochs (int): Количество эпох обучения.
            batch_size (int): Размер batch.
            lr (float): Начальная скорость обучения.
            device (str): 'cpu' или 'cuda'.
        """
        # Определяем устройство
        device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model.to(device)

        # Разделяем DataFrame на train и val по колонке 'set'
        train_df = self.df[self.df['set'] == 'train'].copy()
        val_df = self.df[self.df['set'] == 'val'].copy()

        # Создаём датасеты
        train_dataset = ClassificationDataset(
            df=train_df,
            transform=transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
        )

        val_dataset = ClassificationDataset(
            df=val_df,
            transform=transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
        )

        # Создаём DataLoader-ы
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Функция потерь и оптимизатор
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # Запуск цикла обучения
        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            correct_preds = 0
            total_samples = 0

            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)

                # Обнуляем градиенты
                optimizer.zero_grad()

                # Прямой проход
                logits = self.model(images)
                loss = criterion(logits, labels)

                # Обратный проход
                loss.backward()
                optimizer.step()

                # Статистика
                running_loss += loss.item() * images.size(0)
                _, preds = torch.max(logits, dim=1)
                correct_preds += (preds == labels).sum().item()
                total_samples += images.size(0)

            train_loss = running_loss / total_samples
            train_acc = correct_preds / total_samples

            # Валидация
            val_loss, val_acc = self._validate(val_loader, criterion, device)

            print(f"Epoch [{epoch+1}/{epochs}], "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    def _validate(self, val_loader, criterion, device):
        """Оценка модели на валидационном наборе."""
        self.model.eval()
        running_loss = 0.0
        correct_preds = 0
        total_samples = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                logits = self.model(images)
                loss = criterion(logits, labels)

                running_loss += loss.item() * images.size(0)
                _, preds = torch.max(logits, dim=1)
                correct_preds += (preds == labels).sum().item()
                total_samples += images.size(0)

        val_loss = running_loss / total_samples
        val_acc = correct_preds / total_samples
        return val_loss, val_acc

