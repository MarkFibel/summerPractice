import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def train_model(model, dataset, batch_size, lr, optimizer_name, epochs=100, val_split=0.2, verbose=False):
    """
    Обучает модель на заданном датасете с разделением на тренировочную и валидационную части.

    Параметры:
    -----------
    model : torch.nn.Module
        Нейронная сеть для обучения.
    dataset : torch.utils.data.Dataset
        Датасет, содержащий признаки и метки.
    batch_size : int
        Размер батча для загрузчика данных.
    lr : float
        Скорость обучения.
    optimizer_name : str
        Название оптимизатора. Поддерживаются: 'SGD', 'Adam', 'RMSprop'.
    epochs : int, optional (default=100)
        Количество эпох обучения.
    val_split : float, optional (default=0.2)
        Доля данных для валидации (от 0 до 1).
    verbose : bool, optional (default=False)
        Если True, выводит прогресс обучения по эпохам.

    Возвращает:
    -----------
    dict
        Словарь с ключами:
        - 'val_accuracy' : float, точность на валидации по последней эпохе
        - 'val_loss' : float, средняя ошибка на валидации по последней эпохе
        - 'history' : dict, история обучения с ключами 'train_loss', 'val_loss', 'val_accuracy'
        - 'model' : обученная модель
    """

    # Разделяем индексы датасета на тренировочные и валидационные
    indices = list(range(len(dataset)))
    train_idx, val_idx = train_test_split(indices, test_size=val_split, random_state=42, shuffle=True)

    # Создаем загрузчики данных для train и val
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False)

    # Функция потерь для бинарной классификации с логитами
    criterion = nn.BCEWithLogitsLoss()

    # Выбор оптимизатора
    if optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr)
    elif optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    # История обучения
    history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}

    for epoch in range(epochs):
        model.train()  # режим обучения
        total_train_loss = 0

        # Тренировочный цикл по батчам
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_logits = model(X_batch)
            loss = criterion(y_logits, y_batch)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * X_batch.size(0)

        # Средний тренировочный лосс за эпоху
        avg_train_loss = total_train_loss / len(train_loader.dataset)

        # Валидация
        model.eval()  # режим оценки
        total_val_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for X_val, y_val in val_loader:
                y_logits = model(X_val)
                val_loss = criterion(y_logits, y_val)
                total_val_loss += val_loss.item() * X_val.size(0)

                y_prob = torch.sigmoid(y_logits)
                y_pred = (y_prob > 0.5).float()
                all_preds.extend(y_pred.cpu().numpy())
                all_labels.extend(y_val.cpu().numpy())

        avg_val_loss = total_val_loss / len(val_loader.dataset)
        val_acc = accuracy_score(all_labels, all_preds)

        # Записываем историю
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_accuracy'].append(val_acc)

        if verbose:
            print(f"[{epoch+1}/{epochs}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")

    return {
        'val_accuracy': history['val_accuracy'][-1],
        'val_loss': history['val_loss'][-1],
        'history': history,
    }
