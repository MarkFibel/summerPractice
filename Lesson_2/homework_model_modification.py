# 1.1
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from utils import make_regression_data, mse, log_epoch, RegressionDataset, make_classification_data, ClassificationDataset
from models.models import LinearRegression, LogisticRegression


def linear_regression():
    # Генерируем данные
    X, y = make_regression_data(n=200)
    
    # Создаём датасет и даталоадер
    dataset = RegressionDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    print(f'Размер датасета: {len(dataset)}')
    print(f'Количество батчей: {len(dataloader)}')
    
    # Создаём модель, функцию потерь и оптимизатор
    model = LinearRegression(in_features=1)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    
    # Параметры регуляризации
    l1_lambda = 0.001  # коэффициент L1
    l2_lambda = 0.001  # коэффициент L2
    
    # Параметры ранней остановки
    best_loss = float('inf')
    patience = 10  # сколько эпох ждать улучшения
    patience_counter = 0
    
    epochs = 100
    for epoch in range(1, epochs + 1):
        total_loss = 0
        
        for i, (batch_X, batch_y) in enumerate(dataloader):
            optimizer.zero_grad()
            y_pred = model(batch_X)
            loss = criterion(y_pred, batch_y)
            
            # Добавляем L1 и L2 регуляризацию к потере
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            l2_norm = sum(p.pow(2).sum() for p in model.parameters())
            loss = loss + l1_lambda * l1_norm + l2_lambda * l2_norm
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / (i + 1)
        log_epoch(epoch, avg_loss)
        
        # Проверка на раннюю остановку
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            # Можно сохранять лучшую модель
            torch.save(model.state_dict(), 'Lesson_2/models/best_linreg.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Ранняя остановка на эпохе {epoch}')
                break

    # После обучения загрузить лучшую модель
    best_model = LinearRegression(in_features=1)
    best_model.load_state_dict(torch.load('best_linreg.pth'))
    best_model.eval()

def logistic_regression():
    # Генерируем многоклассовые данные
    X, y = make_classification_data(n=200, n_classes=3)  # например, 3 класса
    
    dataset = ClassificationDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    num_classes = 3
    model = LogisticRegression(in_features=2, num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    
    epochs = 100
    all_preds = []
    all_targets = []

    for epoch in range(1, epochs + 1):
        total_loss = 0
        total_acc = 0
        all_preds_epoch = []
        all_targets_epoch = []

        for i, (batch_X, batch_y) in enumerate(dataloader):
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y.long())
            loss.backward()
            optimizer.step()

            # Предсказания
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds_epoch.extend(preds.cpu().numpy())
            all_targets_epoch.extend(batch_y.cpu().numpy())

            # Метрика accuracy
            correct = (preds == batch_y).sum().item()
            acc_batch = correct / batch_y.size(0)

            total_loss += loss.item()
            total_acc += acc_batch

        avg_loss = total_loss / (i + 1)
        avg_acc = total_acc / (i + 1)

        # В конце эпохи собираем метрики
        precision = precision_score(all_targets_epoch, all_preds_epoch, average='macro')
        recall = recall_score(all_targets_epoch, all_preds_epoch, average='macro')
        f1 = f1_score(all_targets_epoch, all_preds_epoch, average='macro')
        # ROC-AUC для многоклассовых задач
        # Нужно получить вероятности
        # Для этого можно сделать один проход по данным или хранить их
        # В простом случае пропустим ROC-AUC или сделаем отдельно

        if epoch % 10 == 0:
            log_epoch(epoch, avg_loss, acc=avg_acc)
            print(f"Epoch {epoch} - Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}")

        # Визуализация confusion matrix раз в несколько эпох
        if epoch % 50 == 0:
            cm = confusion_matrix(all_targets_epoch, all_preds_epoch)
            plt.figure(figsize=(6,4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion matrix at epoch {epoch}')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.savefig(f'Lesson_2/plots/logistic_regression_confusion_matrix_{epoch}.png')

if __name__ == '__main__':
    # linear_regression()
    logistic_regression()