import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


class CustomCSVDataset(Dataset):
    """
    Кастомный класс PyTorch Dataset для работы с табличными CSV-данными.

    Возможности:
    - Загрузка данных из CSV-файла или pandas DataFrame
    - Обработка категориальных признаков с помощью LabelEncoder
    - Нормализация числовых признаков с помощью StandardScaler
    - Автоматическая обработка пропущенных значений (заполнение средним)
    
    Параметры:
    ----------
    data : str | pd.DataFrame
        Путь к CSV-файлу или DataFrame с данными.
    label_column : str
        Название столбца, содержащего целевую переменную.
    """
    
    def __init__(self, data: str | pd.DataFrame, label_column: str):
        super().__init__()
        
        # Загрузка данных
        if isinstance(data, str):
            self.data = pd.read_csv(data)
        elif isinstance(data, pd.DataFrame):
            self.data = data.copy()
        else:
            raise ValueError("Ожидался путь к файлу или объект DataFrame")

        self.label_column = label_column
        self.feature_columns = [col for col in self.data.columns if col != label_column]

        # Инициализация кодировщиков и нормализатора
        self.label_encoders = {}        # для категориальных признаков
        self.scaler = StandardScaler()  # для числовых признаков

        # Предобработка данных
        self._preprocess()

    def _preprocess(self):
        """
        Выполняет предобработку данных:
        - Кодирует категориальные признаки
        - Заполняет пропущенные числовые значения
        - Нормализует числовые признаки
        """
        features = self.data[self.feature_columns]
        label = self.data[self.label_column]

        # Выделяем категориальные и числовые столбцы
        categorical_cols = features.select_dtypes(include=["object", "category"]).columns
        numerical_cols = features.select_dtypes(include=["number"]).columns

        # Кодирование категориальных признаков
        for col in categorical_cols:
            le = LabelEncoder()
            features[col] = le.fit_transform(features[col].astype(str))
            self.label_encoders[col] = le

        # Заполнение пропусков в числовых данных средним значением
        features[numerical_cols] = features[numerical_cols].fillna(features[numerical_cols].mean())

        # Нормализация числовых признаков
        features[numerical_cols] = self.scaler.fit_transform(features[numerical_cols])

        # Конвертация в тензоры PyTorch
        self.X = torch.tensor(features.values, dtype=torch.float32)
        self.y = torch.tensor(label.values, dtype=torch.float32)
        if self.y.ndim == 1:
            self.y = self.y.unsqueeze(1)  # Приведение y к форме [N, 1] для совместимости

    def __len__(self):
        """
        Возвращает количество примеров в датасете.
        """
        return len(self.data)

    def __getitem__(self, index):
        """
        Возвращает пару (X, y) по индексу.
        
        Параметры:
        ----------
        index : int
            Индекс примера в датасете.
        
        Возвращает:
        -----------
        tuple(torch.Tensor, torch.Tensor)
            Фича-вектор и метка.
        """
        return self.X[index], self.y[index]


# Пример использования
if __name__ == '__main__':
    # Датасет с пассажирами Титаника
    dataset = CustomCSVDataset('Lesson_2/data/titanic.csv', label_column="survived")
    print(dataset[0])
    
    # Датасет с ценами на дома
    dataset = CustomCSVDataset('Lesson_2/data/house_cost.csv', label_column="House_Price")
    print(dataset[0])
