import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch
from anomaly_detector.preprocessing import DataPreprocessor


class TestDataPreprocessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Создаем тестовые данные один раз для всех тестов"""
        cls.test_data = pd.DataFrame({
            "time": ["2023-01-01 00:00", "2023-01-01 01:00", "2023-01-01 02:00", "2023-01-01 03:00"],
            "value": [1.0, np.nan, 15.5, 100.0]  # Нормальное, пропущенное, граничное и аномальное значения
        })

    def test_data_loading(self):
        """Тест загрузки данных"""
        with patch('pandas.read_csv') as mock_read:
            mock_read.return_value = self.test_data
            processor = DataPreprocessor()
            result = processor.load_data()

            self.assertIsInstance(result, pd.DataFrame)
            self.assertEqual(len(result), 4)
            mock_read.assert_called_once()

    def test_cleaning_functionality(self):
        """Тест основных функций очистки"""
        processor = DataPreprocessor()

        # Тестируем очистку данных
        cleaned_df = processor.clean_data(self.test_data)

        # Проверяем удаление пропущенных значений и аномалий
        # Должны остаться только 1.0 и 15.5 (100.0 - аномалия, NaN - пропуск)
        self.assertEqual(len(cleaned_df), 2)

        # Проверяем что остались только корректные значения
        self.assertListEqual(sorted(cleaned_df["value"].tolist()), [1.0, 15.5])

    def test_save_clean_data(self):
        """Тест сохранения данных"""
        processor = DataPreprocessor()
        cleaned_df = self.test_data.dropna().query("value <= 15 and value >= -15").copy()

        with patch.object(cleaned_df, 'to_csv') as mock_to_csv:
            processor.save_clean_data(cleaned_df)
            mock_to_csv.assert_called_once()

            # Получаем аргументы вызова
            args, kwargs = mock_to_csv.call_args

            # Проверяем что index=False
            self.assertFalse(kwargs.get('index', True))

            # Проверяем что путь по умолчанию
            self.assertEqual(str(args[0]), 'cleaned_sensor_data.csv')

    def test_empty_data_handling(self):
        """Тест обработки пустых данных"""
        processor = DataPreprocessor()
        empty_df = pd.DataFrame(columns=["time", "value"])

        # Для clean_data не должно быть исключения - просто вернет пустой DataFrame
        cleaned_empty = processor.clean_data(empty_df)
        self.assertTrue(cleaned_empty.empty)

        # Проверяем сохранение пустых данных - должно вызывать ValueError
        with self.assertRaises(ValueError):
            processor.save_clean_data(pd.DataFrame())

    if __name__ == "__main__":
        unittest.main()