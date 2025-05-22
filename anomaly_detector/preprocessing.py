import pandas as pd
import yaml


class DataPreprocessor:
    def __init__(self, config_file='config.yaml'):
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f).get('preprocessing', {})

    def load_data(self, file_path=None):
        """Загрузка сырых данных"""
        path = file_path or self.config.get('input_file', 'sensor_data.csv')
        df = pd.read_csv(path)
        return df

    def clean_data(self, df):
        """Очистка данных"""
        # Удаление дубликатов
        df = df.drop_duplicates()

        # Удаление пропусков
        df.dropna(inplace=True)

        # Удаление явных аномалий
        bounds = self.config.get('bounds', {'lower': -15, 'upper': 15})
        mask = (df['value'] >= bounds['lower']) & (df['value'] <= bounds['upper'])
        cleaned_df = df[mask].copy()

        return cleaned_df

    def save_clean_data(self, df, file_path=None):
        """Сохранение очищенных данных"""
        path = file_path or self.config.get('output_file', 'cleaned_sensor_data.csv')
        df.to_csv(path, index=False)
        return path