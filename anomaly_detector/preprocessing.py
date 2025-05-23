import pandas as pd
import yaml
from typing import Optional, Union
from pathlib import Path


class DataPreprocessor:
    def __init__(self, config_file: Union[str, Path] = None):
        """Инициализация обработчика данных с конфигурацией из YAML-файла.

        Args:
            config_file (Union[str, Path], optional): Путь к YAML-файлу конфигурации.
                Если None, будет использован файл config.yaml из директории пакета.

        Raises:
            FileNotFoundError: Если конфигурационный файл не найден.
            ValueError: Если файл содержит невалидный YAML или структуру данных.
        """
        # Определение пути к конфигурационному файлу
        config_path = self._resolve_config_path(config_file)

        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f) or {}  # На случай пустого файла
                self.config = config.get('preprocessing', {})
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Конфигурационный файл не найден: {e}\n"
                f"Ожидаемый путь: {config_path}\n"
                "Проверьте наличие файла или укажите правильный путь."
            ) from e
        except yaml.YAMLError as e:
            raise ValueError(f"Ошибка в YAML-файле конфигурации: {str(e)}") from e

        # Установка значений по умолчанию
        self._set_default_config()

    def _resolve_config_path(self, config_file: Union[str, Path, None]) -> Path:
        """Определяет путь к конфигурационному файлу."""
        if config_file is not None:
            return Path(config_file)
        return Path(__file__).parent / "config.yaml"

    def _set_default_config(self):
        """Устанавливает значения по умолчанию для конфигурации."""
        defaults = {
            'input_file': 'sensor_data.csv',
            'output_file': 'cleaned_sensor_data.csv',
            'bounds': {'lower': -15, 'upper': 15}
        }

        for key, value in defaults.items():
            self.config.setdefault(key, value)

    def load_data(self, file_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
        """Загружает данные из CSV-файла.

        Args:
            file_path (Optional[Union[str, Path]]): Путь к CSV-файлу.
                Если None, используется путь из конфигурации.

        Returns:
            pd.DataFrame: Загруженные данные.

        Raises:
            FileNotFoundError: Если файл данных не найден.
            ValueError: Если файл данных имеет неверный формат.
        """
        data_path = Path(file_path) if file_path else Path(self.config['input_file'])
        try:
            df = pd.read_csv(data_path)
            if 'value' not in df.columns:
                raise ValueError("Файл данных должен содержать колонку 'value'")
            # Если time нет — добавляем индекс как time
            if 'time' not in df.columns:
                df['time'] = df.index
            # Если time есть и это числа — переводим в секунды (если это наносекунды)
            elif df['time'].dtype in (int, float):
                df['time'] = df['time'] / 1_000_000_000  # наносекунды -> секунды
            return df
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Файл данных не найден: {data_path}") from e
        except pd.errors.EmptyDataError as e:
            raise ValueError(f"Файл данных пуст: {data_path}") from e

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Очищает данные от аномалий и пропусков.

        Args:
            df (pd.DataFrame): Исходные данные для очистки.

        Returns:
            pd.DataFrame: Очищенные данные.
        """
        # Создаем копию, чтобы не изменять исходные данные
        cleaned_df = df.copy()

        # Удаление дубликатов
        cleaned_df = cleaned_df.drop_duplicates()

        # Удаление пропусков
        cleaned_df = cleaned_df.dropna(subset=['value'])

        # Фильтрация по граничным значениям
        bounds = self.config['bounds']
        mask = (cleaned_df['value'] >= bounds['lower']) & (cleaned_df['value'] <= bounds['upper'])
        return cleaned_df[mask]

    def save_clean_data(self, df: pd.DataFrame, file_path: Optional[Union[str, Path]] = None) -> Path:
        """Сохраняет очищенные данные в CSV-файл.

        Args:
            df (pd.DataFrame): Данные для сохранения.
            file_path (Optional[Union[str, Path]]): Путь для сохранения.
                Если None, используется путь из конфигурации.

        Returns:
            Path: Путь к сохраненному файлу.

        Raises:
            ValueError: Если DataFrame пуст.
        """
        if df.empty:
            raise ValueError("Нельзя сохранить пустой DataFrame")

        output_path = Path(file_path) if file_path else Path(self.config['output_file'])
        df.to_csv(output_path, index=False)
        return output_path