import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import yaml
from typing import Optional, Dict, List, Union, Tuple
from pathlib import Path


class AnomalyDetector:
    def __init__(self, config_file: Union[str, Path] = None):
        """Инициализация детектора аномалий с конфигурацией из YAML-файла

        Args:
            config_file (Union[str, Path], optional): Путь к YAML-файлу конфигурации.
                Если None, будет использован файл config.yaml из директории пакета.
        """
        # Определяем путь к конфигурационному файлу
        if config_file is None:
            # Получаем путь к директории текущего модуля
            package_dir = Path(__file__).parent
            config_file = package_dir / "config.yaml"

        try:
            with open(config_file, 'r') as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Конфигурационный файл не найден по пути: {config_file}\n"
                "Пожалуйста, укажите правильный путь к файлу конфигурации или "
                "разместите файл config.yaml в директории пакета."
            )
        except yaml.YAMLError as e:
            raise ValueError(f"Ошибка при чтении YAML-файла: {str(e)}")

        # Проверка наличия обязательных секций в конфиге
        if not isinstance(self.config, dict):
            raise ValueError("Конфигурационный файл должен содержать словарь")

        # Инициализация доступных методов
        self.methods = {
            'iqr': self._iqr_filter,
            'iforest': self._isolation_forest,
            'lof': self._local_outlier_factor
        }

    def load_clean_data(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """Загрузка очищенных данных из CSV файла

        Args:
            file_path: Путь к CSV файлу. Если None, берется из конфига

        Returns:
            pd.DataFrame: Загруженные данные с преобразованной колонкой времени
        """
        path = file_path or self.config.get('preprocessing', {}).get('output_file', 'cleaned_sensor_data.csv')
        df = pd.read_csv(path)
        df['time'] = pd.to_datetime(df['time'])
        return df

    def detect_anomalies(
            self,
            df: pd.DataFrame,
            methods: Optional[List[str]] = None,
            inplace: bool = False
    ) -> pd.DataFrame:
        """Основной метод детектирования аномалий

        Args:
            df: DataFrame с данными для анализа
            methods: Список методов для применения. Если None, берется из конфига
            inplace: Модифицировать ли исходный DataFrame

        Returns:
            pd.DataFrame: DataFrame с добавленными колонками аномалий
        """
        if not inplace:
            df = df.copy()

        methods = methods or self.config.get('methods', ['iqr', 'iforest', 'lof'])

        for method in methods:
            if method in self.methods:
                df = self.methods[method](df)

        # Комбинируем результаты разных методов
        self._combine_results(df, methods)

        return df

    def evaluate_health_index(
            self,
            df: pd.DataFrame,
            window: Optional[int] = None
    ) -> Union[Dict[str, float], pd.DataFrame]:
        """Оценка индекса исправности системы

        Args:
            df: DataFrame с результатами детектирования
            window: Размер окна для скользящего среднего. Если None, глобальная оценка

        Returns:
            Если window=None: словарь с оценками для каждого метода
            Иначе: DataFrame с добавленными колонками health_index
        """
        if window is None:
            return self._global_health_evaluation(df)
        return self._rolling_health_evaluation(df, window)

    def get_critical_periods(
            self,
            df: pd.DataFrame,
            method: str = 'combined',
            threshold: Optional[float] = None,
            consecutive: Optional[int] = None
    ) -> List[tuple]:
        # Убедимся, что индекс целочисленный
        df = df.reset_index(drop=True)

        # Получаем параметры из конфига
        crit_config = self.config.get('critical_periods', {})
        threshold = threshold or crit_config.get('threshold', 0.5)
        consecutive = consecutive or crit_config.get('consecutive', 10)

        # Проверяем наличие необходимой колонки
        health_col = f'health_index_{method}'
        if health_col not in df.columns:
            raise ValueError(f"DataFrame должен содержать колонку '{health_col}'")

        # Основная логика поиска периодов
        alerts = []
        count = 0
        start_idx = None

        for i in range(len(df)):
            value = df.at[i, health_col]
            if value < threshold:
                if count == 0:
                    start_idx = i
                count += 1
            else:
                if count >= consecutive:
                    alerts.append((start_idx, i - 1))
                count = 0
                start_idx = None

        if count >= consecutive:
            alerts.append((start_idx, len(df) - 1))

        return alerts

    def _isolation_forest(self, df: pd.DataFrame) -> pd.DataFrame:
        """Метод Isolation Forest"""
        params = self.config.get('iforest', {})
        model = IsolationForest(
            contamination=params.get('contamination', 0.05),
            random_state=params.get('random_state', 42),
            n_estimators=params.get('n_estimators', 100)
        )
        df['anomaly_if'] = model.fit_predict(df[['value']])
        df['anomaly_if'] = df['anomaly_if'].map({1: 0, -1: 1})
        return df

    def _local_outlier_factor(self, df: pd.DataFrame) -> pd.DataFrame:
        """Метод Local Outlier Factor"""
        params = self.config.get('lof', {})
        model = LocalOutlierFactor(
            n_neighbors=params.get('n_neighbors', 20),
            contamination=params.get('contamination', 0.05),
            novelty=params.get('novelty', False)
        )
        df['anomaly_lof'] = model.fit_predict(df[['value']])
        df['anomaly_lof'] = df['anomaly_lof'].map({1: 0, -1: 1})
        return df

    def _iqr_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """Метод межквартильного размаха (IQR)"""
        params = self.config.get('iqr', {})
        column = params.get('column', 'value')
        threshold = params.get('threshold', 1.5)

        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr

        df['anomaly_iqr'] = ((df[column] < lower_bound) | (df[column] > upper_bound)).astype(int)
        return df

    def _combine_results(self, df: pd.DataFrame, methods: List[str]) -> None:
        """Комбинирование результатов разных методов"""
        anomaly_cols = [f'anomaly_{m[:3]}' if m != 'iforest' else 'anomaly_if' for m in methods]
        anomaly_cols = [col for col in anomaly_cols if col in df.columns]

        if anomaly_cols:
            df['anomaly_combined'] = df[anomaly_cols].mean(axis=1)
            df['health_index_combined'] = 1 - df['anomaly_combined']

    def _global_health_evaluation(self, df: pd.DataFrame) -> Dict[str, float]:
        """Глобальная оценка health index (без окон)"""
        results = {}
        for method in self.config.get('methods', ['iqr', 'iforest', 'lof']):
            col = f'anomaly_{method[:3]}' if method != 'iforest' else 'anomaly_if'
            if col in df.columns:
                results[method] = 1 - df[col].mean()

        if 'anomaly_combined' in df.columns:
            results['combined'] = 1 - df['anomaly_combined'].mean()

        return results

    def _rolling_health_evaluation(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        """Оценка health index с использованием скользящего окна"""
        df = df.copy()
        for method in self.config.get('methods', ['iqr', 'iforest', 'lof']):
            col = f'anomaly_{method[:3]}' if method != 'iforest' else 'anomaly_if'
            if col in df.columns:
                df[f'health_index_{method[:3]}'] = 1 - df[col].rolling(window=window, min_periods=1).mean()

        if 'anomaly_combined' in df.columns:
            df['health_index_combined'] = 1 - df['anomaly_combined'].rolling(window=window, min_periods=1).mean()

        return df