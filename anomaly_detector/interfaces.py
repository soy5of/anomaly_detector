from typing import Optional, Union, Dict
from pydantic import BaseModel
from enum import Enum
from pathlib import Path
import pandas as pd


class ProcessingStage(str, Enum):
    PREPROCESSING = "preprocessing"
    DETECTION = "detection"
    VISUALIZATION = "visualization"


class PipelineConfig(BaseModel):
    run_preprocessing: bool = True
    run_detection: bool = True
    run_visualization: bool = True
    save_results: bool = True
    show_plots: bool = True


class AnomalyDetectionPipeline:
    """Основной класс для выполнения полного пайплайна"""

    def __init__(self, config_path: Union[str, Path] = "config.yaml"):
        self.config_path = config_path
        self.visualizer = None
        self.detector = None
        self.preprocessor = None

    def run_full_pipeline(self, input_file: Optional[str] = None) -> Dict:
        """Выполняет полный пайплайн обработки"""
        from .preprocessing import DataPreprocessor
        from .detection import AnomalyDetector
        from .visualization import AnomalyVisualizer

        # Инициализация компонентов
        self.preprocessor = DataPreprocessor(self.config_path)
        raw_df = self.preprocessor.load_data(input_file)
        clean_df = self.preprocessor.clean_data(raw_df)

        # Сохраняем очищенные данные
        clean_path = self.preprocessor.save_clean_data(clean_df)

        # Детекция аномалий
        self.detector = AnomalyDetector(self.config_path)
        data = self.detector.load_clean_data(clean_path)
        result = self.detector.detect_anomalies(data)

        health_cfg = self.detector.config.get('health_index', {})
        method = health_cfg.get('method', 'combined')
        window = health_cfg.get('window', None)

        # Всегда вычисляем и выводим глобальный health index
        if method == 'combined' and 'anomaly_combined' in result.columns:
            global_health = 1 - result['anomaly_combined'].mean()
            print(f"\nГлобальный индекс исправности ({method.upper()}): {global_health:.2%}")

            # Дополнительно вычисляем скользящее среднее, если указано окно
            if window is not None:
                result['health_index_combined'] = 1 - result['anomaly_combined'].rolling(
                    window=window, min_periods=1).mean()
                print(f"Используется скользящее окно: {window} точек")
            else:
                result['health_index_combined'] = global_health

        # Визуализация
        self.visualizer = AnomalyVisualizer(self.config_path)
        self.visualizer.plot_data_comparison(raw_df, clean_df)

        # Убедимся, что индекс целочисленный
        result.reset_index(drop=True, inplace=True)
        self.visualizer.visualize_results(result, self.detector)

        # Анализ критических периодов
        if 'health_index_combined' in result.columns:
            alerts = self.detector.get_critical_periods(result, method=method)

            # Вычисляем процент критического состояния
            critical_percentage = (result['health_index_combined'] < health_cfg.get('threshold', 0.5)).mean()
            print(f"\nПроцент времени в критическом состоянии: {critical_percentage:.2%}")

            if alerts:
                print(f"\nКритические периоды ({method.upper()}):")
                for start, end in alerts:
                    duration = end - start + 1
                    health_value = result.loc[start:end, 'health_index_combined'].mean()
                    print(f"Период: {result.iloc[start]['time']} - {result.iloc[end]['time']} "
                          f"(длительность: {duration} точек, средний индекс: {health_value:.2f})")
            else:
                print(f"\nДля метода {method.upper()} критических периодов не обнаружено")
        else:
            print("\nКолонка 'health_index_combined' отсутствует в DataFrame")

        return {
            "raw_data": raw_df,
            "clean_data": clean_df,
            "result": result,
            "clean_data_path": clean_path,
            "global_health_index": global_health if 'global_health' in locals() else None,
            "critical_percentage": critical_percentage if 'critical_percentage' in locals() else None
        }

    def plot_comparison(self, raw_df: pd.DataFrame, clean_df: pd.DataFrame):
        """Визуализация сравнения сырых и очищенных данных"""
        if self.visualizer is None:
            from .visualization import AnomalyVisualizer
            self.visualizer = AnomalyVisualizer(self.config_path)
        self.visualizer.plot_data_comparison(raw_df, clean_df)

    def plot_results(self, results: pd.DataFrame):
        """Визуализация результатов обнаружения аномалий"""
        if self.visualizer is None:
            from .visualization import AnomalyVisualizer
            self.visualizer = AnomalyVisualizer(self.config_path)
        if self.detector is None:
            from .detection import AnomalyDetector
            self.detector = AnomalyDetector(self.config_path)
        self.visualizer.visualize_results(results, self.detector)