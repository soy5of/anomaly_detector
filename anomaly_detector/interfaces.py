from typing import Optional, Union
from pydantic import BaseModel
from enum import Enum
from pathlib import Path


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

    def run_full_pipeline(self, input_file: Optional[str] = None):
        """Выполняет полный пайплайн обработки"""
        from .preprocessing import DataPreprocessor
        from .detection import AnomalyDetector
        from .visualization import visualize_results, plot_data_comparison

        # 1. Препроцессинг
        preprocessor = DataPreprocessor(self.config_path)
        raw_df = preprocessor.load_data(input_file)
        clean_df = preprocessor.clean_data(raw_df)
        clean_path = preprocessor.save_clean_data(clean_df)

        # 2. Детекция аномалий
        detector = AnomalyDetector(self.config_path)
        data = detector.load_clean_data(clean_path)
        result = detector.detect_anomalies(data)

        # 3. Визуализация
        plot_data_comparison(raw_df, clean_df)
        visualize_results(result, detector)

        return {
            "raw_data": raw_df,
            "clean_data": clean_df,
            "result": result,
            "clean_data_path": clean_path
        }