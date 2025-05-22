from .preprocessing import DataCleaner
from .detection import AnomalyDetector


class AnomalyPipeline:
    def __init__(self, file_path):
        self.file_path = file_path

    def run(self):
        # 1. Очистка
        cleaner = DataCleaner(self.file_path)
        cleaned_data, _ = cleaner.clean()

        # 2. Детекция
        detector = AnomalyDetector(cleaned_data)
        result = detector.detect_all()

        # 3. Визуализация (опционально)
        # cleaner.plot_comparison(cleaned_data)

        return result