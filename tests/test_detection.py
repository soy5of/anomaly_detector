import unittest
import pandas as pd
from anomaly_detector.detection import AnomalyDetector

class TestAnomalyDetector(unittest.TestCase):
    def setUp(self):
        """Подготовка тестовых данных."""
        self.detector = AnomalyDetector()
        self.data = pd.DataFrame({"value": [1, 2, 3, 100]})  # Пример с аномалией (100)

    def test_iqr_detection(self):
        """Проверка работы IQR-метода."""
        result = self.detector.detect_anomalies(self.data, methods=["iqr"])
        self.assertIn("anomaly_iqr", result.columns)  # Проверяем, что колонка создана
        self.assertEqual(result["anomaly_iqr"].sum(), 1)  # Должна быть 1 аномалия

if __name__ == "__main__":
    unittest.main()