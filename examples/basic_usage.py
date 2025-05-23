from anomaly_detector import AnomalyDetectionPipeline

# Создание и запуск пайплайна
pipeline = AnomalyDetectionPipeline("config.yaml")
results = pipeline.run_full_pipeline("sensor_data.csv")

