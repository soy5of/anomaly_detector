<h3>Here lies my diploma project on Anomaly (Outliers) Detection topic.</h3>

# Anomaly Detector

Advanced anomaly detection package for sensor data with multiple algorithms:

- Isolation Forest
- Local Outlier Factor (LOF)
- Interquartile Range (IQR)
- Combined approach


## Installation

```bash
pip install anomaly-detector
```

## Basic usage

```python
from anomaly_detector import AnomalyDetectionPipeline

pipeline = AnomalyDetectionPipeline()
results = pipeline.run_full_pipeline("sensor_data.csv")
```
