# anomaly_detector/__init__.py

from .interfaces import AnomalyPipeline  # базовый импорт
from .preprocessing import DataCleaner  # можно оставить явные импорты для важных компонентов
from .detection import AnomalyDetector


# Автоматический импорт ВСЕХ классов из всех модулей
def _import_components():
    import inspect
    from pathlib import Path

    components = {}
    for module_file in Path(__file__).parent.glob('*.py'):
        if module_file.name != '__init__.py':
            module_name = module_file.stem
            module = __import__(f'anomaly_detector.{module_name}', fromlist=['*'])
            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and obj.__module__ == f'anomaly_detector.{module_name}':
                    components[name] = obj
    return components


# Обновляем глобальное пространство имён
_components = _import_components()
globals().update(_components)

# Версия пакета
__version__ = "0.1.0"

# Контролируем что попадает в import *
__all__ = list(_components.keys()) + ['AnomalyPipeline']  # добавляем вручную, если нужно

# Документация пакета
__doc__ = "Автоматически загруженные компоненты: " + ", ".join(_components.keys())