import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from typing import Optional, Dict, List, Union
from pathlib import Path
import yaml
from .detection import AnomalyDetector


class AnomalyVisualizer:
    def __init__(self, config_file: Union[str, Path] = 'config.yaml'):
        """Инициализация визуализатора с конфигурацией из YAML-файла

        Args:
            config_file: Путь к YAML-файлу конфигурации
        """
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f).get('visualization', {})

        # Настройки стилей из конфига
        self.styles = self.config.get('styles', {})

    def plot_data_comparison(
            self,
            raw_df: pd.DataFrame,
            clean_df: pd.DataFrame,
            output_file: Optional[str] = None,
            show: Optional[bool] = None
    ) -> None:
        """Визуализация сравнения исходных и очищенных данных

        Args:
            raw_df: DataFrame с исходными данными
            clean_df: DataFrame с очищенными данными
            output_file: Путь для сохранения графика (None - не сохранять)
            show: Показывать ли график (None - из конфига)
        """
        figsize = self.styles.get('comparison_figsize', [12, 10])
        fig, axs = plt.subplots(2, 1, figsize=figsize, sharex=True)

        # Стили из конфига
        raw_style = self.styles.get('raw_data', {'color': 'blue', 'linewidth': 1})
        clean_style = self.styles.get('clean_data', {'color': 'green', 'linewidth': 1})

        # Исходные данные
        axs[0].plot(raw_df['time'], raw_df['value'],
                    label='Исходные данные', **raw_style)
        axs[0].set_title('Исходные данные')
        axs[0].grid(True, **self.styles.get('grid', {}))
        axs[0].legend()

        # Очищенные данные
        axs[1].plot(clean_df['time'], clean_df['value'],
                    label='Очищенные данные', **clean_style)
        axs[1].set_title('Данные после очистки')
        axs[1].grid(True, **self.styles.get('grid', {}))
        axs[1].legend()

        plt.tight_layout()

        if output_file or self.config.get('save_comparison', True):
            save_path = output_file or self.config.get('comparison_output', 'data_cleaning_comparison.png')
            plt.savefig(save_path, dpi=self.config.get('dpi', 300))

        if show or self.config.get('show_plots', True):
            plt.show()
        else:
            plt.close()

    def visualize_results(
            self,
            df: pd.DataFrame,
            detector: AnomalyDetector,
            output_file: Optional[str] = None,
            show: Optional[bool] = None
    ) -> None:
        """Визуализация результатов обнаружения аномалий

        Args:
            df: DataFrame с результатами обнаружения
            detector: Объект AnomalyDetector
            output_file: Путь для сохранения графика (None - не сохранять)
            show: Показывать ли график (None - из конфига)
        """
        # Получаем настройки из конфига
        figsize = self.styles.get('results_figsize', [15, 10])
        grid_spec = self.config.get('grid_spec', {'rows': 3, 'cols': 1, 'height_ratios': [3, 1, 0.5]})

        plt.figure(figsize=figsize)
        gs = plt.GridSpec(
            grid_spec['rows'],
            grid_spec['cols'],
            height_ratios=grid_spec['height_ratios']
        )

        ax1 = plt.subplot(gs[0])  # Основной график
        ax2 = plt.subplot(gs[1])  # Health index
        ax3 = plt.subplot(gs[2])  # Легенда

        # Получаем критические периоды
        crit_config = self.config.get('critical_periods', {})
        critical_periods = detector.get_critical_periods(
            df,
            method=crit_config.get('method', 'combined'),
            threshold=crit_config.get('threshold', 0.5),
            consecutive=crit_config.get('consecutive', 10)
        )

        # Стили для графиков
        main_style = self.styles.get('main_plot', {'color': 'b', 'alpha': 0.7, 'linewidth': 1})
        crit_style = self.styles.get('critical_periods', {'color': 'red', 'alpha': 0.1})
        threshold_style = self.styles.get('threshold_line', {'color': 'gray', 'linestyle': '--', 'alpha': 0.5})

        # Основной график с данными
        ax1.plot(df['time'], df['value'], label='Сигнал', **main_style)

        # Разметка критических периодов
        for start, end in critical_periods:
            start_time = df.iloc[start]['time']
            end_time = df.iloc[end]['time']
            ax1.axvspan(start_time, end_time, **crit_style)

        # Добавление аномалий разных методов
        markers = self.config.get('markers', {
            'anomaly_if': {'color': 'red', 'marker': 'o', 'label': 'Isolation Forest'},
            'anomaly_iqr': {'color': 'green', 'marker': 'x', 'label': 'IQR'},
            'anomaly_lof': {'color': 'blue', 'marker': '^', 'label': 'LOF'}
        })

        for col, style in markers.items():
            if col in df.columns:
                anomalies = df[df[col] == 1]
                ax1.scatter(anomalies['time'], anomalies['value'],
                            c=style['color'], s=30, marker=style['marker'],
                            label=style['label'], alpha=0.7)

        ax1.set_title('Обнаружение аномалий', pad=20)
        ax1.grid(True, **self.styles.get('grid', {}))
        ax1.legend(loc='upper left')

        # График Health Index
        if 'health_index_combined' in df.columns:
            health_style = self.styles.get('health_index', {'color': 'purple', 'linewidth': 1.5})
            ax2.plot(df['time'], df['health_index_combined'],
                     label='Индекс исправности', **health_style)

            # Разметка критических периодов
            for start, end in critical_periods:
                start_time = df.iloc[start]['time']
                end_time = df.iloc[end]['time']
                ax2.axvspan(start_time, end_time, **crit_style)

            # Пороговая линия
            ax2.axhline(y=crit_config.get('threshold', 0.5), **threshold_style)

        ax2.set_ylim(0, 1.05)
        ax2.set_ylabel('Индекс исправности')
        ax2.grid(True, **self.styles.get('grid', {}))
        ax2.legend(loc='upper left')

        # Легенда для критических периодов
        handles = [
            Rectangle((0, 0), 1, 1,
                      color=crit_style['color'],
                      alpha=crit_style['alpha'],
                      label='Критические периоды'),
            plt.Line2D([0], [0],
                       color=threshold_style['color'],
                       linestyle=threshold_style['linestyle'],
                       label=f"Порог ({crit_config.get('threshold', 0.5)})")
        ]
        ax3.axis('off')
        ax3.legend(handles=handles, loc='center', ncol=2)

        plt.tight_layout()

        if output_file or self.config.get('save_results', True):
            save_path = output_file or self.config.get('results_output', 'anomaly_detection_results.png')
            plt.savefig(save_path, dpi=self.config.get('dpi', 300), bbox_inches='tight')

        if show or self.config.get('show_plots', True):
            plt.show()
        else:
            plt.close()

    def plot_health_history(
            self,
            df: pd.DataFrame,
            methods: List[str] = ['combined'],
            output_file: Optional[str] = None,
            show: Optional[bool] = None
    ) -> None:
        """Визуализация истории изменения health index

        Args:
            df: DataFrame с health index
            methods: Список методов для отображения
            output_file: Путь для сохранения графика
            show: Показывать ли график
        """
        figsize = self.styles.get('health_figsize', [12, 6])
        plt.figure(figsize=figsize)

        colors = self.styles.get('health_colors', ['purple', 'blue', 'green', 'orange'])

        for i, method in enumerate(methods):
            col = f'health_index_{method}'
            if col in df.columns:
                plt.plot(df['time'], df[col],
                         color=colors[i % len(colors)],
                         label=f'Health Index ({method})',
                         linewidth=1.5)

        # Пороговая линия
        crit_config = self.config.get('critical_periods', {})
        threshold = crit_config.get('threshold', 0.5)
        plt.axhline(y=threshold,
                    color='gray',
                    linestyle='--',
                    alpha=0.5,
                    label=f'Порог ({threshold})')

        plt.title('История изменения индекса исправности')
        plt.ylabel('Индекс исправности')
        plt.ylim(0, 1.05)
        plt.grid(True, **self.styles.get('grid', {}))
        plt.legend()
        plt.tight_layout()

        if output_file or self.config.get('save_health', True):
            save_path = output_file or self.config.get('health_output', 'health_index_history.png')
            plt.savefig(save_path, dpi=self.config.get('dpi', 300))

        if show or self.config.get('show_plots', True):
            plt.show()
        else:
            plt.close()


# Функции для обратной совместимости
def plot_data_comparison(*args, **kwargs):
    visualizer = AnomalyVisualizer()
    visualizer.plot_data_comparison(*args, **kwargs)


def visualize_results(*args, **kwargs):
    visualizer = AnomalyVisualizer()
    visualizer.visualize_results(*args, **kwargs)