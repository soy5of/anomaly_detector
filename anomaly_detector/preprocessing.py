import pandas as pd
import matplotlib.pyplot as plt


class DataCleaner:
    def __init__(self, file_path):
        self.df = pd.read_csv(file_path)

    def clean(self):
        self._remove_duplicates()
        self._handle_missing()
        self._convert_time()
        return self._remove_obvious_anomalies()

    def _remove_duplicates(self):
        self.df = self.df.drop_duplicates()

    def _handle_missing(self):
        self.df.fillna(self.df.median(), inplace=True)

    def _convert_time(self):
        self.df['time'] = pd.to_datetime(self.df['time'])

    def _remove_obvious_anomalies(self, column='value', bounds=(-20, 20)):
        mask = (self.df[column] >= bounds[0]) & (self.df[column] <= bounds[1])
        cleaned = self.df[mask].copy()
        return cleaned, len(self.df) - len(cleaned)

    def plot_comparison(self, cleaned_df):
        fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

        # Исходные данные
        axs[0].plot(self.df['time'], self.df['value'],
                    label='Исходные данные', color='blue', linewidth=1)
        axs[0].set_title('Исходные данные')
        axs[0].set_ylabel('Значение датчика')
        axs[0].grid(True)
        axs[0].legend()

        # Очищенные данные
        axs[1].plot(cleaned_df['time'], cleaned_df['value'],
                    label='Данные после очистки', color='green', linewidth=1)
        axs[1].set_title('Данные после удаления аномалий')
        axs[1].set_xlabel('Время')
        axs[1].set_ylabel('Значение датчика')
        axs[1].grid(True)
        axs[1].legend()

        plt.tight_layout()
        return fig