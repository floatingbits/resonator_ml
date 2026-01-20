from resonator_ml.machine_learning.view.training import TimeSeriesPlotter
import numpy as np

from resonator_ml.ports.series_provider import SeriesProvider


class PlotTrainingResult:
    def __init__(self, series_provider: SeriesProvider):
        self.series_provider = series_provider

    def execute(self):
        # Erstelle Beispiel-Daten

        # Erstelle Plotter
        plotter = TimeSeriesPlotter(figsize=(12, 6))

        # Füge mehrere Zeitreihen hinzu
        for n in range(self.series_provider.num_plots()):
            plotter.add_series(self.series_provider.data_at(n), label=self.series_provider.title_at(n))


        # Passe den Plot an
        plotter.customize(
            title='Zeitreihen-Darstellung',
            xlabel='Epoch',
            ylabel='Loss',
            grid=True,
            legend=True,
            log_scale=True
        )

        # Zeige den Plot
        plotter.show()

        # Optional: Speichere den Plot
        # plotter.save('zeitreihe.png')