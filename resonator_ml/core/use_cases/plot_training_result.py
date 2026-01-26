from resonator_ml.machine_learning.view.training import TimeSeriesPlotter
import random

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
            if n > self.series_provider.num_plots() - 11:
                plotter.add_series(self.series_provider.data_at(n), label=self.series_provider.title_at(n),
                               linestyle=random.choice(["-", "--", "-.", ":"]))


        # Passe den Plot an
        plotter.customize(
            title='Epoch vs Loss',
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