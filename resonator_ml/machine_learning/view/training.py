import matplotlib.pyplot as plt
import numpy as np


class TimeSeriesPlotter:
    def __init__(self, figsize=(10, 6)):
        """
        Initialisiert den Plotter für Zeitreihen.

        Args:
            figsize: Tuple für die Größe der Figur (Breite, Höhe)
        """
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.series_count = 0

    def add_series(self, y_data, x_data=None, label=None, color=None, linestyle='-', linewidth=2):
        """
        Fügt eine Zeitreihe zum Plot hinzu.

        Args:
            y_data: Liste oder Array mit y-Werten
            x_data: Liste oder Array mit x-Werten (optional, wird automatisch generiert falls None)
            label: Beschriftung für die Legende
            color: Farbe der Linie (optional)
            linestyle: Linienstil ('-', '--', '-.', ':')
            linewidth: Breite der Linie
        """
        if x_data is None:
            x_data = np.arange(len(y_data))

        if label is None:
            label = f'Serie {self.series_count + 1}'

        self.ax.plot(x_data, y_data, label=label, color=color,
                     linestyle=linestyle, linewidth=linewidth)
        self.series_count += 1

    def customize(self, title=None, xlabel='Zeit', ylabel='Wert',
                  grid=True, legend=True, log_scale=False):
        """
        Passt das Aussehen des Plots an.

        Args:
            title: Titel des Plots
            xlabel: Beschriftung der x-Achse
            ylabel: Beschriftung der y-Achse
            grid: Zeige Gitternetz
            legend: Zeige Legende
            log_scale: Setze y-Achse auf logarithmische Skalierung
        """
        if title:
            self.ax.set_title(title, fontsize=14, fontweight='bold')
        self.ax.set_xlabel(xlabel, fontsize=12)
        self.ax.set_ylabel(ylabel, fontsize=12)

        if grid:
            self.ax.grid(True, alpha=0.3, linestyle='--')

        if legend and self.series_count > 0:
            self.ax.legend(loc='best', framealpha=0.9)

        # Logarithmische Skalierung der y-Achse
        if log_scale:
            self.ax.set_yscale('log')

        # Automatische Skalierung (wird bereits standardmäßig von matplotlib gemacht)
        self.ax.autoscale(enable=True, axis='both', tight=False)

    def show(self):
        """Zeigt den Plot an."""
        plt.tight_layout()
        plt.show()

    def save(self, filename, dpi=300):
        """
        Speichert den Plot als Datei.

        Args:
            filename: Name der Datei (z.B. 'plot.png')
            dpi: Auflösung
        """
        plt.tight_layout()
        self.fig.savefig(filename, dpi=dpi, bbox_inches='tight')