import matplotlib.pyplot as plt

from resonator_ml.audio.io import load_wav_mono
from resonator_ml.audio.metering import compute_spectrogram


class PlotSpectrumComparison:
    def __init__(self, wav_paths: dict[str, str]):
        self.wav_paths = wav_paths

    def execute(self):
        last_sample_rate = None

        fig, axes = plt.subplots(len(self.wav_paths), 1, figsize=(12, 6), sharex=True, sharey=True)
        last_im = None
        frequencies = []
        spectra = []
        timelines = []
        vmin = float("inf")
        vmax = float("-inf")
        for ax_id, path_key in enumerate(self.wav_paths.keys()):
            path = self.wav_paths[path_key]
            # ---- WAVs laden ----
            fs1, x1 = load_wav_mono(path)


            assert last_sample_rate is None or fs1 == last_sample_rate, "Samplerates müssen gleich sein"
            last_sample_rate = fs1

            # ---- Spektrogramme ----
            f1, t1, S1 = compute_spectrogram(x1, fs1, n_fft=2048)

            frequencies.append(f1)
            spectra.append(S1)
            timelines.append(t1)

            # Todo: Gemeinsame Farbskala
            vmin = min(vmin, S1.min())
            vmax = max(vmax, S1.max())

        # Find vmin/vmax first. Common scale is important for comparison So iterate overall wavs to find it.

        # Then iterate again
        for ax_id, path_key in enumerate(self.wav_paths.keys()):
            f1 = frequencies[ax_id]
            S1 = spectra[ax_id]
            t1 = timelines[ax_id]

            # 0 Hz entfernen für log darstellung
            f = f1
            S = S1

            f = f[1:]
            S = S[1:, :]
            # ---- Plot ----
            last_im = fig.axes[ax_id].imshow(
                S,
                origin="lower",
                aspect="auto",
                extent=(t1[0], t1[-1], f[0], f[-1]),
                vmin=vmin,
                vmax=vmax,
                cmap="magma",
            )
            fig.axes[ax_id].set_title(path_key)
            fig.axes[ax_id].set_ylabel("Frequenz [Hz]")


        fig.colorbar(last_im, ax=fig.axes, label="Magnitude [dB]")
        # plt.tight_layout()
        plt.yscale("log")
        plt.show()

