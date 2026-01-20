import matplotlib.pyplot as plt
import numpy as np

class BatchFeatureViewer:
    def __init__(self, inputs, targets):
        """
        inputs:  Tensor [B, F_in]
        targets: Tensor [B, F_out]
        """
        assert inputs.ndim == 2, "inputs must be [B, F_in]"
        assert targets.ndim == 2, "targets must be [B, F_out]"

        self.inputs = inputs.detach().cpu()
        self.targets = targets.detach().cpu()

        self.B, self.F_in = self.inputs.shape
        _, self.F_out = self.targets.shape

        self.idx = 0

        self.fig, self.axes = plt.subplots(
            nrows=2,
            figsize=(14, 6),
            gridspec_kw={"height_ratios": [3, 1]}
        )

        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.draw()

    def draw(self):
        for ax in self.axes:
            ax.clear()

        x_in = np.arange(self.F_in)
        x_out = np.arange(self.F_out)

        # Inputs
        bars_in = self.axes[0].bar(x_in, self.inputs[self.idx].numpy())
        self.axes[0].set_title(f"Sample {self.idx+1}/{self.B} — Inputs")
        self.axes[0].set_ylabel("Value")
        self.axes[0].set_xlabel("Input Feature")

        # Targets
        bars_out = self.axes[1].bar(x_out, self.targets[self.idx].numpy(), color="yellow")
        self.axes[1].set_title("Targets")
        self.axes[1].set_ylabel("Value")
        self.axes[1].set_xlabel("Target Feature")

        for bar in bars_in:
            height = bar.get_height()
            self.axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                height/2,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=8
            )

        for bar in bars_out:
            height = bar.get_height()
            self.axes[1].text(
                bar.get_x() + bar.get_width() / 2,
                height/2,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=8
            )

        plt.tight_layout()
        self.fig.canvas.draw_idle()

    def on_key(self, event):
        if event.key == "right":
            self.idx = (self.idx + 1) % self.B
            self.draw()
        elif event.key == "left":
            self.idx = (self.idx - 1) % self.B
            self.draw()