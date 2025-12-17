import numpy as np
import soundfile as sf


def karplus_strong(
    frequency=220.0,      # Tonhöhe in Hz
    duration=3.0,         # Länge in Sekunden
    sample_rate=44100,    # Abtastrate
    decay=0.996           # Dämpfung (0.99–0.999)
):
    """
    Erzeugt einen ausgeschwungenen Ton mit dem Karplus-Strong-Algorithmus.
    """
    buffer_size = int(sample_rate / frequency)

    # Initialer Noise-Impuls (gezupfte Saite)
    buffer = np.random.uniform(-1.0, 1.0, buffer_size)

    num_samples = int(sample_rate * duration)
    output = np.zeros(num_samples, dtype=np.float32)

    for i in range(num_samples):
        # Einfaches Tiefpass-Feedback
        avg = decay * 0.5 * (buffer[0] + buffer[1])

        output[i] = buffer[0]

        # Ringpuffer-Update
        buffer[:-1] = buffer[1:]
        buffer[-1] = avg

    # Normalisieren
    max_val = np.max(np.abs(output))
    if max_val > 0:
        output /= max_val

    return output


if __name__ == "__main__":
    sample_rate = 44100

    tone = karplus_strong(
        frequency=83.05,  # Low E string as in Strat File
        duration=20.0,
        sample_rate=sample_rate,
        decay=0.995
    )

    # Schreiben mit soundfile
    sf.write(
        file="data/raw/KS_E_0_plectrum.wav",
        data=tone,
        samplerate=sample_rate,
        subtype="FLOAT"
    )

    print("WAV-Datei erzeugt: karplus_strong_tone.wav")
