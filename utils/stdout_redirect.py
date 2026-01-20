from pathlib import Path
from datetime import datetime
import sys

class TimestampedTeeStream:
    def __init__(self, *streams, time_format="%Y-%m-%d %H:%M:%S"):
        self.streams = streams
        self.time_format = time_format
        self._at_line_start = True
        self._buffer = ""

    def write(self, data: str):
        if not data:
            return

        self._buffer += data

        while True:
            if "\n" in self._buffer:
                line, self._buffer = self._buffer.split("\n", 1)
                self._write_line(line + "\n")
            else:
                break

    def _write_line(self, data: str):
        if self._at_line_start:
            ts = datetime.now().strftime(self.time_format)
            prefix = f"[{ts}] "
            for s in self.streams:
                s.write(prefix)

        for s in self.streams:
            s.write(data)

        self._at_line_start = data.endswith("\n")

        for s in self.streams:
            s.flush()

    def flush(self):
        if self._buffer:
            self._write_line(self._buffer)
            self._buffer = ""

        for s in self.streams:
            s.flush()


def redirect_stdout_to_file(log_dir: str | Path, script_name: str, also_console: bool = True):
    """
    Redirects stdout and stderr to a timestamped logfile.

    :param log_dir: Directory where logs are stored
    :param script_name: Name of the script / component (used in filename)
    :param also_console: If True, output is still shown in console
    """

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logfile = log_dir / f"{script_name}.log"

    f = open(logfile, "w", buffering=1, encoding="utf-8")

    if also_console:
        sys.stdout = TimestampedTeeStream(sys.__stdout__, f)
        sys.stderr = TimestampedTeeStream(sys.__stderr__, f)
    else:
        sys.stdout = f
        sys.stderr = f

    return logfile
