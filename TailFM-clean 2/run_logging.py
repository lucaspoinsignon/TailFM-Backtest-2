"""Mirror everything printed by a run into a text file (`tee`).

    with tee_output(f"{outdir}/report.log", header="run_baselines.py"):
        run(args)

The process-level streams are swapped for the duration of the block, so the on-disk
report is byte-identical to the terminal output.  Only Python-level writes are
captured, not output from C extensions or subprocesses.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import io
import os
import sys
import traceback


class Tee(io.TextIOBase):
    """Write-through text stream duplicating each write to several streams."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, s: str) -> int:
        for st in self._streams:
            st.write(s)
            st.flush()          # keep the log complete if the run is killed
        return len(s)

    def flush(self) -> None:
        for st in self._streams:
            st.flush()

    def isatty(self) -> bool:
        return False            # plain lines, not carriage-return animations


@contextlib.contextmanager
def tee_output(path: str, header: str | None = None, mode: str = "w",
               capture_stderr: bool = True):
    """Duplicate stdout (and stderr) into `path` for the duration of the block, with a
    timestamp + command-line banner first and a traceback appended if the block raises.
    """
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    fh = open(path, mode, buffering=1)                      # line buffered
    stdout0, stderr0 = sys.stdout, sys.stderr
    sys.stdout = Tee(stdout0, fh)
    if capture_stderr:
        sys.stderr = Tee(stderr0, fh)
    started = dt.datetime.now()
    try:
        if header:
            print(f"# {header}")
        print(f"# started  {started.isoformat(timespec='seconds')}")
        print(f"# command  {' '.join(sys.argv)}")
        print(f"# log      {os.path.abspath(path)}")
        yield fh
    except BaseException:
        traceback.print_exc(file=fh)                        # failures stay in the log
        raise
    finally:
        elapsed = (dt.datetime.now() - started).total_seconds()
        print(f"\n# finished {dt.datetime.now().isoformat(timespec='seconds')} "
              f"({elapsed / 60:.1f} min)")
        sys.stdout, sys.stderr = stdout0, stderr0
        fh.close()
