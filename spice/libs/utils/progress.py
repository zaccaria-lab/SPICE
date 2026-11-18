"""Text progress bar for parallel jobs, written to stderr.

`ProgressBar` redraws one line each time a job finishes, e.g.
`Progress: |███---------------------------| 10.0% Complete [[2026-Oct-19 12:00:00]Pair: 3]`, and ends it with a
newline once all `total` jobs are done.
"""

import sys
import datetime


class ProgressBar:
    """Text progress bar on stderr, advanced once per finished job.

    Parameters
    ----------
    total : int
        Number of jobs. The bar ends with a newline once the counter reaches it.
    length : int
        Width of the bar in characters.
    counter : int, default 0
        Number of jobs already done.
    verbose : bool, default False
        If True, each message is printed on its own line above the bar; otherwise it is shown after the bar, on the
        same line.
    decimals : int, default 1
        Number of decimals in the percentage.
    fill : str, default chr(9608)
        Character for the done part of the bar (█); the rest is '-'.
    prefix : str, default 'Progress:'
        Text before the bar.
    suffix : str, default 'Complete'
        Text after the percentage.
    """

    def __init__(self, total: int, length: int, counter: int = 0, verbose: bool = False, decimals: int = 1,
                 fill: str = chr(9608), prefix: str = 'Progress:', suffix: str = 'Complete') -> None:
        self.total = total
        self.length = length
        self.decimals = decimals
        self.fill = fill
        self.prefix = prefix
        self.suffix = suffix
        self.counter = counter
        self.verbose = verbose

    def progress(self, advance: bool = True, msg: str = "") -> bool:
        """Redraw the bar with `msg` (one job further if `advance`); always returns True, for comprehension filters."""
        self.progress_unlocked(advance, msg)
        return True

    def progress_unlocked(self, advance: bool, msg: str) -> None:
        """Redraw the bar on stderr with a timestamped `msg`, first adding one to the counter if `advance`."""
        flush = sys.stderr.flush
        write = sys.stderr.write
        if advance:
            self.counter += 1
        v = 0 if float(self.total) == 0 else (self.counter / float(self.total))
        percent = ("{0:." + str(self.decimals) + "f}").format(100 * v)
        filledLength = 0 if self.total == 0 else int(self.length * self.counter // self.total)
        bar = self.fill * filledLength + '-' * (self.length - filledLength)
        rewind = '\x1b[2K\r'
        result = '%s |%s| %s%% %s' % (self.prefix, bar, percent, self.suffix)
        msg = '[{:%Y-%b-%d %H:%M:%S}]'.format(datetime.datetime.now()) + msg
        if not self.verbose:
            toprint = rewind + result + " [%s]" % msg
        else:
            toprint = rewind + msg + "\n" + result
        write(toprint)
        flush()
        if self.counter == self.total:
            write("\n")
            flush()
