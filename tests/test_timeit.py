import re
import time

from eqm_decompositional_graph_generator.timeit import _verbosity_level, timeit


class _Worker:
    def __init__(self, verbose):
        self.verbose = verbose

    @timeit
    def compute(self, x):
        time.sleep(0.001)
        return x + 1


def test_verbosity_level_handles_bool_and_int_and_missing():
    class _NoVerbose:
        pass

    assert _verbosity_level(_NoVerbose()) == 0
    assert _verbosity_level(_Worker(verbose=False)) == 0
    assert _verbosity_level(_Worker(verbose=True)) == 1
    assert _verbosity_level(_Worker(verbose=2)) == 2
    assert _verbosity_level(_Worker(verbose="3")) == 3


def test_timeit_prints_only_at_verbose_level_3(capfd):
    quiet = _Worker(verbose=2)
    assert quiet.compute(4) == 5
    quiet_out = capfd.readouterr().out
    assert quiet_out == ""

    loud = _Worker(verbose=3)
    assert loud.compute(9) == 10
    loud_out = capfd.readouterr().out
    assert "Function 'compute' executed in" in loud_out
    assert re.search(r"Class '_Worker', Function 'compute' executed in", loud_out)
