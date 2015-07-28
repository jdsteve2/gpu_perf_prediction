import sys
sys.path.append("../performance_model")
from perf_model import GPUStats, KernelStats, ThreadConfig, PerfModel
#from future import division
import numpy as np
from pyopencl.tools import (  # noqa
        pytest_generate_tests_for_pyopencl
        as pytest_generate_tests)

TOLERANCE = 0.001

def test_HK_example():
    gstats = GPUStats('HKexample')
    kstats = KernelStats(27, 6, 0, 6)
    tconfig = ThreadConfig(128, 80)
    model = PerfModel(gstats, kstats, tconfig, np.dtype(np.float32), active_blocks=5)
    expected = 50738
    assert (abs(model.compute_exec_cycles() - expected) / expected) < TOLERANCE

def test_HK_sepia():
    # input size: 7000x7000
    n = 7000
    threads = 256
    gstats = GPUStats('FX5600')
    kstats = KernelStats(71, 6, 0, 0) #TODO synch_insns=0 ?
    tconfig = ThreadConfig(threads, n*n/threads)
    model = PerfModel(gstats, kstats, tconfig, np.dtype(np.float32), active_blocks=5)
    time_ms = model.compute_exec_cycles()/(gstats.sm_clock_freq*(10**9))*(10**3)
    print "time: ", time_ms
    expected = 158
    assert (abs(model.compute_exec_cycles() - expected) / expected) < TOLERANCE


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from py.test.cmdline import main
        main([__file__])



