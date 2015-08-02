import sys
sys.path.append("../performance_model")
from perf_model import GPUStats, KernelStats, ThreadConfig, PerfModel
#from future import division
import numpy as np
import matplotlib.pyplot as plt
'''
from pyopencl.tools import (  # noqa
        pytest_generate_tests_for_pyopencl
        as pytest_generate_tests)
'''
TOLERANCE = 0.001

def test_HK_example():
    gstats = GPUStats('HKexample')
    kstats = KernelStats(27, 6, 0, 6)
    tconfig = ThreadConfig(128, 80)
    model = PerfModel(gstats, kstats, tconfig, np.dtype(np.float32), active_blocks=5)
    expected = 50738
    print "total cycles: ", model.compute_total_cycles(), "expected: ~", expected

    expect_cwp = 20
    expect_mwp = 2.28

    print "CWP: ", model.CWP, "expected: ~", expect_cwp
    print "MWP: ", model.MWP, "expected: ~", expect_mwp
    assert (abs(model.compute_total_cycles() - expected) / expected) < TOLERANCE


def test_HK_sepia():

    print "TESTING sepia..."

    # input size: 7000x7000
    n = 7000
    gstats = GPUStats('FX5600')
    kstats = KernelStats(71, 6, 0, 0)  # TODO synch_insns=0 ?
    expected = 153

    trials = 17
    threads = [(x+6)*(x+6) for x in range(trials)]
    active_blocks = [8, 8, 8, 8, 6, 6, 4, 4, 3, 3, 3, 2, 2, 2, 1, 1, 1]
    #active_blocks = [8, 8, 8, 8, 8, 6, 6, 4, 4, 3, 3, 3, 2, 2, 1, 1, 1]
    times = []
    occupancies = []
    CPIs = []
    CWPs = []
    MWPs = []
    for i in range(trials):
        tconfig = ThreadConfig(threads[i], n*n/threads[i])
        model = PerfModel(gstats, kstats, tconfig,
                        np.dtype(np.float32), active_blocks=active_blocks[i])
        times.append(model.compute_total_cycles() /
                    (gstats.sm_clock_freq*(10**9))*(10**3))
        occupancies.append(model.occ)
        CPIs.append(model.CPI)
        CWPs.append(model.CWP)
        MWPs.append(model.MWP)
        print threads[i], active_blocks[i], times[i]

    expect_avg_occ = 0.835
    expect_avg_cpi = 26
    expect_avg_cwp = 14
    expect_avg_mwp = 2

    print "avg occ: ", np.average(occupancies), "expected: ~", expect_avg_occ
    print "avg CPI: ", np.average(CPIs), "expected: ~", expect_avg_cpi
    print "avg CWP: ", np.average(CWPs), "expected: ~", expect_avg_cwp
    print "avg MWP: ", np.average(MWPs), "expected: ~", expect_avg_mwp

    '''
    plt.figure("sepia")
    plt.title("sepia")
    plt.plot(threads, times, 'b*')
    axes = plt.gca()
    axes.set_ylim([0, 330])
    plt.show()
    '''
    #assert 1 == 0


def test_HK_linear():

    print "TESTING linear..."

    # input size: 10000x10000
    n = 10000
    gstats = GPUStats('FX5600')
    kstats = KernelStats(111, 30, 0, 0)  # TODO synch_insns=0 ?
    expected = 775

    trials = 9
    threads = [(x*2+6)*(x*2+6) for x in range(trials)]
    active_blocks = [8, 8, 4, 2, 2, 2, 1, 1, 1]
    #active_blocks = [8, 8, 4, 4, 2, 2, 2, 1, 1]
    times = []
    occupancies = []
    CPIs = []
    CWPs = []
    MWPs = []
    for i in range(trials):
        tconfig = ThreadConfig(threads[i], n*n/threads[i])
        model = PerfModel(gstats, kstats, tconfig,
                        np.dtype(np.float32), active_blocks=active_blocks[i])
        times.append(model.compute_total_cycles() /
                    (gstats.sm_clock_freq*(10**9))*(10**3))
        occupancies.append(model.occ)
        CPIs.append(model.CPI)
        CWPs.append(model.CWP)
        MWPs.append(model.MWP)
        print threads[i], active_blocks[i], times[i]

    expect_avg_occ = 0.59
    expect_avg_cpi = 73
    expect_avg_cwp = 14
    expect_avg_mwp = 2

    print "avg occ: ", np.average(occupancies), "expected: ~", expect_avg_occ
    print "avg CPI: ", np.average(CPIs), "expected: ~", expect_avg_cpi
    print "avg CWP: ", np.average(CWPs), "expected: ~", expect_avg_cwp
    print "avg MWP: ", np.average(MWPs), "expected: ~", expect_avg_mwp
    '''
    plt.figure("linear")
    plt.title("linear")
    plt.plot(threads, times, 'b*')
    axes = plt.gca()
    axes.set_ylim([0, 1600])
    plt.show()
    '''
    #assert 1 == 0
    #assert (abs(model.compute_total_cycles() - expected) / expected) < TOLERANCE


def test_HK_blackscholes():

    print "TESTING blackscholes..."

    # input size: 9000000
    n = 9000000
    gstats = GPUStats('FX5600')
    kstats = KernelStats(137, 7, 0, 0)  # TODO synch_insns=0 ?
    expected = 34

    trials = 9
    threads = [(x*2+6)*(x*2+6) for x in range(trials)]
    active_blocks = [8, 8, 5, 3, 2, 2, 1, 1, 1]
    #active_blocks = [8, 8, 8, 5, 3, 3, 2, 1, 1]
    times = []
    occupancies = []
    CPIs = []
    CWPs = []
    MWPs = []
    for i in range(trials):
        tconfig = ThreadConfig(threads[i], n/threads[i])
        model = PerfModel(gstats, kstats, tconfig,
                        np.dtype(np.float32), active_blocks=active_blocks[i])
        times.append(model.compute_total_cycles() /
                    (gstats.sm_clock_freq*(10**9))*(10**3))
        occupancies.append(model.occ)
        CPIs.append(model.CPI)
        CWPs.append(model.CWP)
        MWPs.append(model.MWP)
        print threads[i], active_blocks[i], times[i]

    expect_avg_occ = 0.745
    expect_avg_cpi = 16
    expect_avg_cwp = 9
    expect_avg_mwp = 2

    print "avg occ: ", np.average(occupancies), "expected: ~", expect_avg_occ
    print "avg CPI: ", np.average(CPIs), "expected: ~", expect_avg_cpi
    print "avg CWP: ", np.average(CWPs), "expected: ~", expect_avg_cwp
    print "avg MWP: ", np.average(MWPs), "expected: ~", expect_avg_mwp
    '''
    plt.figure("blackscholes")
    plt.title("blackscholes")
    plt.plot(threads, times, 'b*')
    axes = plt.gca()
    axes.set_ylim([0, 78])
    plt.show()
    '''
    #assert 1 == 0
    #assert (abs(model.compute_total_cycles() - expected) / expected) < TOLERANCE


def test_HK_SVM():

    print "TESTING SVM..."

    # input size: 736*992
    n1 = 736
    n2 = 992
    gstats = GPUStats('FX5600')
    kstats = KernelStats(10871, 0, 819, 0)  # TODO synch_insns=0 ?
    expected = 14

    trials = 17
    threads = [(x+6)*(x+6) for x in range(trials)]
    active_blocks = [8, 8, 8, 6, 6, 6, 4, 4, 3, 3, 3, 2, 2, 2, 1, 1, 1]
    #active_blocks = [8, 8, 8, 8, 6, 6, 6, 4, 4, 3, 3, 3, 2, 2, 2, 1, 1]
    times = []
    occupancies = []
    CPIs = []
    CWPs = []
    MWPs = []
    for i in range(trials):
        tconfig = ThreadConfig(threads[i], (n1*n2)/threads[i])
        model = PerfModel(gstats, kstats, tconfig,
                        np.dtype(np.float32), active_blocks=active_blocks[i])
        times.append(model.compute_total_cycles() /
                    (gstats.sm_clock_freq*(10**9))*(10**3))
        occupancies.append(model.occ)
        CPIs.append(model.CPI)
        CWPs.append(model.CWP)
        MWPs.append(model.MWP)
        print threads[i], active_blocks[i], times[i]

    expect_avg_occ = 0.84
    expect_avg_cpi = 3.5
    expect_avg_cwp = 9
    expect_avg_mwp = 12

    print "avg occ: ", np.average(occupancies), "expected: ~", expect_avg_occ
    print "avg CPI: ", np.average(CPIs), "expected: ~", expect_avg_cpi
    print "avg CWP: ", np.average(CWPs), "expected: ~", expect_avg_cwp
    print "avg MWP: ", np.average(MWPs), "expected: ~", expect_avg_mwp
    '''
    plt.figure("svm")
    plt.title("svm")
    plt.plot(threads, times, 'b*')
    axes = plt.gca()
    axes.set_ylim([0, 60])
    plt.show()
    '''
    #assert 1 == 0
    #assert (abs(model.compute_total_cycles() - expected) / expected) < TOLERANCE

test_HK_example()
test_HK_sepia()
test_HK_linear()
test_HK_blackscholes()
test_HK_SVM()
'''
if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from py.test.cmdline import main
        main([__file__])
'''

