from __future__ import division, print_function

import numpy as np
import loopy as lp
import pyopencl as cl
import pyopencl.array
import pyopencl.clrandom
from loopy.statistics import get_op_poly, get_DRAM_access_poly, get_barrier_poly
import sys
sys.path.append("../performance_model")
from perf_model import GPUStats, KernelStats, ThreadConfig, PerfModel
import islpy as isl
import math

# setup
# -----
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

predicted_times = []
actual_times = []


trials_n = 3
nvals = [2**(11+x) for x in range(trials_n)]
configs_t = [(8, 8), (16, 16), (32, 32)]

for n in nvals:
    a_mat_dev = cl.clrandom.rand(queue, (n, n), dtype=np.float32)
    b_mat_dev = cl.clrandom.rand(queue, (n, n), dtype=np.float32)
    order = "C"
    dtype = np.float32
    knl = lp.make_kernel(
            "{[i,j]: 0<=i,j<%d}" % n,
            [
                "b[i, j] = a[j, i]"
                ],
            [
                lp.GlobalArg("a", dtype, shape=(n, n), order=order),
                lp.GlobalArg("b", dtype, shape=(n, n), order=order),
                ],
            name="transpose")

    ref_knl = knl

    for BSIZEx, BSIZEy in configs_t:

        knl = ref_knl
        knl = lp.split_iname(knl, "i", BSIZEy, outer_tag="g.0", inner_tag="l.1")
        knl = lp.split_iname(knl, "j", BSIZEx, outer_tag="g.1", inner_tag="l.0")
        knl = lp.add_prefetch(knl, 'a', ["i_inner", "j_inner"])

        #check = lp.auto_test_vs_ref(ref_knl, ctx, knl, print_code=True)
        #print "Correctness check: \n", check

        # use ptx src to determine resource usage
        cknl = lp.compiled.CompiledKernel(ctx, knl)
        ptx_src = cknl.cl_kernel_info().cl_kernel.program.binaries[0]
        ptx_src_file = open(knl.name+".ptx", 'w')
        ptx_src_file.write(ptx_src)

        barrier_poly = get_barrier_poly(knl)
        barrier_count = barrier_poly.eval_with_dict({'n': n})
        op_map = get_op_poly(knl)
        flops = op_map.dict.get(
                            np.dtype(np.float32),isl.PwQPolynomial('{ 0 }')
                            ).eval_with_dict({'n': n})
        iops = op_map.dict.get(
                            np.dtype(np.int32),isl.PwQPolynomial('{ 0 }')
                            ).eval_with_dict({'n': n})
        sub_map = get_DRAM_access_poly(knl)  # noqa

        f32coal_l = sub_map.dict.get(
                            (np.dtype(np.float32), 'consecutive', 'load'),
                            isl.PwQPolynomial('{ 0 }')
                            ).eval_with_dict({'n': n})
        f32coal_s = sub_map.dict.get(
                            (np.dtype(np.float32), 'consecutive', 'store'),
                            isl.PwQPolynomial('{ 0 }')
                            ).eval_with_dict({'n': n})
        f32coal = f32coal_l + f32coal_s
        #print "coalesced: %i, (stores: %i, loads: %i)" % (f32coal, f32coal_s, f32coal_l)
        f32uncoal_l = sub_map.dict.get(
                            (np.dtype(np.float32), 'nonconsecutive', 'load'),
                            isl.PwQPolynomial('{ 0 }')
                            ).eval_with_dict({'n': n})
        f32uncoal_s = sub_map.dict.get(
                            (np.dtype(np.float32), 'nonconsecutive', 'store'),
                            isl.PwQPolynomial('{ 0 }')
                            ).eval_with_dict({'n': n})
        f32uncoal = f32uncoal_l + f32uncoal_s

        # execute
        # -------
        #print "="*40+"TIMING RESULTS"
        print("running kernel...")
        #knl = lp.set_options(knl, write_cl=True, highlight_cl=True)
        evt, (out,) = knl(queue, a=a_mat_dev, b=b_mat_dev)
        evt.wait()

        gstats = GPUStats('TeslaK20')
        reg32_per_thread = 8
        shared_mem_per_block = 4*BSIZEx*BSIZEy
        total_blocks = math.ceil(n/BSIZEx)*math.ceil(n/BSIZEy)
        total_threads = total_blocks*BSIZEx*BSIZEy
        kstats = KernelStats(flops/(n*n), f32uncoal/(n*n), f32coal/(n*n),
                             barrier_count, reg32_per_thread, shared_mem_per_block)
        tconfig = ThreadConfig(BSIZEx*BSIZEy, total_blocks)

        model = PerfModel(gstats, kstats, tconfig,
                        np.dtype(np.float32))  #, active_blocks=active_blks)
        cycles = model.compute_total_cycles()

        actual_times.append((evt.profile.END - evt.profile.START)*1e-9)
        predicted_times.append(cycles/(gstats.sm_clock_freq*10**9))

print("="*40+"TIMING RESULTS")
print("n\tBx\tBy\tactual\t\tpredicted\terror")
rel_error = []
for i in range(trials_n):
    rel_error.append([])
    for j in range(len(configs_t)):
        predicted = predicted_times[i*len(configs_t)+j]
        actual = actual_times[i*len(configs_t)+j]
        print("%i\t%i\t%i\t%f\t%f\t%f" % (nvals[i], configs_t[j][0], configs_t[j][1],
                            actual, predicted, (predicted-actual)/actual))
        rel_error[i].append((predicted-actual)/actual)

print("\n\t", end='')
for config in configs_t:
    print("(%i,%i)\t\t" % (config[0],config[1]), end='')
print("")
for i in range(trials_n):
    print("%i\t" % (nvals[i]), end='')
    for j in range(len(configs_t)):
        print("%f\t" % (rel_error[i][j]), end='')
    print("")

