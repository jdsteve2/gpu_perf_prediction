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

"""

A = [[1, 2, 3, 4],
     [1.5, 4, 2.1, 6.5],
     [2, 2, 2, 2],
     [0, 9, 10.2, 7],
     [2.5, 2.5, 6.1, 6]]
init = [0, 0, 0, 0]
a = [1, 2, 3, 4]
b = [1.5, 4, 2.1, 6.5]
c = [2, 2, 2, 2]
d = [0, 9, 10.2, 7]
e = [2.5, 2.5, 6.1, 6]

A = np.vstack((init,a,b,c,d,e))

A = []
A.append(a)
A.append(b)
A.append(c)
A.append(d)
A.append(e)

y = [10,14,8,26.2,17]

print(A)
print(y)

(result,resid,q,q) = np.linalg.lstsq(A,y)
print(result)

"""

# setup
# -----
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

lstsq_A = []
lstsq_y = []
predicted_times = []
actual_times = []
trials_n = 4
nvals = [2**(10+x) for x in range(trials_n)]
configs_t = [(8, 8, 8), (16, 8, 8), (16, 16, 8), (32, 16, 4), (32, 32, 2)]
#n = 2**10
for n in nvals:
    a_mat_dev = cl.clrandom.rand(queue, (n, n), dtype=np.float32)
    b_mat_dev = cl.clrandom.rand(queue, (n, n), dtype=np.float32)
    c_mat_dev = cl.clrandom.rand(queue, (n, n), dtype=np.float32)

    order = "C"
    knl = lp.make_kernel(
        "{[i,j,k]: 0<=i,j,k<%d}" % n,
        [
            "c[i, j] = sum(k, a[i, k]*b[k, j])"
            ],
        [
            lp.GlobalArg("a", np.float32, shape=(n, n), order=order),
            lp.GlobalArg("b", np.float32, shape=(n, n), order=order),
            lp.GlobalArg("c", np.float32, shape=(n, n), order=order),
            ],
        name="matmul")

    ref_knl = knl

    for BSIZEx, BSIZEy, active_blks in configs_t:

        knl = ref_knl
        knl = lp.split_iname(knl, "i", BSIZEy, outer_tag="g.0", inner_tag="l.1")
        knl = lp.split_iname(knl, "j", BSIZEx, outer_tag="g.1", inner_tag="l.0")
        knl = lp.split_iname(knl, "k", BSIZEy)
        knl = lp.add_prefetch(knl, "a", ["k_inner", "i_inner"])
        knl = lp.add_prefetch(knl, "b", ["j_inner", "k_inner", ])

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
        flops = op_map.dict[np.dtype(np.float32)].eval_with_dict({'n': n})
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

        '''
        print "="*40+"PTX SOURCE"
        print "PTX source written to "+knl.name+".ptx"
        print "To determine resource usage from PTX source, do:"
        print "ptxas -v --gpu-name <compute capability> <filename.ptx>"
        print "For example, with compute capability 3.5, do:"
        print "ptxas -v --gpu-name sm_35 "+knl.name+".ptx"
        print "="*40

        print "="*40+"DEVICES"
        print ctx.get_info(cl.context_info.DEVICES)
        print "="*40

        print "="*40+"KERNEL STATS"
        print "barrier count: ", barrier_count
        print "flops: ", flops
        print(sub_map)
        print "="*40
        '''

        # execute
        # -------
        #print "="*40+"TIMING RESULTS"
        print("running kernel...")
        #knl = lp.set_options(knl, write_cl=True, highlight_cl=True)
        evt, (out,) = knl(queue, a=a_mat_dev, b=b_mat_dev, c=c_mat_dev)
        evt.wait()

        gstats = GPUStats('TeslaK20')
        total_blocks = math.ceil(n/BSIZEx)*math.ceil(n/BSIZEy)
        total_threads = total_blocks*BSIZEx*BSIZEy
        kstats = KernelStats(flops/(n*n), f32uncoal/(n*n),
                             f32coal/(n*n), barrier_count)
        tconfig = ThreadConfig(BSIZEx*BSIZEy, total_blocks)

        model = PerfModel(gstats, kstats, tconfig,
                        np.dtype(np.float32), active_blocks=active_blks)
        cycles = model.compute_total_cycles()

        '''
        print "actual runtime: ", (evt.profile.END - evt.profile.START)*1e-9
        print "total predicted time: ", cycles/(gstats.sm_clock_freq*10**9)
        print "total predicted execution cycles: ", cycles
        print "="*40
        '''
        actual_times.append((evt.profile.END - evt.profile.START)*1e-9)
        predicted_times.append(cycles/(gstats.sm_clock_freq*10**9))

        lstsq_A.append([n*n, total_blocks, 1.0/total_threads, BSIZEx*BSIZEy,
                        1.0/active_blks, np.dtype(np.float32).itemsize, flops/(n*n),
                        f32uncoal/(n*n), f32coal/(n*n), barrier_count, 1.0])
        lstsq_y.append(actual_times[-1])

result_lstsq, resid, q, q = np.linalg.lstsq(lstsq_A,lstsq_y)
U, s, V = np.linalg.svd(lstsq_A, full_matrices=False)
print("Least Squares Residual:\n", np.dot(lstsq_A,result_lstsq)-lstsq_y)
print("Least Squares singular values:\n", s)


print("="*40+"TIMING RESULTS")
print("n\tBx\tBy\tactual\t\tpredicted\terror\t\tlstsq\t\terror")
rel_error = []
for i in range(trials_n):
    rel_error.append([])
    for j in range(len(configs_t)):
        predicted = predicted_times[i*len(configs_t)+j]
        actual = actual_times[i*len(configs_t)+j]
        rel_error[i].append((predicted-actual)/actual)
        # least squares
        predicted_lstsq = np.dot(lstsq_A[i*len(configs_t)+j],result_lstsq)
        print("%i\t%i\t%i\t%f\t%f\t%f\t%f\t%f" % 
                            (nvals[i], configs_t[j][0], configs_t[j][1],
                            actual, predicted, rel_error[i][-1], predicted_lstsq,
                            (predicted_lstsq-actual)/actual))
        

print("\n\t", end='')
for config in configs_t:
    print("(%i,%i)\t\t" % (config[0],config[1]), end='')
print("")
for i in range(trials_n):
    print("%i\t" % (nvals[i]), end='')
    for j in range(len(configs_t)):
        print("%f\t" % (rel_error[i][j]), end='')
    print("")
"""
print("="*40)
print("A: ")
for row in lstsq_A:
    print(row)
print("y: ")
print(lstsq_y)
(result,resid,q,q) = np.linalg.lstsq(lstsq_A,lstsq_y)
print("result: ")
print(result)
print("resid: ")
print(resid)

print("="*40)
for i in range(len(lstsq_y)):
    print(np.dot(lstsq_A[i],result), lstsq_y[i])
"""
