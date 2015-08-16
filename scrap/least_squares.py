from __future__ import division, print_function

import numpy as np
import loopy as lp
import pyopencl as cl
import pyopencl.array
import pyopencl.clrandom  # noqa
from loopy.statistics import get_op_poly, get_DRAM_access_poly, get_barrier_poly
import sys
sys.path.append("../performance_model")
from perf_model import GPUStats, KernelStats, ThreadConfig, PerfModel
import islpy as isl
import math

run_matmul = True
run_axpy = True
run_transpose = True


# setup
# -----
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

lstsq_A = []
lstsq_y = []
predicted_times = []
actual_times = []

if run_matmul:

    #trials_n = 4
    trials_n = 4
    nvals = [2**(9+x) for x in range(trials_n)]
    #configs_t = [(8, 8), (16, 8), (16, 16), (32, 16), (32, 32)]
    configs_t = [(16, 8), (16, 16), (32, 16)]
    #TODO figure out smem usage issue
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

        #for BSIZEx, BSIZEy, active_blks in configs_t:
        for BSIZEx, BSIZEy in configs_t:

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
            reg32_per_thread = 25
            shared_mem_per_block = 2*4*BSIZEx*BSIZEy
            total_blocks = math.ceil(n/BSIZEx)*math.ceil(n/BSIZEy)
            total_threads = total_blocks*BSIZEx*BSIZEy
            kstats = KernelStats(flops/(n*n), f32uncoal/(n*n), f32coal/(n*n),
                                 barrier_count, reg32_per_thread, shared_mem_per_block)
            tconfig = ThreadConfig(BSIZEx*BSIZEy, total_blocks)

            model = PerfModel(gstats, kstats, tconfig,
                            np.dtype(np.float32))  #, active_blocks=active_blks)
            cycles = model.compute_total_cycles()

            '''
            print "actual runtime: ", (evt.profile.END - evt.profile.START)*1e-9
            print "total predicted time: ", cycles/(gstats.sm_clock_freq*10**9)
            print "total predicted execution cycles: ", cycles
            print "="*40
            '''
            actual_times.append((evt.profile.END - evt.profile.START)*1e-9)
            predicted_times.append(cycles/(gstats.sm_clock_freq*10**9))

            lstsq_A.append([n*n,
                            total_blocks,
                            BSIZEx*BSIZEy,
                            1.0/model.active_blocks_per_SM,
                            np.dtype(np.float32).itemsize,
                            flops/(n*n),
                            f32uncoal/(n*n),
                            f32coal/(n*n),
                            barrier_count,
                            reg32_per_thread,
                            shared_mem_per_block,
                            1.0])
            lstsq_y.append(actual_times[-1])

# now train on axpy
if run_axpy:
    #trials_n = 4
    trials_n = 4
    nvals = [2**(24+x) for x in range(trials_n)]
    #configs_t = [(8, 8), (16, 8), (16, 16), (32, 16), (32, 32)]
    configs_t = [(16, 1), (32, 1), (64, 1), (128, 1), (256, 1), (512, 1)]
    #TODO figure out smem usage issue
    #n = 2**10
    for n in nvals:
        x_vec_dev = cl.clrandom.rand(queue, n, dtype=np.float32)
        y_vec_dev = cl.clrandom.rand(queue, n, dtype=np.float32)
        z_vec_dev = cl.clrandom.rand(queue, n, dtype=np.float32)

        dtype = np.float32
        knl = lp.make_kernel(
            "[n] -> {[i]: 0<=i<%d}" % n,
            [
                "z[i] = 5.0*x[i]+7.0*y[i]"
                ],
            [
                lp.GlobalArg("x", dtype, shape=n),
                lp.GlobalArg("y", dtype, shape=n),
                lp.GlobalArg("z", dtype, shape=n),
                ],
            name="axpy")

        ref_knl = knl

        for BSIZEx, BSIZEy in configs_t:

            knl = ref_knl
            unroll = 4
            knl = lp.split_iname(knl, "i", unroll*BSIZEx,
                 outer_tag="g.0", slabs=(0, 1))
            knl = lp.split_iname(knl, "i_inner", BSIZEx,
                 outer_tag="unr", inner_tag="l.0")

            #check = lp.auto_test_vs_ref(ref_knl, ctx, knl, print_code=False)
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
            evt, (out,) = knl(queue, x=x_vec_dev, y=y_vec_dev, z=z_vec_dev)
            evt.wait()

            gstats = GPUStats('TeslaK20')
            reg32_per_thread = 20
            shared_mem_per_block = 0
            total_blocks = math.ceil(n/(BSIZEx*unroll))
            kstats = KernelStats(flops*unroll/n, f32uncoal*unroll/n, f32coal*unroll/n,
                                 barrier_count, reg32_per_thread, shared_mem_per_block)
            tconfig = ThreadConfig(BSIZEx*BSIZEy, total_blocks)
            model = PerfModel(gstats, kstats, tconfig, np.dtype(np.float32))

            cycles = model.compute_total_cycles()

            actual_times.append((evt.profile.END - evt.profile.START)*1e-9)
            predicted_times.append(cycles/(gstats.sm_clock_freq*10**9))

            lstsq_A.append([n,
                            total_blocks,
                            BSIZEx*BSIZEy,
                            1.0/model.active_blocks_per_SM,
                            np.dtype(np.float32).itemsize,
                            flops*unroll/n,
                            f32uncoal*unroll/n,
                            f32coal*unroll/n,
                            barrier_count,
                            reg32_per_thread,
                            shared_mem_per_block,
                            1.0])
            # TODO try adding other items like regs per thread, shared mem, etc
            lstsq_y.append(actual_times[-1])

if run_transpose:
    trials_n = 4
    nvals = [2**(10+x) for x in range(trials_n)]
    configs_t = [(8, 8), (16, 16), (24, 24), (32, 32)]

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

            lstsq_A.append([n*n,
                            total_blocks,
                            BSIZEx*BSIZEy,
                            1.0/model.active_blocks_per_SM,
                            np.dtype(np.float32).itemsize,
                            flops/(n*n),
                            f32uncoal/(n*n),
                            f32coal/(n*n),
                            barrier_count,
                            reg32_per_thread,
                            shared_mem_per_block,
                            1.0])
            lstsq_y.append(actual_times[-1])

# least squares
result_lstsq, resid, q, q = np.linalg.lstsq(lstsq_A, lstsq_y)
U, s, V = np.linalg.svd(lstsq_A, full_matrices=False)
print("Least Squares Residual:\n", np.dot(lstsq_A, result_lstsq)-lstsq_y)
print("Least Squares singular values:\n", s)

print("="*40+"TIMING RESULTS")
print("i\tactual\t\tpredicted\terror\t\tlstsq\t\terror")
rel_error = []
rel_error_lstsq = []
for i in range(len(actual_times)):
    predicted = predicted_times[i]
    predicted_lstsq = np.dot(lstsq_A[i], result_lstsq)
    actual = actual_times[i]
    rel_error.append((predicted-actual)/actual)
    rel_error_lstsq.append((predicted_lstsq-actual)/actual)
    print("%i\t%f\t%f\t%f\t%f\t%f" % (i, actual, predicted, rel_error[i],
                                      predicted_lstsq, rel_error_lstsq[i]))

print("avg relative error HK: ", np.average(rel_error)) 
print("avg relative error LS: ", np.average(rel_error_lstsq))
"""
print("="*40+"TIMING RESULTS")
print("n\tBx\tBy\tactual\t\tpredicted\terror\t\tlstsq\t\terror")
rel_error = []
rel_error_lstsq = []
for i in range(len(actual_times)):
    predicted = predicted_times[i]
    predicted_lstsq = np.dot(lstsq_A[i], result_lstsq)
    actual = actual_times[i]
    rel_error.append((predicted-actual)/actual)
    rel_error_lstsq.append((predicted_lstsq-actual)/actual)
    print("%i\t%i\t%i\t%f\t%f\t%f\t%f\t%f" %
                        (nvals[int(i/len(configs_t))], configs_t[i%len(configs_t)][0], 
                         configs_t[i%len(configs_t)][1], actual, predicted,
                         rel_error[i], predicted_lstsq, rel_error_lstsq[i]))
"""



"""
print("="*40+"TIMING RESULTS")
print("n\tBx\tBy\tactual\t\tpredicted\terror\t\tlstsq\t\terror")
rel_error = []
rel_error_lstsq = []
for i in range(trials_n):
    rel_error.append([])
    rel_error_lstsq.append([])
    for j in range(len(configs_t)):
        predicted = predicted_times[i*len(configs_t)+j]
        predicted_lstsq = np.dot(lstsq_A[i*len(configs_t)+j], result_lstsq)
        actual = actual_times[i*len(configs_t)+j]
        rel_error[i].append((predicted-actual)/actual)
        rel_error_lstsq[i].append((predicted_lstsq-actual)/actual)
        print("%i\t%i\t%i\t%f\t%f\t%f\t%f\t%f" %
                            (nvals[i], configs_t[j][0], configs_t[j][1],
                            actual, predicted, rel_error[i][-1], predicted_lstsq,
                            rel_error_lstsq[i][-1]))

print("\nHong Kim relative error:")
print("\t", end='')
for config in configs_t:
    print("(%i,%i)\t\t" % (config[0], config[1]), end='')
print("")
for i in range(trials_n):
    print("%i\t" % (nvals[i]), end='')
    for j in range(len(configs_t)):
        print("%f\t" % (rel_error[i][j]), end='')
    print("")

print("\nLeast squares relative error:")
print("\t", end='')
for config in configs_t:
    print("(%i,%i)\t\t" % (config[0], config[1]), end='')
print("")
for i in range(trials_n):
    print("%i\t" % (nvals[i]), end='')
    for j in range(len(configs_t)):
        print("%f\t" % (rel_error_lstsq[i][j]), end='')
    print("")
"""

