from __future__ import division, print_function

import numpy as np
import islpy as isl
import pyopencl as cl
import copy
import loopy as lp


def unit_vector(vector):
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    else:
        return vector / np.linalg.norm(vector)


def cos_angle_btw(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.dot(v1_u, v2_u)


def get_cos_angles_bt_rows(A):
    cos_angles = []
    for row1 in range(len(A)):
        for j in range(len(A)-1-row1):
            row2 = row1+1+j
            cos_angles.append(cos_angle_btw(A[row1], A[row2]))
    return cos_angles


def print_ptx_src_msg(knl_name):
    print("="*40+"PTX SOURCE")
    print("PTX source written to "+knl_name+".ptx")
    print("To determine resource usage from PTX source, do:")
    print("ptxas -v --gpu-name <compute capability> <filename.ptx>")
    print("For example, with compute capability 3.5, do:")
    print("ptxas -v --gpu-name sm_35 "+knl_name+".ptx")
    print("="*40)


def print_device_info(ctx):
    print("="*40+"DEVICES")
    print(ctx.get_info(cl.context_info.DEVICES))
    print("="*40)


def print_Ay(A, y):
    for row in range(len(A)):
        print("%d\t" % (row), end='')
        for col in range(len(A[0])):
            if col < 2:
                print("%e\t" % (A[row][col]), end='')
            else:
                print("%e\t" % (A[row][col]), end='')
        print("| %f" % (y[row]))


def get_DRAM_f32_accesses(sub_map, param_dict):
    f32coal_l = sub_map.get(
                        (np.dtype(np.float32), 'consecutive', 'load'),
                        isl.PwQPolynomial('{ 0 }')
                        ).eval_with_dict(param_dict)
    f32coal_s = sub_map.get(
                        (np.dtype(np.float32), 'consecutive', 'store'),
                        isl.PwQPolynomial('{ 0 }')
                        ).eval_with_dict(param_dict)
    f32uncoal_l = sub_map.get(
                        (np.dtype(np.float32), 'nonconsecutive', 'load'),
                        isl.PwQPolynomial('{ 0 }')
                        ).eval_with_dict(param_dict)
    f32uncoal_s = sub_map.get(
                        (np.dtype(np.float32), 'nonconsecutive', 'store'),
                        isl.PwQPolynomial('{ 0 }')
                        ).eval_with_dict(param_dict)
    return (f32coal_l, f32coal_s, f32uncoal_l, f32uncoal_s)


def get_32b_ops(op_map, param_dict):
    flops = op_map.get(
                       np.dtype(np.float32), isl.PwQPolynomial('{ 0 }')
                       ).eval_with_dict(param_dict)
    iops = op_map.get(
                      np.dtype(np.int32), isl.PwQPolynomial('{ 0 }')
                      ).eval_with_dict(param_dict)
    return (flops, iops)


def get_32b_ops_all(op_map, param_dict):
    typef32 = np.dtype(np.float32)
    typei32 = np.dtype(np.int32)
    total = 0
    for (dtype, optype) in op_map:
        if dtype == typef32 or dtype == typei32:
            total += op_map[(dtype, optype)].eval_with_dict(param_dict)
    return total


def get_32b_flops_all(op_map, param_dict):
    typef32 = np.dtype(np.float32)
    total = 0
    for (dtype, optype) in op_map:
        if dtype == typef32:
            total += op_map[(dtype, optype)].eval_with_dict(param_dict)
    return total


def get_32b_amd_ops(op_map, param_dict):
    typef32 = np.dtype(np.float32)
    typei32 = np.dtype(np.int32)
    zero_poly = isl.PwQPolynomial('{ 0 }')
    op_counts = []
    #addf32
    op_counts.append(op_map.get(
                               (typef32, 'add'), zero_poly
                               ).eval_with_dict(param_dict))
    #subf32 
    op_counts[-1] += op_map.get(
                               (typef32, 'sub'), zero_poly
                               ).eval_with_dict(param_dict)
    #mulf32
    op_counts.append(op_map.get(
                               (typef32, 'mul'), zero_poly
                               ).eval_with_dict(param_dict))
    #divf32
    op_counts.append(op_map.get(
                               (typef32, 'div'), zero_poly
                               ).eval_with_dict(param_dict))
    #addi32
    op_counts.append(op_map.get(
                               (typei32, 'add'), zero_poly
                               ).eval_with_dict(param_dict))
    #subi32
    op_counts[-1] += op_map.get(
                               (typei32, 'sub'), zero_poly
                               ).eval_with_dict(param_dict)
    #muli32
    op_counts.append(op_map.get(
                               (typei32, 'mul'), zero_poly
                               ).eval_with_dict(param_dict))
    #divi32
    op_counts.append(op_map.get(
                               (typei32, 'div'), zero_poly
                               ).eval_with_dict(param_dict))
    return op_counts


def get_32b_amd_flops(op_map, param_dict):
    typef32 = np.dtype(np.float32)
    zero_poly = isl.PwQPolynomial('{ 0 }')
    op_counts = []
    #addf32
    op_counts.append(op_map.get(
                               (typef32, 'add'), zero_poly
                               ).eval_with_dict(param_dict))
    #subf32 
    op_counts[-1] += op_map.get(
                               (typef32, 'sub'), zero_poly
                               ).eval_with_dict(param_dict)
    #mulf32
    op_counts.append(op_map.get(
                               (typef32, 'mul'), zero_poly
                               ).eval_with_dict(param_dict))
    #divf32
    op_counts.append(op_map.get(
                               (typef32, 'div'), zero_poly
                               ).eval_with_dict(param_dict))
    return op_counts


def append_mat(A1, A2):
    for row in range(len(A2)):
        A1.append(copy.deepcopy(A2[row]))


def append_mats(mats_1, mats_2):
    for i in range(len(mats_1)):
        append_mat(mats_1[i], mats_2[i])


def divide_rows_by_weights(A, y):
    for row in range(len(A)):
        for col in range(len(A[0])):
            A[row][col] = A[row][col]/y[row]


def ptx_dump(ctx, knl, n, bx, by):
    cknl = lp.compiled.CompiledKernel(ctx, knl)
    ptx_src = cknl.cl_kernel_info().cl_kernel.program.binaries[0]
    filename = "ptx_files/"+knl.name+"_"+str(n)+"_"+str(bx)+"_"+str(by)+".ptx"
    ptx_src_file = open(filename, 'w')
    ptx_src_file.write(ptx_src)


