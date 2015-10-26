from __future__ import division, print_function

import numpy as np
import islpy as isl
import pyopencl as cl
import copy
import loopy as lp
import warnings
from pymbolic.mapper import CombineMapper
#from functools import reduce


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


class RegisterUsageEstimator(CombineMapper):

    def __init__(self, knl):
        self.knl = knl
        from loopy.expression import TypeInferenceMapper
        self.type_inf = TypeInferenceMapper(knl)
        self.vars_found = []
        self.subs_found = []

    def combine(self, values):
        return sum(values)

    def forget_prev_vars(self):
        del self.vars_found[:]

    def forget_prev_subs(self):
        del self.subs_found[:]

    def map_constant(self, expr):
        return 0

    def map_variable(self, expr):
        name = expr.name
        if expr in self.vars_found:
            return 0

        self.vars_found.append(expr)
        if name in self.knl.temporary_variables:
            if self.knl.temporary_variables[name].is_local:
                return 0
            else:
                return 1
        elif name in self.knl.all_inames():
            from loopy.kernel.data import AxisTag, VectorizeTag, UnrollTag
            tag = self.knl.iname_to_tag.get(name)
            if (tag is None or not(isinstance(tag, AxisTag)
                                   or isinstance(tag, VectorizeTag)
                                   or isinstance(tag, UnrollTag))):
                return 1
            else:
                return 0
        else:
            return 1

    map_tagged_variable = map_variable

    def map_call(self, expr):
        return self.rec(expr.parameters)

    def map_subscript(self, expr):
        name = expr.aggregate.name  # name of array

        if name in self.knl.arg_dict:
            # not a temporary variable
            array = self.knl.arg_dict[name]
        elif self.knl.temporary_variables[name].is_local:
            # temp var is in shared mem
            return 0 + self.rec(expr.index)
        elif (expr.index, expr.aggregate) in self.subs_found:
            # temp var is NOT shared, but already counted
            return 0 + self.rec(expr.index)
        else:
            # temp var is NOT shared and NOT already counted
            self.subs_found.append((expr.index, expr.aggregate))
            return 1 + self.rec(expr.index)

        # expr is not a temporary variable

        if not isinstance(array, lp.GlobalArg):
            # This array is not in global memory, and is not a temporary variable
            # TODO how should we count arrays in const/texture mem? ImageArg?
            # Ignore for now
            return self.rec(expr.index)

        # this is a global mem access
        if (expr.index, expr.aggregate) in self.subs_found:
            return 0 + self.rec(expr.index)
        else:
            self.subs_found.append((expr.index, expr.aggregate))
            return 1 + self.rec(expr.index)

    def map_sum(self, expr):
        assert expr.children
        return sum(self.rec(child) for child in expr.children)

    map_product = map_sum

    def map_quotient(self, expr, *args):
        return self.rec(expr.numerator) + self.rec(expr.denominator)

    map_floor_div = map_quotient
    map_remainder = map_quotient

    def map_power(self, expr):
        return self.rec(expr.base) + self.rec(expr.exponent)

    def map_left_shift(self, expr):
        return self.rec(expr.shiftee)+self.rec(expr.shift)

    map_right_shift = map_left_shift

    def map_bitwise_not(self, expr):
        return self.rec(expr.child)

    def map_bitwise_or(self, expr):
        return sum(self.rec(child) for child in expr.children)

    map_bitwise_xor = map_bitwise_or
    map_bitwise_and = map_bitwise_or

    def map_comparison(self, expr):
        return self.rec(expr.left)+self.rec(expr.right)

    map_logical_not = map_bitwise_not
    map_logical_or = map_bitwise_or
    map_logical_and = map_logical_or

    def map_if(self, expr):
        warnings.warn("RegisterUsageEstimator counting register usage as "
                      "sum of if-statement branches.")
        return self.rec(expr.condition) + self.rec(expr.then) + self.rec(expr.else_)

    def map_if_positive(self, expr):
        warnings.warn("RegisterUsageEstimator counting register usage as "
                      "sum of if_pos-statement branches.")
        return self.rec(expr.criterion) + self.rec(expr.then) + self.rec(expr.else_)

    map_min = map_bitwise_or
    map_max = map_min

    def map_common_subexpression(self, expr):
        raise NotImplementedError("GlobalSubscriptCounter encountered "
                                  "common_subexpression, "
                                  "map_common_subexpression not implemented.")

    def map_substitution(self, expr):
        raise NotImplementedError("GlobalSubscriptCounter encountered "
                                  "substitution, "
                                  "map_substitution not implemented.")

    def map_derivative(self, expr):
        raise NotImplementedError("GlobalSubscriptCounter encountered "
                                  "derivative, "
                                  "map_derivative not implemented.")

    def map_slice(self, expr):
        raise NotImplementedError("GlobalSubscriptCounter encountered slice, "
                                  "map_slice not implemented.")


def estimate_regs_per_thread(knl):

    """Estimate registers per thread usage by a loopy kernel.

    :parameter knl: A :class:`loopy.LoopKernel` whose reg usage will be estimated.

    :return: An :class:`integer` holding an estimate for the number of registers
             used per thread. This number will most likely be too low, but will
             hopefully be consistantly too low by the same constant factor.

    """

    from loopy.preprocess import preprocess_kernel, infer_unknown_types
    from loopy.schedule import EnterLoop, LeaveLoop, Barrier, RunInstruction  # noqa
    knl = infer_unknown_types(knl, expect_completion=True)
    knl = preprocess_kernel(knl)
    knl = lp.get_one_scheduled_kernel(knl)
    max_regs = 0
    block_reg_totals = [0]
    # counters to track nested sets of previously used iname+index combinations
    reg_counters = [RegisterUsageEstimator(knl)]

    for sched_item in knl.schedule:
        if isinstance(sched_item, EnterLoop):
            block_reg_totals.append(0)
            # start a new estimator
            reg_counters.append(RegisterUsageEstimator(knl))

        elif isinstance(sched_item, LeaveLoop):
            if block_reg_totals[-1] > max_regs:
                max_regs = block_reg_totals[-1]
            # pop to resume previous total
            block_reg_totals.pop()
            reg_counters.pop()

        elif isinstance(sched_item, RunInstruction):
            insn = knl.id_to_insn[sched_item.insn_id]
            block_reg_totals[-1] += reg_counters[-1](insn.assignee) + \
                                    reg_counters[-1](insn.expression)

    # finished looping, check outer block
    if block_reg_totals[-1] > max_regs:
        max_regs = block_reg_totals[-1]

    return max_regs



