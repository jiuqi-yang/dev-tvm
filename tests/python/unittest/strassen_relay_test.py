# ***********************************************
# MYID   : Chen Fan
# LANG   : PYTHON
# PROG   : 
# ***********************************************

import os
import sys
import tvm
from tvm import relay
from tvm.contrib import graph_runtime
import numpy as np

def strassen_gemm(N, K, M, max_level=1):
    # A [N, K]
    # B [K, M]
    # C [N, M]
    def gemm(A, B, N, K, M, level):
        if (level < max_level and N % 2 == 0 and
                K % 2 == 0 and M % 2 == 0):
            return strassen(A, B, N, K, M, level)
        else:
            return direct(A, B, N, K, M)

    def direct(A, B, N, K, M):
        C = relay.nn.dense(A, relay.transpose(B, [1, 0]))
        return C

    def split(A, new_x, new_y):
        A11 = relay.strided_slice(A, [0, 0], [new_x, new_y])
        A12 = relay.strided_slice(A, [0, new_y], [new_x, new_y*2])
        A21 = relay.strided_slice(A, [new_x, 0], [new_x*2, new_y])
        A22 = relay.strided_slice(A, [new_x, new_y], [new_x*2, new_y*2])
        return A11, A12, A21, A22

    def strassen(A, B, N, K, M, level):
        new_n = int(N / 2)
        new_k = int(K / 2)
        new_m = int(M / 2)

        A11, A12, A21, A22 = split(A, new_n, new_k)
        B11, B12, B21, B22 = split(B, new_k, new_m)

        S1 = B12 - B22
        P1 = gemm(A11, S1, new_n, new_k, new_m, level+1)
        S2 = A11 + A12
        P2 = gemm(S2, B22, new_n, new_k, new_m, level+1)
        C12 = P1 + P2

        S3 = A21 + A22
        P3 = gemm(S3, B11, new_n, new_k, new_m, level+1)
        S4 = B21 - B11
        P4 = gemm(A22, S4, new_n, new_k, new_m, level+1)
        C21 = P3 + P4

        S5 = A11 + A22
        S6 = B11 + B22
        P5 = gemm(S5, S6, new_n, new_k, new_m, level+1)
        S7 = A12 - A22
        S8 = B21 + B22
        P6 = gemm(S7, S8, new_n, new_k, new_m, level+1)
        C11 = P5 + P4 - P2 + P6

        S9 = A11 - A21
        S10 = B11 + B12
        P7 = gemm(S9, S10, new_n, new_k, new_m, level+1)
        C22 = P5 + P1 - P3 - P7

        C1 = relay.concatenate([C11, C12], 1)
        C2 = relay.concatenate([C21, C22], 1)
        C = relay.concatenate([C1, C2], 0)
        return C

    def strassen_merge(A, B, N):
        new_n = int(N / 2)

        A11, A12, A21, A22 = split(A, new_n)
        B11, B12, B21, B22 = split(B, new_n)

        S1 = B12 - B22
        S2 = A11 + A12
        S3 = A21 + A22
        S4 = B21 - B11
        S5 = A11 + A22
        S6 = B11 + B22
        S7 = A12 - A22
        S8 = B21 + B22
        S9 = A11 - A21
        S10 = B11 + B12

        if new_n > direct_size:
            P1 = gemm(A11, S1, new_n)
            P2 = gemm(S2, B22, new_n)
            P3 = gemm(S3, B11, new_n)
            P4 = gemm(A22, S4, new_n)
            P5 = gemm(S5, S6, new_n)
            P6 = gemm(S7, S8, new_n)
            P7 = gemm(S9, S10, new_n)
        else:
            Merge_A = []
            for a in [A11, S2, S3, A22, S5, S7, S9]:
                Merge_A.append(relay.expand_dims(a, 0))
            Merge_A = relay.concatenate(Merge_A, 0)

            Merge_B = []
            for b in [S1, B22, B11, S4, S6, S8, S10]:
                Merge_B.append(relay.expand_dims(b, 0))
            Merge_B = relay.concatenate(Merge_B, 0)

            Merge_C = relay.nn.batch_matmul(Merge_A, relay.transpose(Merge_B, [0, 2, 1]))
            ss = relay.split(Merge_C, 7)
            P1 = relay.reshape(ss[0], [new_n, new_n])
            P2 = relay.reshape(ss[1], [new_n, new_n])
            P3 = relay.reshape(ss[2], [new_n, new_n])
            P4 = relay.reshape(ss[3], [new_n, new_n])
            P5 = relay.reshape(ss[4], [new_n, new_n])
            P6 = relay.reshape(ss[5], [new_n, new_n])
            P7 = relay.reshape(ss[6], [new_n, new_n])

        C11 = P5 + P4 - P2 + P6
        C12 = P1 + P2
        C21 = P3 + P4
        C22 = P5 + P1 - P3 - P7

        C1 = relay.concatenate([C11, C12], 1)
        C2 = relay.concatenate([C21, C22], 1)
        C = relay.concatenate([C1, C2], 0)
        return C

    A = relay.var("A", shape=(N, K))
    B = relay.var("B", shape=(K, M))
    C = gemm(A, B, N, K, M, 0)
    return A, B, C

def main():
    tgt_host = "llvm"
    tgt = "llvm"
    ctx = tvm.context(tgt, 0)

    N = 128
    K = 256
    M = 512
    A, B, C = strassen_gemm(N, K, M)

    a = tvm.nd.array(np.random.uniform(size=(N, K)).astype(A.type_annotation.dtype), ctx)
    b = tvm.nd.array(np.random.uniform(size=(K, M)).astype(B.type_annotation.dtype), ctx)
    params = {}
    params['A'] = a
    params['B'] = b

    func = relay.Function([A, B], C)
    mod = relay.Module.from_expr(func)
    print(mod)
    print(relay.transform.FuseOps(2)(mod))
    with relay.build_config(opt_level=3, disabled_pass={"KernelLayoutTransform", "DenseWeightTranspose"}):
        graph, lib, opt_params = relay.build_module.build(
            mod, target=tgt)
    m = graph_runtime.create(graph, lib, ctx)

    m.set_input(**params)
    m.run()
    res = m.get_output(0)

    tvm.testing.assert_allclose(res.asnumpy(), np.dot(a.asnumpy(), b.asnumpy()), rtol=1e-5)

if __name__ == "__main__":
    main()