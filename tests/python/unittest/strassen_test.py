# ***********************************************
# MYID   : Chen Fan
# LANG   : PYTHON
# PROG   : 
# ***********************************************

import os
import sys
import tvm
import numpy as np

GEMM_LEVEL = 0
GEMM_COUNT = 0
DIRECT_SIZE = 64

def strassen_gemm(N):
    def gemm(A, B, N, name=""):
        global GEMM_COUNT
        if name != "":
            name += "G%d_" % GEMM_COUNT
            GEMM_COUNT += 1
        if (N > DIRECT_SIZE):
            return strassen(A, B, N, name)
        else:
            return direct(A, B, N, name)

    def direct(A, B, N, name):
        k = tvm.reduce_axis((0, N))
        C = tvm.compute(A.shape, lambda i, j: tvm.sum(A[i][k] * B[k][j], axis=k),
                        name=name+'C')
        return C

    def split(A, new_n, ori_name="Matrix"):
        A11 = tvm.compute((new_n, new_n),
            lambda i, j: A[i][j], name=ori_name+"11")
        A12 = tvm.compute((new_n, new_n),
            lambda i, j: A[i][j+new_n], name=ori_name+"12")
        A21 = tvm.compute((new_n, new_n),
            lambda i, j: A[i+new_n][j], name=ori_name+"21")
        A22 = tvm.compute((new_n, new_n),
            lambda i, j: A[i+new_n][j+new_n], name=ori_name+"22")
        return A11, A12, A21, A22

    def sub(A, B, N, name):
        C = tvm.compute((N, N),
            lambda i, j: A[i][j] - B[i][j], name=name)
        return C

    def add(A, B, N, name):
        C = tvm.compute((N, N),
            lambda i, j: A[i][j] + B[i][j], name=name)
        return C

    def strassen(A, B, N, name):
        global GEMM_LEVEL
        new_n = int(N / 2)

        A11, A12, A21, A22 = split(A, new_n, name+"A")
        B11, B12, B21, B22 = split(B, new_n, name+"B")

        S1 = sub(B12, B22, new_n, name+"S1")
        S2 = add(A11, A12, new_n, name+"S2")
        S3 = add(A21, A22, new_n, name+"S3")
        S4 = sub(B21, B11, new_n, name+"S4")
        S5 = add(A11, A22, new_n, name+"S5")
        S6 = add(B11, B22, new_n, name+"S6")
        S7 = sub(A12, A22, new_n, name+"S7")
        S8 = add(B21, B22, new_n, name+"S8")
        S9 = sub(A11, A21, new_n, name+"S9")
        S10 = add(B11, B12, new_n, name+"S10")

        level = GEMM_LEVEL
        GEMM_LEVEL += 1
        P1 = gemm(A11, S1, new_n, name+"L%d_"%level)
        P2 = gemm(S2, B22, new_n, name+"L%d_"%level)
        P3 = gemm(S3, B11, new_n, name+"L%d_"%level)
        P4 = gemm(A22, S4, new_n, name+"L%d_"%level)
        P5 = gemm(S5, S6, new_n, name+"L%d_"%level)
        P6 = gemm(S7, S8, new_n, name+"L%d_"%level)
        P7 = gemm(S9, S10, new_n, name+"L%d_"%level)

        C11 = tvm.compute((new_n, new_n),
                lambda i, j: P5[i][j] + P4[i][j] - P2[i][j] + P6[i][j], name=name+"C11")
        C12 = add(P1, P2, new_n, name+"C12")
        C21 = add(P3, P4, new_n, name+"C21")
        C22 = tvm.compute((new_n, new_n),
                lambda i, j: P5[i][j] + P1[i][j] - P3[i][j] - P7[i][j], name=name+"C22")

        C = tvm.compute((N, N),
                lambda i, j: tvm.if_then_else(i < new_n,
                    tvm.if_then_else(j < new_n, C11[i][j], C12[i][j-new_n]),
                    tvm.if_then_else(j < new_n, C21[i-new_n][j], C22[i-new_n][j-new_n])),
                name=name+"C")
        return C

    A = tvm.placeholder((N, N), name="A")
    B = tvm.placeholder((N, N), name="B")
    C = gemm(A, B, N)
    sch = tvm.create_schedule(C.op)
    return sch, [A, B, C]

def main():
    tgt_host = "llvm"
    tgt = "llvm"

    N = 128
    sch, args = strassen_gemm(N)

    print(tvm.lower(sch, args, simple_mode=True))
    print(len(sch.stages))

    m = tvm.build(sch, args, target=tgt, target_host=tgt_host)
    ctx = tvm.context(tgt, 0)
    a = tvm.nd.array(np.random.uniform(size=(N, N)).astype(args[0].dtype), ctx)
    b = tvm.nd.array(np.random.uniform(size=(N, N)).astype(args[1].dtype), ctx)
    c = tvm.nd.array(np.zeros((N, N), dtype=args[2].dtype), ctx)
    m(a, b, c)

    tvm.testing.assert_allclose(c.asnumpy(), np.dot(a.asnumpy(), b.asnumpy()), rtol=1e-5)

if __name__ == "__main__":
    main()