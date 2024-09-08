include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Infini_Dialect : Dialect {
    let name = "infini";
    let cppNamespace = "infini::infini_mlir";
}

def Infini_TransposeOp : Op<Infini_Dialect, "transpose", [Pure]> {
  let summary = "Transpose operation";
  let description = [{
    "Transpose the input tensor."
  }];
  let arguments = (ins
    AnyRankedTensor:$input,
    ArrayAttr:$permutation,
    BoolAttr:$conjugate
  );
  let results = (outs AnyRankedTensor:$output);
  let hasCanonicalizer = 1;
}

def Infini_MatMulOp : Op<Infini_Dialect, "matmul", [Pure]> {
    let summary = "Matrix multiplication operation";
    let description = [{
        This operation performs the multiplication of two matrices.
    }];
    let arguments = (ins
      F32Tensor:$lhs,
      F32Tensor:$rhs,
      BoolAttr:$transpose_lhs,
      BoolAttr:$transpose_rhs
    );
    let results = (outs F32Tensor:$result);
    let hasCanonicalizer = 1;
}