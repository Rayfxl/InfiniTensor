#include "core/tensor.h"
#include "operators/matmul.h"
#include "operators/transpose.h"
#include "infinimlir/Conversion/InfiniToMLIR.h"
#include "infinimlir/Utils/MLIRUtils.h"

namespace infini {
namespace infini_mlir {

mlir::Operation *convertOpToMLIR(mlir::OpBuilder &builder, const Operator &op, const std::vector<mlir::Value> &inputs) {
    if (op->getOpType() == OpType::Transpose) {
        return convertTransposeToMLIR(builder, op, inputs);
    } else if (op->getOpType() == OpType::MatMul) {
        return convertMatMulToMLIR(builder, op, inputs);
    }
    return nullptr;
}

mlir::Operation *convertMatMulToMLIR(mlir::OpBuilder &builder, const Operator &op, const std::vector<mlir::Value> &inputs) {
    IT_ASSERT(inputs.size() == 2);

    auto lhs = inputs[0];
    auto rhs = inputs[1];
    
    auto resultType = mlir::RankedTensorType::get(int_to_int64t(op->getOutput()->getDims()), builder.getF32Type());
    auto Op = op.get();
    auto matmulOp = dynamic_cast<const MatmulObj*>(Op);
    bool transpose_lhs = matmulOp -> getTransA();
    bool transpose_rhs = matmulOp -> getTransB(); 
    return builder.create<MatMulOp>(builder.getUnknownLoc(), resultType, lhs, rhs, transpose_lhs, transpose_rhs);
}

mlir::Operation *convertTransposeToMLIR(mlir::OpBuilder &builder, const Operator &op, const std::vector<mlir::Value> &inputs) {
    IT_ASSERT(inputs.size() == 1);
    auto inputTensor = inputs[0];

    auto resultType = mlir::RankedTensorType::get(int_to_int64t(op->getOutput()->getDims()), builder.getF32Type());
    auto Op = op.get();
    auto transposeOp = dynamic_cast<const TransposeObj*>(Op);
    auto permuteAttr = builder.getI64ArrayAttr(int_to_int64t(transposeOp -> getPermute()));

    return builder.create<TransposeOp>(builder.getUnknownLoc(), resultType, inputTensor, permuteAttr, false);
}

}// namespace infini_mlir
}// namespace infini

