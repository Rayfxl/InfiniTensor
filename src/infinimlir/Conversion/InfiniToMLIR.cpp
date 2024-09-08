#include "infinimlir/Conversion/InfiniToMLIR.h"
#include "infinimlir/Dialect/InfiniOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include <optional>

namespace infini {
namespace infini_mlir {

mlir::Type convertTensorToMLIRType(mlir::OpBuilder &builder, const Tensor &tensor) {
    std::vector<int64_t> dims;
    dims.reserve(tensor->getDims().size());
    for (auto dim : tensor->getDims()) {
        int64_t dim64 = static_cast<int64_t>(dim);
        dims.push_back(dim64);
    }
    // create a tensor type with the given dimensions and element type
    return mlir::RankedTensorType::get(dims, builder.getF32Type());
}

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
    std::vector<int64_t> resultDims;
    resultDims.reserve(op->getOutput()->getDims().size());
    for (auto dim : op->getOutput()->getDims()) {
        int64_t dim64 = static_cast<int64_t>(dim);
        resultDims.push_back(dim64);
    }  
    
    auto resultType = mlir::RankedTensorType::get(resultDims, builder.getF32Type());
    auto Op = op.get();
    auto matmulOp = dynamic_cast<const MatmulObj*>(Op);
    bool transpose_lhs = matmulOp -> getTransA();
    bool transpose_rhs = matmulOp -> getTransB(); 
    return builder.create<MatMulOp>(builder.getUnknownLoc(), resultType, lhs, rhs, transpose_lhs, transpose_rhs);
}

mlir::Operation *convertTransposeToMLIR(mlir::OpBuilder &builder, const Operator &op, const std::vector<mlir::Value> &inputs) {
    IT_ASSERT(inputs.size() == 1);
     auto inputTensor = inputs[0];
    std::vector<int64_t> resultDims;
    resultDims.reserve(op->getOutput()->getDims().size());
    for (auto dim : op->getOutput()->getDims()) {
        resultDims.push_back(static_cast<int64_t>(dim));
    }

    auto resultType = mlir::RankedTensorType::get(resultDims, builder.getF32Type());
    auto Op = op.get();
    auto transposeOp = dynamic_cast<const TransposeObj*>(Op);

    std::vector<int64_t> permute;
    auto permuteInt = transposeOp -> getPermute();
    permute.reserve(permuteInt.size());
    for (auto p : permuteInt) {
        permute.push_back(static_cast<int64_t>(p));
    }
    auto permuteAttr = builder.getI64ArrayAttr(permute);

    return builder.create<TransposeOp>(builder.getUnknownLoc(), resultType, inputTensor, permuteAttr, false);
}

}// namespace infini_mlir
}// namespace infini

