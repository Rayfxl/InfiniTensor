#pragma once
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Operation.h"
#include "core/operator.h"
#include "core/tensor.h"
#include "operators/matmul.h"
#include "operators/transpose.h"
#include "infinimlir/Dialect/InfiniOps.h"

namespace infini {
namespace infini_mlir {

mlir::Type convertTensorToMLIRType(mlir::OpBuilder &builder, const Tensor &tensor);
mlir::Operation *convertOpToMLIR(mlir::OpBuilder &builder, const Operator &op, const std::vector<mlir::Value> &inputs);
mlir::Operation *convertMatMulToMLIR(mlir::OpBuilder &builder, const Operator &op, const std::vector<mlir::Value> &inputs);
mlir::Operation *convertTransposeToMLIR(mlir::OpBuilder &builder, const Operator &op, const std::vector<mlir::Value> &inputs);

}// namespace infini_mlir
}// namespace infini