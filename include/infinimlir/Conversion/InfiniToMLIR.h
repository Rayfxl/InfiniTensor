#pragma once
#include "core/operator.h"
#include "infinimlir/Dialect/InfiniOps.h"

namespace infini {
namespace infini_mlir {

mlir::Operation *convertOpToMLIR(mlir::OpBuilder &builder, const Operator &op, const std::vector<mlir::Value> &inputs);
mlir::Operation *convertMatMulToMLIR(mlir::OpBuilder &builder, const Operator &op, const std::vector<mlir::Value> &inputs);
mlir::Operation *convertTransposeToMLIR(mlir::OpBuilder &builder, const Operator &op, const std::vector<mlir::Value> &inputs);

}// namespace infini_mlir
}// namespace infini