#pragma once
#include "mlir/IR/Operation.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"
#include "core/graph.h"

namespace infini {
namespace infini_mlir {

Graph convertMLIRToInfini(mlir::ModuleOp module);
void handleMatMulOp(Graph g, mlir::Operation *op, llvm::DenseMap<mlir::Value, Tensor> &valueToTensorMap);

} // namespace infini_mlir
} // namespace infini