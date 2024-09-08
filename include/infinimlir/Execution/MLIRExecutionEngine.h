#pragma once
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "core/graph.h"

namespace infini {
namespace infini_mlir {

class MLIRExecutionEngine {
public:
    MLIRExecutionEngine(mlir::MLIRContext &context);
    void compileAndRun(GraphObj *graph);

private:
    mlir::MLIRContext &context;
    mlir::PassManager passManager;
};

}// namespace infini_mlir
}// namespace infini