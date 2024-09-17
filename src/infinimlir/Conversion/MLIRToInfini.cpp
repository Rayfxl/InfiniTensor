#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "operators/matmul.h"
#include "infinimlir/Conversion/MLIRToInfini.h"
#include "infinimlir/Dialect/InfiniOps.h"
#include "infinimlir/Utils/MLIRUtils.h"

namespace infini {
namespace infini_mlir {

void handleMatMulOp(Graph g, mlir::Operation *op, llvm::DenseMap<mlir::Value, Tensor> &valueToTensorMap) {
    auto matmulOp = llvm::dyn_cast<MatMulOp>(op);

    // create lhs tensor
    auto lhsValue = matmulOp.getLhs();
    if (!valueToTensorMap.count(lhsValue)) {
        auto lhsShape = mlir::cast<mlir::RankedTensorType>(lhsValue.getType()).getShape();
        Tensor lhsTensor = g->addTensor(int64t_to_int(lhsShape), DataType::UInt32);
        valueToTensorMap[lhsValue] = lhsTensor;
    }
    Tensor lhs = valueToTensorMap[lhsValue];

    // create rhs tensor
    auto rhsValue = matmulOp.getRhs();
    if (!valueToTensorMap.count(rhsValue)) {
        auto rhsShape = mlir::cast<mlir::RankedTensorType>(rhsValue.getType()).getShape();
        Tensor rhsTensor = g->addTensor(int64t_to_int(rhsShape), DataType::UInt32);
        valueToTensorMap[rhsValue] = rhsTensor;
    }
    Tensor rhs = valueToTensorMap[rhsValue];

    // create output tensor
    auto outputShape = mlir::cast<mlir::RankedTensorType>(matmulOp.getResult().getType()).getShape();
    Tensor output = g->addTensor(int64t_to_int(outputShape), DataType::UInt32);

    // create MatmulObj operator
    g->addOpWithOutputs<MatmulObj>(lhs, rhs, output, matmulOp.getTransposeLhs(), matmulOp.getTransposeRhs());

    // save output tensor to valueToTensorMap
    valueToTensorMap[matmulOp.getResult()] = output;
}

Graph convertMLIRToInfini(mlir::ModuleOp module) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);

    // use DenseMap to store mapping from MLIR Value to InfiniTensor
    llvm::DenseMap<mlir::Value, Tensor> valueToTensorMap;

    // traverse all functions in the module
    for (auto func : module.getOps<mlir::func::FuncOp>()) {
        for (auto &block : func.getBlocks()) {
            for (auto &op : block.getOperations()) {
                // if operation is MatMulOp, handle it
                if (llvm::isa<MatMulOp>(op)) {
                    handleMatMulOp(g, &op, valueToTensorMap);
                }
                // others
            }
        }
    }
    return g;
}

} // namespace infini_mlir
} // namespace infini
