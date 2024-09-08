#include "infinimlir/Execution/MLIRExecutionEngine.h"
#include "infinimlir/Conversion/InfiniToMLIR.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "infinimlir/Passes/InfiniPasses.h"

namespace infini {
namespace infini_mlir {

MLIRExecutionEngine::MLIRExecutionEngine(mlir::MLIRContext &context)
    : context(context), passManager(&context)
{
    context.getOrLoadDialect<mlir::func::FuncDialect>();
    context.getOrLoadDialect<infini::infini_mlir::InfiniDialect>();
    passManager.addPass(mlir::createCanonicalizerPass());
}

void MLIRExecutionEngine::compileAndRun(GraphObj *graph) {
    mlir::OpBuilder builder(&context);
    // create module
    auto module = mlir::ModuleOp::create(builder.getUnknownLoc());
    
    // Inputs to argument map
    std::unordered_map<Tensor, mlir::Value> tensorToArgumentMap;
    // get input types for function signature
    std::vector<mlir::Type> inputTypes;
    for (const auto &inputTensor : graph->getInputs()) {
        auto type = convertTensorToMLIRType(builder, inputTensor);
        inputTypes.push_back(type);
      
    }
    
    // get the last operation in the graph
    auto lastOp = graph->getOperators().back();
    // get output type for function signature
    auto outputType = convertTensorToMLIRType(builder, lastOp->getOutput());
    // create function type
    auto funcType = builder.getFunctionType(inputTypes, outputType);
    
    // create function
    auto func = mlir::func::FuncOp::create(builder.getUnknownLoc(), "maingraph", funcType);
    
    // create entry block
    auto *entryBlock = func.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    // create mapping from input tensors to function arguments
    for (size_t i = 0; i < graph->getInputs().size(); ++i) {
        mlir::Value arg = entryBlock->getArgument(i);
        tensorToArgumentMap[graph->getInputs()[i]] = arg;
    }

    // create mapping from operation to result value
    std::unordered_map<UidBaseType, mlir::Value> opToResultMap;

    // traverse the graph and convert each operation to MLIR
    for (const auto &op : graph -> getOperators()) {
        std::vector<mlir::Value> inputs;
        for (const auto &inputTensor : op->getInputs()) {
            Operator producer = inputTensor->getSource();
            // if no producer, then input is an argument of the function
            // otherwise, input is the result of the producer operation
            if (!producer) {
                if (tensorToArgumentMap.find(inputTensor) != tensorToArgumentMap.end()) {
                    inputs.push_back(tensorToArgumentMap[inputTensor]);
                } else {
                    std::cerr << "Error: Missing input tensor argument!" << std::endl;
                    return;
                }
            } else {
                if (opToResultMap.find(producer -> getGuid()) != opToResultMap.end()) {
                    inputs.push_back(opToResultMap[producer -> getGuid()]);
                } else {
                    std::cerr << "Error: Missing input from previous operation!" << std::endl;
                    return;
                }
            }            
        }
        // convert operation to MLIR
        mlir::Operation *mlirOp = convertOpToMLIR(builder, op, inputs);
        if (mlirOp) {
            // save the result of the operation in the map
            opToResultMap[op -> getGuid()] = mlirOp->getResult(0);
        } else {
            std::cerr << "Failed to convert operation." << std::endl;
            return;
        }
    }
    
    // create return operation
    mlir::Value finalResult = opToResultMap[lastOp -> getGuid()];
    if (finalResult) {
        builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(), finalResult);
    } else {
        std::cerr << "Error: Missing final operation result!" << std::endl;
        return;
    }
    // add function to module
    module.push_back(func);
    // print before optimization
    module.dump();

    if (mlir::failed(passManager.run(module))) {
        module.emitError("Optimization failed.");
        return;
    }
    // print after optimization
    module.dump();
 
}

}// namespace infini_mlir
}// namespace infini