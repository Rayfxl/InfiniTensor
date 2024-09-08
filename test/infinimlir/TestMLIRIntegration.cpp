#include "core/graph.h"
#include "core/runtime.h"
#include "operators/matmul.h"
#include "operators/transpose.h"
#include "gtest/gtest.h"

namespace infini {
namespace infini_mlir {

 TEST(Graph, MLIRIntegration) {
    
        Runtime runtime = NativeCpuRuntimeObj::getInstance();
        Graph g = make_ref<GraphObj>(runtime);
        Tensor i1 = g->addTensor({2, 3, 4, 5}, DataType::UInt32);
        Tensor i2 = g->addTensor({2, 3, 4, 5}, DataType::UInt32);
        Tensor t1 = g->addTensor({2, 3, 5, 4}, DataType::UInt32);
        Tensor t2 = g->addTensor({2, 3, 4, 5}, DataType::UInt32);
        Tensor t3 = g->addTensor({2, 3, 5, 4}, DataType::UInt32);
        Tensor o = g->addTensor({2, 3, 4, 4}, DataType::UInt32);
        g->addOpWithOutputs<TransposeObj>(i1, t1, Shape{0, 1, 3, 2});
        g->addOpWithOutputs<TransposeObj>(t1, t2, Shape{0, 1, 3, 2});
        g->addOpWithOutputs<TransposeObj>(i2, t3, Shape{0, 1, 3, 2});
        g->addOpWithOutputs<MatmulObj>(t2, t3, o);
    
        g->print();
        g->optimize();

        std::cout << "optimize success" << std::endl;

    }

}// namespace infini_mlir
}// namespace infini