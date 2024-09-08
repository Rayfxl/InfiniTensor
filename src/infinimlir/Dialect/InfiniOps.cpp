#include "infinimlir/Dialect/InfiniOps.h"
#include "infinimlir/Dialect/InfiniOpsDialect.cpp.inc"

namespace infini {
namespace infini_mlir{

void InfiniDialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include "infinimlir/Dialect/InfiniOps.cpp.inc"
    >();
}

} // namespace infini_mlir
} // namespace infini

#define GET_OP_CLASSES
#include "infinimlir/Dialect/InfiniOps.cpp.inc"
