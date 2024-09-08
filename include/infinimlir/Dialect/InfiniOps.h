#pragma once
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/TypeID.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Builders.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"

#include "infinimlir/Dialect/InfiniOpsDialect.h.inc"
#define GET_OP_CLASSES
#include "infinimlir/Dialect/InfiniOps.h.inc"

