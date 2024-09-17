#pragma once
#include <vector>
#include <cstdint>
#include "llvm/ADT/ArrayRef.h"

namespace infini {
namespace infini_mlir {

std::vector<int> int64t_to_int(const llvm::ArrayRef<int64_t> &shape);
std::vector<int64_t> int_to_int64t(const std::vector<int> &shape);

} // namespace infini_mlir
} // namespace infini
