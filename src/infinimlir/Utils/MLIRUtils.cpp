#include "infinimlir/Utils/MLIRUtils.h"
#include <algorithm>

namespace infini {
namespace infini_mlir {

std::vector<int> int64t_to_int(const llvm::ArrayRef<int64_t> &shape) {
    std::vector<int> result(shape.size());
    std::transform(shape.begin(), shape.end(), result.begin(),
                   [](int64_t dim) { return static_cast<int>(dim); });
    return result;
}

std::vector<int64_t> int_to_int64t(const std::vector<int> &shape) {
    std::vector<int64_t> result(shape.size());
    std::transform(shape.begin(), shape.end(), result.begin(),
                   [](int dim) { return static_cast<int64_t>(dim); });
    return result;
}

} // namespace infini_mlir
} // namespace infini
