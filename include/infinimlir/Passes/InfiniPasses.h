#pragma once
#include "mlir/IR/PatternMatch.h"
#include "infinimlir/Dialect/InfiniOps.h"

namespace infini {
namespace infini_mlir {

// Pattern to simplify redundant Transpose operations
struct SimplifyRedundantTranspose : public ::mlir::OpRewritePattern<TransposeOp> {
    using OpRewritePattern<TransposeOp>::OpRewritePattern;

    // Override the matchAndRewrite function to define the rewrite behavior
    ::llvm::LogicalResult matchAndRewrite(TransposeOp op, ::mlir::PatternRewriter &rewriter) const override;
};

// Pattern to fuse Transpose with Matmul operations
struct FuseTransposeMatmul : public ::mlir::OpRewritePattern<MatMulOp> {
    using OpRewritePattern<MatMulOp>::OpRewritePattern;

    // Override the matchAndRewrite function to define the rewrite behavior
    ::llvm::LogicalResult matchAndRewrite(MatMulOp op, ::mlir::PatternRewriter &rewriter) const override;
};

} // namespace infini_mlir
} // namespace infini
