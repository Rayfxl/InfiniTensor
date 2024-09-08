#include "infinimlir/Dialect/InfiniOps.h"
#include "mlir/IR/PatternMatch.h"

namespace infini {
namespace infini_mlir {

struct SimplifyRedundantTranspose : public ::mlir::OpRewritePattern<TransposeOp> {
    using OpRewritePattern<TransposeOp>::OpRewritePattern;

    ::llvm::LogicalResult matchAndRewrite(TransposeOp op, ::mlir::PatternRewriter &rewriter) const override {
        ::mlir::Value input = op.getInput();
        ::mlir::Operation *definingOp = input.getDefiningOp();
        // check existence of lastop
        if (!definingOp)
            return ::mlir::failure();
        
        // check if lastop is a TransposeOp
        auto prevTranspose = ::llvm::dyn_cast<TransposeOp>(definingOp);
        if (!prevTranspose)
            return ::mlir::failure();

        // check if the input of the lastop is the same as the output of the current op
        ::mlir::Value result = definingOp->getResult(0);
        if (result != input)
            return ::mlir::failure();
        
        // check if the permutation of the lastop is the same as the permutation of the current op
        auto currentPermutation = op.getPermutation();
        auto prevPermutation = prevTranspose.getPermutation();
        if (currentPermutation != prevPermutation)
            return ::mlir::failure();

         // replace current op with the input of the lastop
        rewriter.replaceOp(op, definingOp -> getOperand(0));
        // delete the lastop
        rewriter.eraseOp(definingOp);
        
        return ::mlir::success();
    }
};

void TransposeOp::getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context) {
    results.add<SimplifyRedundantTranspose>(context);
}


struct FuseTransposeMatmul : public ::mlir::OpRewritePattern<MatMulOp> {
  using OpRewritePattern<MatMulOp>::OpRewritePattern;

  ::llvm::LogicalResult matchAndRewrite(MatMulOp op, ::mlir::PatternRewriter &rewriter) const override {
    // get operands
    ::mlir::Value lhs = op.getLhs();
    ::mlir::Value rhs = op.getRhs();

    // check if lhs is a result of a TransposeOp
    auto lhsTranspose = lhs.getDefiningOp<TransposeOp>();
    // check if rhs is a result of a TransposeOp
    auto rhsTranspose = rhs.getDefiningOp<TransposeOp>();

    bool transposeLhs = false;
    bool transposeRhs = false;

    if (!lhsTranspose && !rhsTranspose) return ::mlir::failure();

    // if lhs is a result of a TransposeOp, get the input of the TransposeOp and mark the transpose attribute as true
    if (lhsTranspose) {
        lhs = lhsTranspose.getInput();
        transposeLhs = true;
    }

    if (rhsTranspose) {
        rhs = rhsTranspose.getInput();
        transposeRhs = true;
    }

    // get the result type of the MatMulOp
    ::mlir::TypedValue<::mlir::TensorType> result = op.getResult();
   
    ::mlir::Type resultType = result.getType();
    
    // create a new MatMulOp with the new operands and the transpose attributes
    auto newMatMul = rewriter.create<MatMulOp>(op.getLoc(), resultType, lhs, rhs, transposeLhs, transposeRhs);
    
    
    // replace the current MatMulOp with the new MatMulOp
    rewriter.replaceOp(op, newMatMul.getResult());
    // delete the TransposeOps if they exist
    if (lhsTranspose) rewriter.eraseOp(lhsTranspose);
    if (rhsTranspose) rewriter.eraseOp(rhsTranspose);
    return ::mlir::success();
  }
};

void MatMulOp::getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context) {
    results.add<FuseTransposeMatmul>(context);
}
}// namespace infini_mlir
}// namespace infini
