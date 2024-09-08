/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: InfiniOps.td                                                         *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace infini {
namespace infini_mlir {

class InfiniDialect : public ::mlir::Dialect {
  explicit InfiniDialect(::mlir::MLIRContext *context);

  void initialize();
  friend class ::mlir::MLIRContext;
public:
  ~InfiniDialect() override;
  static constexpr ::llvm::StringLiteral getDialectNamespace() {
    return ::llvm::StringLiteral("infini");
  }
};
} // namespace infini_mlir
} // namespace infini
MLIR_DECLARE_EXPLICIT_TYPE_ID(infini::infini_mlir::InfiniDialect)
