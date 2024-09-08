/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: InfiniOps.td                                                         *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

MLIR_DEFINE_EXPLICIT_TYPE_ID(infini::infini_mlir::InfiniDialect)
namespace infini {
namespace infini_mlir {

InfiniDialect::InfiniDialect(::mlir::MLIRContext *context)
    : ::mlir::Dialect(getDialectNamespace(), context, ::mlir::TypeID::get<InfiniDialect>())
    
     {
  
  initialize();
}

InfiniDialect::~InfiniDialect() = default;

} // namespace infini_mlir
} // namespace infini
