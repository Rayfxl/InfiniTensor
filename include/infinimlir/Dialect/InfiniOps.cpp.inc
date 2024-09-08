/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Definitions                                                             *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: InfiniOps.td                                                         *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_OP_LIST
#undef GET_OP_LIST

infini::infini_mlir::MatMulOp,
infini::infini_mlir::TransposeOp
#endif  // GET_OP_LIST

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//

namespace infini {
namespace infini_mlir {

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_InfiniOps0(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((::llvm::isa<::mlir::TensorType>(type))) && ([](::mlir::Type elementType) { return (elementType.isF32()); }(::llvm::cast<::mlir::ShapedType>(type).getElementType())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be tensor of 32-bit float values, but got " << type;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_InfiniOps1(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((::llvm::isa<::mlir::RankedTensorType>(type))) && ([](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>(type).getElementType())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be ranked tensor of any type values, but got " << type;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_InfiniOps0(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !((::llvm::isa<::mlir::BoolAttr>(attr))))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: bool attribute";
  return ::mlir::success();
}
static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_InfiniOps0(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_InfiniOps0(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_InfiniOps1(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !((::llvm::isa<::mlir::ArrayAttr>(attr))))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: array attribute";
  return ::mlir::success();
}
static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_InfiniOps1(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_InfiniOps1(attr, attrName, [op]() {
    return op->emitOpError();
  });
}
} // namespace infini_mlir
} // namespace infini
namespace infini {
namespace infini_mlir {

//===----------------------------------------------------------------------===//
// infini::infini_mlir::MatMulOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
MatMulOpGenericAdaptorBase::MatMulOpGenericAdaptorBase(MatMulOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

bool MatMulOpGenericAdaptorBase::getTransposeLhs() {
  auto attr = getTransposeLhsAttr();
  return attr.getValue();
}

bool MatMulOpGenericAdaptorBase::getTransposeRhs() {
  auto attr = getTransposeRhsAttr();
  return attr.getValue();
}

} // namespace detail
MatMulOpAdaptor::MatMulOpAdaptor(MatMulOp op) : MatMulOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult MatMulOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_transpose_lhs = getProperties().transpose_lhs; (void)tblgen_transpose_lhs;
  if (!tblgen_transpose_lhs) return emitError(loc, "'infini.matmul' op ""requires attribute 'transpose_lhs'");
  auto tblgen_transpose_rhs = getProperties().transpose_rhs; (void)tblgen_transpose_rhs;
  if (!tblgen_transpose_rhs) return emitError(loc, "'infini.matmul' op ""requires attribute 'transpose_rhs'");

  if (tblgen_transpose_lhs && !((::llvm::isa<::mlir::BoolAttr>(tblgen_transpose_lhs))))
    return emitError(loc, "'infini.matmul' op ""attribute 'transpose_lhs' failed to satisfy constraint: bool attribute");

  if (tblgen_transpose_rhs && !((::llvm::isa<::mlir::BoolAttr>(tblgen_transpose_rhs))))
    return emitError(loc, "'infini.matmul' op ""attribute 'transpose_rhs' failed to satisfy constraint: bool attribute");
  return ::mlir::success();
}

::llvm::LogicalResult MatMulOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.transpose_lhs;
       auto attr = dict.get("transpose_lhs");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `transpose_lhs` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.transpose_rhs;
       auto attr = dict.get("transpose_rhs");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `transpose_rhs` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute MatMulOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.transpose_lhs;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("transpose_lhs",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.transpose_rhs;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("transpose_rhs",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code MatMulOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.transpose_lhs.getAsOpaquePointer()), 
    llvm::hash_value(prop.transpose_rhs.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> MatMulOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "transpose_lhs")
      return prop.transpose_lhs;

    if (name == "transpose_rhs")
      return prop.transpose_rhs;
  return std::nullopt;
}

void MatMulOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "transpose_lhs") {
       prop.transpose_lhs = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.transpose_lhs)>>(value);
       return;
    }

    if (name == "transpose_rhs") {
       prop.transpose_rhs = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.transpose_rhs)>>(value);
       return;
    }
}

void MatMulOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.transpose_lhs) attrs.append("transpose_lhs", prop.transpose_lhs);

    if (prop.transpose_rhs) attrs.append("transpose_rhs", prop.transpose_rhs);
}

::llvm::LogicalResult MatMulOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getTransposeLhsAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_InfiniOps0(attr, "transpose_lhs", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getTransposeRhsAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_InfiniOps0(attr, "transpose_rhs", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::llvm::LogicalResult MatMulOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readAttribute(prop.transpose_lhs)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readAttribute(prop.transpose_rhs)))
    return ::mlir::failure();
  return ::mlir::success();
}

void MatMulOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;
  writer.writeAttribute(prop.transpose_lhs);
  writer.writeAttribute(prop.transpose_rhs);
}

bool MatMulOp::getTransposeLhs() {
  auto attr = getTransposeLhsAttr();
  return attr.getValue();
}

bool MatMulOp::getTransposeRhs() {
  auto attr = getTransposeRhsAttr();
  return attr.getValue();
}

void MatMulOp::setTransposeLhs(bool attrValue) {
  getProperties().transpose_lhs = ::mlir::Builder((*this)->getContext()).getBoolAttr(attrValue);
}

void MatMulOp::setTransposeRhs(bool attrValue) {
  getProperties().transpose_rhs = ::mlir::Builder((*this)->getContext()).getBoolAttr(attrValue);
}

void MatMulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::BoolAttr transpose_lhs, ::mlir::BoolAttr transpose_rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.getOrAddProperties<Properties>().transpose_lhs = transpose_lhs;
  odsState.getOrAddProperties<Properties>().transpose_rhs = transpose_rhs;
  odsState.addTypes(result);
}

void MatMulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::BoolAttr transpose_lhs, ::mlir::BoolAttr transpose_rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.getOrAddProperties<Properties>().transpose_lhs = transpose_lhs;
  odsState.getOrAddProperties<Properties>().transpose_rhs = transpose_rhs;
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void MatMulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, bool transpose_lhs, bool transpose_rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.getOrAddProperties<Properties>().transpose_lhs = odsBuilder.getBoolAttr(transpose_lhs);
  odsState.getOrAddProperties<Properties>().transpose_rhs = odsBuilder.getBoolAttr(transpose_rhs);
  odsState.addTypes(result);
}

void MatMulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, bool transpose_lhs, bool transpose_rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.getOrAddProperties<Properties>().transpose_lhs = odsBuilder.getBoolAttr(transpose_lhs);
  odsState.getOrAddProperties<Properties>().transpose_rhs = odsBuilder.getBoolAttr(transpose_rhs);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void MatMulOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<MatMulOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
}

::llvm::LogicalResult MatMulOp::verifyInvariantsImpl() {
  auto tblgen_transpose_lhs = getProperties().transpose_lhs; (void)tblgen_transpose_lhs;
  if (!tblgen_transpose_lhs) return emitOpError("requires attribute 'transpose_lhs'");
  auto tblgen_transpose_rhs = getProperties().transpose_rhs; (void)tblgen_transpose_rhs;
  if (!tblgen_transpose_rhs) return emitOpError("requires attribute 'transpose_rhs'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_InfiniOps0(*this, tblgen_transpose_lhs, "transpose_lhs")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_InfiniOps0(*this, tblgen_transpose_rhs, "transpose_rhs")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_InfiniOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_InfiniOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_InfiniOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult MatMulOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void MatMulOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace infini_mlir
} // namespace infini
MLIR_DEFINE_EXPLICIT_TYPE_ID(infini::infini_mlir::MatMulOp)

namespace infini {
namespace infini_mlir {

//===----------------------------------------------------------------------===//
// infini::infini_mlir::TransposeOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
TransposeOpGenericAdaptorBase::TransposeOpGenericAdaptorBase(TransposeOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

::mlir::ArrayAttr TransposeOpGenericAdaptorBase::getPermutation() {
  auto attr = getPermutationAttr();
  return attr;
}

bool TransposeOpGenericAdaptorBase::getConjugate() {
  auto attr = getConjugateAttr();
  return attr.getValue();
}

} // namespace detail
TransposeOpAdaptor::TransposeOpAdaptor(TransposeOp op) : TransposeOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult TransposeOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_conjugate = getProperties().conjugate; (void)tblgen_conjugate;
  if (!tblgen_conjugate) return emitError(loc, "'infini.transpose' op ""requires attribute 'conjugate'");
  auto tblgen_permutation = getProperties().permutation; (void)tblgen_permutation;
  if (!tblgen_permutation) return emitError(loc, "'infini.transpose' op ""requires attribute 'permutation'");

  if (tblgen_permutation && !((::llvm::isa<::mlir::ArrayAttr>(tblgen_permutation))))
    return emitError(loc, "'infini.transpose' op ""attribute 'permutation' failed to satisfy constraint: array attribute");

  if (tblgen_conjugate && !((::llvm::isa<::mlir::BoolAttr>(tblgen_conjugate))))
    return emitError(loc, "'infini.transpose' op ""attribute 'conjugate' failed to satisfy constraint: bool attribute");
  return ::mlir::success();
}

::llvm::LogicalResult TransposeOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.conjugate;
       auto attr = dict.get("conjugate");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `conjugate` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.permutation;
       auto attr = dict.get("permutation");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `permutation` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute TransposeOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.conjugate;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("conjugate",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.permutation;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("permutation",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code TransposeOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.conjugate.getAsOpaquePointer()), 
    llvm::hash_value(prop.permutation.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> TransposeOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "conjugate")
      return prop.conjugate;

    if (name == "permutation")
      return prop.permutation;
  return std::nullopt;
}

void TransposeOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "conjugate") {
       prop.conjugate = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.conjugate)>>(value);
       return;
    }

    if (name == "permutation") {
       prop.permutation = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.permutation)>>(value);
       return;
    }
}

void TransposeOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.conjugate) attrs.append("conjugate", prop.conjugate);

    if (prop.permutation) attrs.append("permutation", prop.permutation);
}

::llvm::LogicalResult TransposeOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getConjugateAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_InfiniOps0(attr, "conjugate", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getPermutationAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_InfiniOps1(attr, "permutation", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::llvm::LogicalResult TransposeOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readAttribute(prop.conjugate)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readAttribute(prop.permutation)))
    return ::mlir::failure();
  return ::mlir::success();
}

void TransposeOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;
  writer.writeAttribute(prop.conjugate);
  writer.writeAttribute(prop.permutation);
}

::mlir::ArrayAttr TransposeOp::getPermutation() {
  auto attr = getPermutationAttr();
  return attr;
}

bool TransposeOp::getConjugate() {
  auto attr = getConjugateAttr();
  return attr.getValue();
}

void TransposeOp::setConjugate(bool attrValue) {
  getProperties().conjugate = ::mlir::Builder((*this)->getContext()).getBoolAttr(attrValue);
}

void TransposeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input, ::mlir::ArrayAttr permutation, ::mlir::BoolAttr conjugate) {
  odsState.addOperands(input);
  odsState.getOrAddProperties<Properties>().permutation = permutation;
  odsState.getOrAddProperties<Properties>().conjugate = conjugate;
  odsState.addTypes(output);
}

void TransposeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::ArrayAttr permutation, ::mlir::BoolAttr conjugate) {
  odsState.addOperands(input);
  odsState.getOrAddProperties<Properties>().permutation = permutation;
  odsState.getOrAddProperties<Properties>().conjugate = conjugate;
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void TransposeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input, ::mlir::ArrayAttr permutation, bool conjugate) {
  odsState.addOperands(input);
  odsState.getOrAddProperties<Properties>().permutation = permutation;
  odsState.getOrAddProperties<Properties>().conjugate = odsBuilder.getBoolAttr(conjugate);
  odsState.addTypes(output);
}

void TransposeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::ArrayAttr permutation, bool conjugate) {
  odsState.addOperands(input);
  odsState.getOrAddProperties<Properties>().permutation = permutation;
  odsState.getOrAddProperties<Properties>().conjugate = odsBuilder.getBoolAttr(conjugate);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void TransposeOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<TransposeOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
}

::llvm::LogicalResult TransposeOp::verifyInvariantsImpl() {
  auto tblgen_conjugate = getProperties().conjugate; (void)tblgen_conjugate;
  if (!tblgen_conjugate) return emitOpError("requires attribute 'conjugate'");
  auto tblgen_permutation = getProperties().permutation; (void)tblgen_permutation;
  if (!tblgen_permutation) return emitOpError("requires attribute 'permutation'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_InfiniOps1(*this, tblgen_permutation, "permutation")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_InfiniOps0(*this, tblgen_conjugate, "conjugate")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_InfiniOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_InfiniOps1(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult TransposeOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void TransposeOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace infini_mlir
} // namespace infini
MLIR_DEFINE_EXPLICIT_TYPE_ID(infini::infini_mlir::TransposeOp)


#endif  // GET_OP_CLASSES

