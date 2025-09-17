from onnxruntime.quantization import quantize_dynamic, QuantType, QuantFormat

quantize_dynamic(
    "ocr/model.onnx",
    "ocr/quantized_model.onnx",
     weight_type=QuantType.QInt8,
     nodes_to_exclude=['/encoder/embeddings/patch_embeddings/projection/Conv']
)


quantize_dynamic(
    model_input="ocr/encoder.onnx",
    model_output="ocr/encoder_int8.onnx",
    weight_type=QuantType.QInt8,
    op_types_to_quantize=['MatMul', 'Gemm']  # 只量化线性层
)

# 对 decoder 量化
quantize_dynamic(
    model_input="ocr/decoder.onnx",
    model_output="ocr/decoder_int.onnx",
    weight_type=QuantType.QInt8
)