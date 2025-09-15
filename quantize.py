from onnxruntime.quantization.quantize import quantize_dynamic
from onnxruntime.quantization.quant_utils import QuantType
quantize_dynamic(
    "ocr/model.onnx",
    "ocr/quantized_model.onnx",
     weight_type=QuantType.QInt8,
     nodes_to_exclude=['/encoder/embeddings/patch_embeddings/projection/Conv']
)