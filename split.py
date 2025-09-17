import torch
from transformers import VisionEncoderDecoderModel

# 加载预训练模型
model = VisionEncoderDecoderModel.from_pretrained("manga-ocr-base")
model.eval()

# =========================
# 1. 导出 Encoder
# =========================
encoder = model.encoder
dummy_image = torch.randn(1, 3, 224, 224)  # 假数据
torch.onnx.export(
    encoder,
    dummy_image,
    "ocr/encoder.onnx",
    input_names=["pixel_values"],
    output_names=["encoder_hidden_states"],
    dynamic_axes={
        "pixel_values": {0: "batch_size"},
        "encoder_hidden_states": {0: "batch_size", 1: "sequence_length"},
    },
    opset_version=11
)
print("Encoder exported successfully.")

# =========================
# 2. 导出 Decoder
# =========================
decoder = model.decoder
# Decoder 输入通常包含：
# - input_ids: 当前已生成的 token
# - encoder_hidden_states: 来自 encoder 的输出
dummy_token_ids = torch.tensor([[2]])  # 假 token
dummy_encoder_outputs = torch.randn(1, 49, model.config.encoder.hidden_size)  # 假 encoder 输出，49 是 7x7 的特征图展开

# 定义一个 wrapper 函数来适配 torch.onnx.export
class DecoderWrapper(torch.nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(self, input_ids, encoder_hidden_states):
        # 注意这里返回 logits 而不是 last_hidden_state
        outputs = self.decoder(input_ids=input_ids, encoder_hidden_states=encoder_hidden_states)
        return outputs.logits  # 改成 logits

decoder_wrapper = DecoderWrapper(decoder)

torch.onnx.export(
    decoder_wrapper,
    (dummy_token_ids, dummy_encoder_outputs),
    "ocr/decoder.onnx",
    input_names=["input_ids", "encoder_hidden_states"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "encoder_hidden_states": {0: "batch_size", 1: "encoder_seq_len"},
        "logits": {0: "batch_size", 1: "sequence_length"},
    },
    opset_version=11
)
print("Decoder exported successfully.")