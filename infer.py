import re
import jaconv
import numpy as np
import time
from onnxruntime import InferenceSession
from PIL import Image


class MangaOCR():
    def __init__(self, model_path, vocab_path):
        self.session = InferenceSession(model_path)
        self.vocab = self._load_vocab(vocab_path)

    def __call__(self, image: Image.Image):
        image = self._preprocess(image)
        token_ids = self._generate(image)
        text = self._decode(token_ids)
        text = self._postprocess(text)

        return text

    def _load_vocab(self, vocab_file):
        with open(vocab_file, "r", encoding="utf-8") as f:
            vocab = f.read().splitlines()

        return vocab

    def _preprocess(self, image):
        # convert to grayscale
        image = image.convert("L").convert("RGB")
        # resize
        image = image.resize((224, 224), resample=2)
        # rescale
        image = np.array(image, dtype=np.float32)
        image /= 255
        # normalize
        image = (image - 0.5) / 0.5
        # reshape from (224, 224, 3) to (3, 224, 224)
        image = image.transpose((2, 0, 1))
        # add batch size
        image = image[None]

        return image

    def _generate(self, image):
        token_ids = [2]

        for _ in range(300):
            # 将 token_ids 转换为 int64 类型
            token_ids_array = np.array([token_ids], dtype=np.int64)
            
            [logits] = self.session.run(
                output_names=["logits"],
                input_feed={
                    "image": image,
                    "token_ids": token_ids_array,  # 使用转换后的 int64 数组
                },
            )

            token_id = logits[0, -1, :].argmax()
            token_ids.append(int(token_id))

            if token_id == 3:
                break

        return token_ids

    def _decode(self, token_ids):
        text = ""

        for token_id in token_ids:
            if token_id < 5:
                continue

            text += self.vocab[token_id]

        return text

    def _postprocess(self, text):
        text = "".join(text.split())
        text = text.replace("…", "...")
        text = re.sub("[・.]{2,}", lambda x: (x.end() - x.start()) * ".", text)
        text = jaconv.h2z(text, ascii=True, digit=True)

        return text
        
        
if __name__ == "__main__":
    # 记录开始时间
    start_time = time.time()
    ocr = MangaOCR(model_path="ocr/quantized_model.onnx", vocab_path="ocr/vocab.txt")
    # 记录模型初始化时间
    init_time = time.time()
    print(f"模型初始化耗时: {init_time - start_time:.3f} 秒")
    text = ocr(Image.open("image.jpg"))
    print(text)
    # 记录识别完成时间
    end_time = time.time()
    print(f"OCR识别耗时: {end_time - init_time:.3f} 秒")
    print(f"总处理时间: {end_time - start_time:.3f} 秒")
    text = ocr(Image.open("image.jpg"))
    print(text)