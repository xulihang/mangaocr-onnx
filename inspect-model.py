import onnx

def inspect_model(model_path):
    model = onnx.load(model_path)

    print("=== Inputs ===")
    for inp in model.graph.input:
        print(inp.name)

    print("\n=== Outputs ===")
    for out in model.graph.output:
        print(out.name)

    print("\n=== Value Infos (中间张量) ===")
    for val in model.graph.value_info:
        print(val.name)

    print("\n=== Initializers (常量) ===")
    for init in model.graph.initializer:
        print(init.name)

if __name__ == "__main__":
    inspect_model("ocr/quantized_model.onnx")