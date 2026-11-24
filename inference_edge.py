#!/usr/bin/env python3
import argparse
import os
import time
import numpy as np
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inferenz & Latenz-Benchmark für MobileNetV2 auf CPU, GPU, EdgeTPU, Apple Silicon (MPS/NPU) & Axelera AIPU"
    )
    parser.add_argument("--image", type=str, required=True, help="Pfad zum Eingabebild")
    parser.add_argument(
        "--device",
        type=str,
        required=True,
        choices=["cpu", "gpu", "tpu", "mps", "npu", "axelera"],
        help="Ziel-Hardware für die Inferenz",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="artifacts/model_training",
        help="Verzeichnis mit den exportierten Modellen und labels.txt",
    )
    parser.add_argument("--image_size", type=int, default=224, help="Bildgröße (Standard: 224)")
    parser.add_argument("--top_k", type=int, default=3, help="Anzahl der Top-Vorhersagen")
    parser.add_argument("--runs", type=int, default=20, help="Anzahl Durchläufe für Benchmark-Messung")
    return parser.parse_args()


def load_labels(labels_path: str):
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"labels.txt nicht gefunden unter: {labels_path}")
    with open(labels_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_image(image_path: str, image_size: int, as_uint8: bool = False) -> np.ndarray:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Bild nicht gefunden: {image_path}")
    img = Image.open(image_path).convert("RGB").resize((image_size, image_size))
    if as_uint8:
        arr = np.array(img, dtype=np.uint8)
    else:
        arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def softmax(x: np.ndarray) -> np.ndarray:
    if np.isclose(np.sum(x), 1.0, atol=1e-2) and np.all(x >= 0.0):
        return x
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def run_tflite_cpu_gpu(model_path: str, image_path: str, image_size: int, device: str, runs: int):
    try:
        import tflite_runtime.interpreter as tflite
    except ImportError:
        import tensorflow.lite as tflite

    delegates = []
    if device == "gpu":
        for lib in ["libtensorflowlite_gpu_delegate.so", "libdelegate.so"]:
            try:
                delegates.append(tflite.load_delegate(lib))
                break
            except Exception:
                continue
        if not delegates:
            print("Hinweis: Kein TFLite GPU-Delegate gefunden, nutze CPU-Threads als Fallback.")

    interpreter = tflite.Interpreter(
        model_path=model_path,
        experimental_delegates=delegates if delegates else None,
        num_threads=os.cpu_count() or 4,
    )
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_data = load_image(image_path, image_size, as_uint8=False)

    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()

    latencies = []
    for _ in range(runs):
        t0 = time.perf_counter()
        interpreter.set_tensor(input_details[0]["index"], input_data)
        interpreter.invoke()
        latencies.append((time.perf_counter() - t0) * 1000.0)

    probs = interpreter.get_tensor(output_details[0]["index"])[0]
    return probs, latencies


def run_edgetpu(model_path: str, image_path: str, image_size: int, runs: int):
    try:
        import tflite_runtime.interpreter as tflite
    except ImportError:
        import tensorflow.lite as tflite

    edgetpu_lib = "libedgetpu.1.dylib" if os.uname().sysname == "Darwin" else "libedgetpu.so.1"
    interpreter = tflite.Interpreter(
        model_path=model_path,
        experimental_delegates=[tflite.load_delegate(edgetpu_lib)],
    )
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_data = load_image(image_path, image_size, as_uint8=True)

    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()

    latencies = []
    for _ in range(runs):
        t0 = time.perf_counter()
        interpreter.set_tensor(input_details[0]["index"], input_data)
        interpreter.invoke()
        latencies.append((time.perf_counter() - t0) * 1000.0)

    raw_out = interpreter.get_tensor(output_details[0]["index"])[0]
    scale, zero_point = output_details[0]["quantization"]
    if scale > 0:
        probs = (raw_out.astype(np.float32) - zero_point) * scale
    else:
        probs = raw_out.astype(np.float32)

    return probs, latencies


def run_coreml(mlpackage_path: str, image_path: str, image_size: int, device: str, runs: int):
    import coremltools as ct

    compute_unit = ct.ComputeUnit.CPU_AND_GPU if device == "mps" else ct.ComputeUnit.ALL
    mlmodel = ct.models.MLModel(mlpackage_path, compute_units=compute_unit)

    input_name = mlmodel.get_spec().description.input[0].name
    output_name = mlmodel.get_spec().description.output[0].name
    input_data = load_image(image_path, image_size, as_uint8=False)

    mlmodel.predict({input_name: input_data})

    latencies = []
    for _ in range(runs):
        t0 = time.perf_counter()
        out = mlmodel.predict({input_name: input_data})
        latencies.append((time.perf_counter() - t0) * 1000.0)

    probs = np.array(out[output_name]).flatten()
    return probs, latencies


def run_axelera(axm_path: str, image_path: str, image_size: int, runs: int):
    from axelera.app import create_inference_session

    input_data = load_image(image_path, image_size, as_uint8=False)
    latencies = []

    with create_inference_session(axm_path) as session:
        session.run(input_data)

        for _ in range(runs):
            t0 = time.perf_counter()
            outputs = session.run(input_data)
            latencies.append((time.perf_counter() - t0) * 1000.0)

    probs = np.array(outputs).flatten()
    return probs, latencies


def main():
    args = parse_args()
    labels = load_labels(os.path.join(args.model_dir, "labels.txt"))

    if args.device in ("cpu", "gpu"):
        model_path = os.path.join(args.model_dir, "model.tflite")
        probs, latencies = run_tflite_cpu_gpu(
            model_path, args.image, args.image_size, args.device, args.runs
        )

    elif args.device == "tpu":
        compiled_path = os.path.join(args.model_dir, "model_edgetpu_int8_edgetpu.tflite")
        raw_int8_path = os.path.join(args.model_dir, "model_edgetpu_int8.tflite")
        model_path = compiled_path if os.path.exists(compiled_path) else raw_int8_path
        probs, latencies = run_edgetpu(model_path, args.image, args.image_size, args.runs)

    elif args.device in ("mps", "npu"):
        mlpackage_path = os.path.join(args.model_dir, "model.mlpackage")
        probs, latencies = run_coreml(
            mlpackage_path, args.image, args.image_size, args.device, args.runs
        )

    elif args.device == "axelera":
        axm_path = os.path.join(args.model_dir, "model.axm")
        probs, latencies = run_axelera(axm_path, args.image, args.image_size, args.runs)

    probs = softmax(probs)
    top_indices = np.argsort(probs)[::-1][: args.top_k]

    print(f"\n=== Ergebnis ({args.device.upper()}) ===")
    for rank, idx in enumerate(top_indices, start=1):
        label = labels[idx] if idx < len(labels) else f"class_{idx}"
        print(f"  #{rank}: {label:<25} ({probs[idx] * 100:.2f}%)")

    avg_ms = float(np.mean(latencies))
    p95_ms = float(np.percentile(latencies, 95))
    fps = 1000.0 / avg_ms if avg_ms > 0 else 0.0

    print(f"\n=== Benchmark ({args.runs} Durchläufe) ===")
    print(f"  Durchschnitt: {avg_ms:.2f} ms ({fps:.1f} FPS)")
    print(f"  P95 Latenz:   {p95_ms:.2f} ms")


if __name__ == "__main__":
    main()
