#!/usr/bin/env python3
import argparse
import os
import subprocess
import tf2onnx
import coremltools as ct
import coremltools.optimize.coreml as cto
import numpy as np
import tensorflow as tf
import tensorflow.lite as lite
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def parse_args():
  parser = argparse.ArgumentParser(description=('Convert existing Keras model to Float32 TFLite, Int8 EdgeTPU & Int8 CoreML'),add_help=False)
  parser.add_argument('--input_model',type=str,required=True,help='Path to trained Keras model (.h5, .keras, or SavedModel directory)')
  parser.add_argument('--output_model_path',type=str,default='model.tflite',help='Base output path for .tflite model (default: model.tflite)')
  parser.add_argument('--calib_dir',type=str,default=None,help=('Path to validation/calibration image folder for real Int8 calibration (optional)'))
  parser.add_argument('--image_size',type=int,default=224,help='Input image size (default: 224)')
  parser.add_argument('--help',action='help',default=argparse.SUPPRESS,help='show this help message and exit')
  return parser.parse_args()


def get_representative_dataset(calib_dir, image_size):
  """Liefert Kalibrierungsdaten aus echten Bildern (falls --calib_dir angegeben)

  oder erzeugt synthetische Kalibrierungsdaten im Bereich [0.0, 1.0].
  """
  if calib_dir and os.path.exists(calib_dir):
    print(f'Nutze echte Bilder zur Int8-Kalibrierung aus: {calib_dir}')
    datagen = ImageDataGenerator(rescale=1.0 / 255)
    generator = datagen.flow_from_directory(calib_dir,target_size=(image_size, image_size),batch_size=1,class_mode=None,shuffle=True)

    def rep_ds():
      num_samples = min(100, generator.samples)
      for _ in range(num_samples):
        img = next(generator)
        yield [img.astype(np.float32)]

    return rep_ds
  else:
    print(
        'Hinweis: Kein --calib_dir angegeben. Nutze Zufallsdaten [0..1] zur'
        ' Int8-Kalibrierung.'
    )

    def rep_ds():
      for _ in range(100):
        yield [np.random.rand(1, image_size, image_size, 3).astype(np.float32)]

    return rep_ds


def main():
  args = parse_args()

  print(f'Lade trainiertes Modell: {args.input_model} ...')
  model = tf.keras.models.load_model(args.input_model)

  # Input-Shape automatisch aus dem geladenen Modell ermitteln, falls möglich
  input_shape = model.input_shape
  img_h = (
      input_shape
      if (input_shape and input_shape is not None)
      else args.image_size
  )
  img_w = (
      input_shape
      if (input_shape and input_shape is not None)
      else args.image_size
  )
  print(f'Erkannte Input-Größe: {img_w}x{img_h}')

  # =========================================================================
  # 1. Standard TFLite (Float32) -> für --device cpu & --device gpu
  # =========================================================================
  print('\n[1/3] Exportiere model.tflite (Float32)...')
  converter_fp32 = lite.TFLiteConverter.from_keras_model(model)
  tflite_fp32 = converter_fp32.convert()

  with open(args.output_model_path, 'wb') as f:
    f.write(tflite_fp32)
  print(f'Gespeichert: {args.output_model_path}')

  # =========================================================================
  # 2. Full-Integer Int8 TFLite -> für --device tpu (Coral Edge TPU)
  # =========================================================================
  print('\n[2/3] Exportiere Int8 TFLite für Edge TPU...')
  int8_tflite_path = args.output_model_path.replace('.tflite', '_int8.tflite')

  converter_int8 = lite.TFLiteConverter.from_keras_model(model)
  converter_int8.optimizations = [tf.lite.Optimize.DEFAULT]
  converter_int8.representative_dataset = get_representative_dataset(args.calib_dir, img_h)
  converter_int8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
  converter_int8.inference_input_type = tf.uint8
  converter_int8.inference_output_type = tf.uint8

  tflite_int8 = converter_int8.convert()
  with open(int8_tflite_path, 'wb') as f:
    f.write(tflite_int8)
  print(f'Gespeichert: {int8_tflite_path}')

  # EdgeTPU-Compiler automatisch anstoßen, falls auf dem System vorhanden
  try:
    subprocess.run(['edgetpu_compiler', int8_tflite_path], check=True)
    print('EdgeTPU-Kompilierung erfolgreich abgeschlossen.')
  except (FileNotFoundError, subprocess.CalledProcessError):
    print(
        'Hinweis: edgetpu_compiler nicht gefunden (unter macOS normal). Auf'
        f' Linux ausführen: edgetpu_compiler {int8_tflite_path}'
    )

  # =========================================================================
  # 3. CoreML .mlpackage (Int8) -> für --device mps & --device npu
  # =========================================================================
  print('\n[3/3] Exportiere CoreML .mlpackage (Int8) für Apple Silicon...')
  coreml_output_path = args.output_model_path.replace('.tflite', '.mlpackage')
  input_tensor_name = model.inputs.name.split(':')

  mlmodel = ct.convert(model,inputs=[ct.TensorType(name=input_tensor_name, shape=(1, img_h, img_w, 3))],compute_units=ct.ComputeUnit.ALL,minimum_deployment_target=ct.target.macOS13)

  op_config = cto.OpLinearQuantizerConfig(mode='linear_symmetric', dtype='int8')
  config = cto.OptimizationConfig(global_config=op_config)
  mlmodel_int8 = cto.linear_quantize_weights(mlmodel, config=config)

  mlmodel_int8.save(coreml_output_path)
  print(f'Gespeichert: {coreml_output_path}')
  print('\nAlle Konvertierungen abgeschlossen!')

  # =========================================================================
  # 4. Axelera AI Metis AIPU (.onnx -> .axm / .axmodel) -> für --device axelera
  # =========================================================================
  print('\n[4/4] Exportiere ONNX und kompiliere für Axelera Metis AIPU...')

  onnx_output_path = args.output_model_path.replace('.tflite', '.onnx')
  axm_output_path = args.output_model_path.replace('.tflite', '.axm')

  # 1. Keras-Modell zu ONNX (Opset 13-17 wird von Voyager am besten unterstützt)
  spec = (tf.TensorSpec((1, img_h, img_w, 3), tf.float32, name='input'),)
  model_proto, _ = tf2onnx.convert.from_keras(
      model, input_signature=spec, opset=15, output_path=onnx_output_path
  )
  print(f'ONNX-Modell gespeichert: {onnx_output_path}')

  # 2. Mit dem Axelera Voyager SDK Compiler quantisieren & zu .axm kompilieren
  # (Erfordert aktiviertes Voyager SDK Environment auf Linux/x86_64)
  try:
  	from axelera.compiler import compile_model  # Voyager SDK Compiler API

  	compile_model(
  		model_path=onnx_output_path,
      output_path=axm_output_path,
      calibration_dataset=args.calib_dir,  # Nutzt deine Validierungsbilder für Mixed-Precision Quantisierung
  	)
  	print(f'Axelera AIPU-Modell gespeichert: {axm_output_path}')
  except ImportError:
  	print(
  		'Hinweis: axelera.compiler nicht gefunden (ist nur im Axelera Voyager'
  		' SDK auf Linux verfügbar).'
  	)
  	print(
  		'Führe auf dem Axelera-Host folgenden Befehl aus, um die .axm-Datei zu'
      ' erzeugen:'
  	)
  	print(
  		f'  axcompiler --model {onnx_output_path} --output {axm_output_path}'
      ' --quantize'
  	)


if __name__ == '__main__':
  main()
