#!/usr/bin/env python3
import argparse
import os
import subprocess
import coremltools as ct
import coremltools.optimize.coreml as cto
import numpy as np
import tensorflow as tf
import tensorflow.lite as lite
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator



def parse_args():
  parser = argparse.ArgumentParser(description='Train a MobileNetV2 image classification model',add_help=False)
  parser.add_argument('image_dir', type=str, default='image', help='path to image directory')
  parser.add_argument('--num_classes', type=int, default=100, help='number of classes')
  parser.add_argument('--image_size',type=int,default=224,help='image size (default 224 for MobileNetV2)')
  parser.add_argument('--batch_size', type=int, default=32, help='batch size')
  parser.add_argument('--output_model_path',type=str,default='model.tflite',required=True,help='Path to output TFLite model.')
  parser.add_argument('--epochs', type=int, default=25, help='number of epochs')
  parser.add_argument('--help',action='help',default=argparse.SUPPRESS,help='show this help message and exit')
  return parser.parse_args()

def save_labels_from_generator(generator, output_file='labels.txt'):
  """Speichert die Klassen in der exakten Reihenfolge des Keras-Generators (0..N)."""
  sorted_classes = sorted(generator.class_indices.items(), key=lambda x: x)
  with open(output_file, 'w') as f:
    for class_name, _ in sorted_classes:
      f.write(f'{class_name}\n')


def main():
  args = parse_args()

  image_dir = args.image_dir
  image_size = (args.image_size, args.image_size)
  batch_size = args.batch_size

  # 1. Data Augmentation für Training
  train_datagen = ImageDataGenerator(rescale=1.0 / 255,rotation_range=20,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.2,zoom_range=0.2,vertical_flip=True,horizontal_flip=True,fill_mode='nearest')

  # Validierung ohne Verzerrungen (nur Rescaling)
  val_datagen = ImageDataGenerator(rescale=1.0 / 255)

  train_generator = train_datagen.flow_from_directory(os.path.join(image_dir, 'train'),target_size=image_size,batch_size=batch_size,class_mode='categorical')

  validation_generator = val_datagen.flow_from_directory(os.path.join(image_dir, 'validate'),target_size=image_size,batch_size=batch_size,class_mode='categorical')

  num_classes = train_generator.num_classes or args.num_classes
  save_labels_from_generator(train_generator, 'labels.txt')

  # 2. MobileNetV2 Modell aufbauen & trainieren
  base_model = MobileNetV2(input_shape=(args.image_size, args.image_size, 3),include_top=False,weights='imagenet')

  for layer in base_model.layers:
    layer.trainable = True

  x = base_model.output
  x = tf.keras.layers.GlobalAveragePooling2D()(x)
  x = tf.keras.layers.Dense(1024, activation='relu')(x)
  predictions = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

  model = tf.keras.Model(inputs=base_model.input, outputs=predictions)
  model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

  model.fit(train_generator,epochs=args.epochs,validation_data=validation_generator)

  # =========================================================================
  # EXPORT 1: Standard TFLite (Float32) -> für --device cpu & --device gpu
  # =========================================================================
  print('\n[1/3] Berechne model.tflite (Float32)...')
  converter_fp32 = lite.TFLiteConverter.from_keras_model(model)
  tflite_fp32_model = converter_fp32.convert()

  with open(args.output_model_path, 'wb') as f:
    f.write(tflite_fp32_model)
  print(f'Gespeichert: {args.output_model_path}')

  # =========================================================================
  # EXPORT 2: Full-Integer Int8 TFLite -> für --device tpu (Coral Edge TPU)
  # =========================================================================
  print('\n[2/3] Berechne Int8 TFLite für Edge TPU...')

  def representative_dataset():
    for _ in range(100):
      images, _ = next(validation_generator)
      for i in range(images.shape):
        yield [np.expand_dims(images[i], axis=0).astype(np.float32)]

  int8_tflite_path = args.output_model_path.replace('.tflite', '_edgetpu_int8.tflite')
  converter_int8 = lite.TFLiteConverter.from_keras_model(model)
  converter_int8.optimizations = [tf.lite.Optimize.DEFAULT]
  converter_int8.representative_dataset = representative_dataset
  converter_int8.target_spec.supported_ops = [
      tf.lite.OpsSet.TFLITE_BUILTINS_INT8
  ]
  converter_int8.inference_input_type = tf.uint8
  converter_int8.inference_output_type = tf.uint8

  tflite_int8_model = converter_int8.convert()
  with open(int8_tflite_path, 'wb') as f:
    f.write(tflite_int8_model)
  print(f'Gespeichert: {int8_tflite_path}')

  # Optional: Falls edgetpu_compiler auf dem System installiert ist, direkt kompilieren
  try:
    subprocess.run(['edgetpu_compiler', int8_tflite_path], check=True)
    print('EdgeTPU-Kompilierung erfolgreich abgeschlossen.')
  except (FileNotFoundError, subprocess.CalledProcessError):
    print(
        'Hinweis: edgetpu_compiler nicht gefunden (unter macOS normal). Auf'
        f' Linux ausführen: edgetpu_compiler {int8_tflite_path}'
    )

  # =========================================================================
  # EXPORT 3: CoreML .mlpackage (Int8) -> für --device mps & --device npu
  # =========================================================================
  print('\n[3/3] Berechne CoreML .mlpackage (Int8) für Apple Silicon...')
  coreml_output_path = args.output_model_path.replace('.tflite', '.mlpackage')
  mlmodel = ct.convert(model,inputs=[ct.TensorType(name='input_1', shape=(1, args.image_size, args.image_size, 3))],compute_units=ct.ComputeUnit.ALL,minimum_deployment_target=ct.target.macOS13)

  op_config = cto.OpLinearQuantizerConfig(mode='linear_symmetric', dtype='int8')
  config = cto.OptimizationConfig(global_config=op_config)
  mlmodel_int8 = cto.linear_quantize_weights(mlmodel, config=config)

  mlmodel_int8.save(coreml_output_path)
  print(f'Gespeichert: {coreml_output_path}')


if __name__ == '__main__':
  main()
