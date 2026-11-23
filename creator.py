#!/usr/bin/env python3
import argparse
import os
import coremltools as ct
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
  # class_indices ist {'klasse_a': 0, 'klasse_b': 1, ...}
  sorted_classes = sorted(generator.class_indices.items(), key=lambda x: x)
  with open(output_file, 'w') as f:
    for class_name, _ in sorted_classes:
      f.write(f'{class_name}\n')


def main():
  args = parse_args()

  image_dir = args.image_dir
  image_size = (args.image_size, args.image_size)
  batch_size = args.batch_size

  # Data Augmentation für Training
  train_datagen = ImageDataGenerator(rescale=1.0 / 255,rotation_range=20,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.2,zoom_range=0.2,vertical_flip=True,horizontal_flip=True,fill_mode='nearest')

  # Validation ohne zufällige Verzerrungen (nur Rescaling)
  val_datagen = ImageDataGenerator(rescale=1.0 / 255)

  train_generator = train_datagen.flow_from_directory(os.path.join(image_dir, 'train'),target_size=image_size,batch_size=batch_size,class_mode='categorical')

  validation_generator = val_datagen.flow_from_directory(os.path.join(image_dir, 'validate'),target_size=image_size,batch_size=batch_size,class_mode='categorical')

  # Klassenanzahl automatisch aus den Ordnern übernehmen, falls vorhanden
  num_classes = train_generator.num_classes or args.num_classes

  # Labels passend für dein Inferenz-Skript speichern
  save_labels_from_generator(train_generator, 'labels.txt')

  # MobileNetV2 Basismodell mit passender image_size erstellen
  base_model = MobileNetV2(input_shape=(args.image_size, args.image_size, 3),include_top=False,weights='imagenet')

  for layer in base_model.layers:
    layer.trainable = True

  # Klassifikations-Head hinzufügen
  x = base_model.output
  x = tf.keras.layers.GlobalAveragePooling2D()(x)
  x = tf.keras.layers.Dense(1024, activation='relu')(x)
  predictions = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

  model = tf.keras.Model(inputs=base_model.input, outputs=predictions)
  model.compile(
      optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']
  )

  # Modell trainieren
  model.fit(
      train_generator,
      epochs=args.epochs,
      validation_data=validation_generator,
  )

  # 1. Als TFLite speichern (für --device cpu / gpu / tpu)
  converter = lite.TFLiteConverter.from_keras_model(model)
  tflite_model = converter.convert()
  with open(args.output_model_path, 'wb') as f:
    f.write(tflite_model)
  print(f'TFLite-Modell gespeichert unter: {args.output_model_path}')

  # 2. Als CoreML .mlpackage speichern (für --device mps / npu auf macOS)
  coreml_output_path = args.output_model_path.replace('.tflite', '.mlpackage')
  mlmodel = ct.convert(model,inputs=[ct.TensorType(name='input_1', shape=(1, args.image_size, args.image_size, 3))],compute_units=ct.ComputeUnit.ALL,minimum_deployment_target=ct.target.macOS13)
  mlmodel.save(coreml_output_path)
  print(f'CoreML-Modell (NPU/MPS) gespeichert unter: {coreml_output_path}')


if __name__ == '__main__':
  main()
