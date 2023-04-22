#!/usr/bin/env python
import os
import subprocess
import argparse
import tensorflow as tf
import tensorflow.lite as lite
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

def parse_args():
    parser = argparse.ArgumentParser(description='Train a MobileNetV2 image classification model',add_help=False)
    parser.add_argument('image_dir', type=str, default='image', help='path to image directory')
    parser.add_argument('--num_classes', type=int, default=100, help='number of classes')
    parser.add_argument('--image_size', type=int, default=255, help='image size')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--output_model_path', type=str, required=True, help='Path to output TFLite model.')
    parser.add_argument('--epochs', type=int, default=10000, help='number of epochs')
    parser.add_argument('--help', action='help', help='show this help message and exit')
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()

    # Define the image directory path
    image_dir = args.image_dir

    # Define the number of classes
    num_classes = args.num_classes

    # Define the image size
    image_size = (args.image_size, args.image_size)

    # Define the batch size
    batch_size = args.batch_size

    # Create an ImageDataGenerator for data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    # Load the training data
    train_generator = train_datagen.flow_from_directory(
        os.path.join(image_dir, 'train'),
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical')
        
    

    # Create the MobileNetV2 base model
    base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')

    # Freeze the pre-trained layers
    for layer in base_model.layers:
        layer.trainable = True

    # Add a new classification layer on top
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    predictions = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    # Combine the base model and the new classification layer
    model = tf.keras.Model(inputs=base_model.input, outputs=predictions)
    
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(train_generator, epochs=args.epochs)
    
    # Create an ImageDataGenerator for data augmentation
    train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2) # 20% of the training data will be used for validation

    # Load the training data
    train_generator = train_datagen.flow_from_directory(
    os.path.join(image_dir, 'train'),
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training')

    # Load the validation data
    validation_generator = train_datagen.flow_from_directory(
    os.path.join(image_dir, 'validate'),
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation')
    
    # Train the model with the training and validation data
    model.fit(train_generator, epochs=args.epochs, validation_data=validation_generator)
    
    # Convert the model to TensorFlow Lite format
    converter = lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Save the TFLite model
    with open(args.output_model_path, 'wb') as f:
        f.write(tflite_model)

if __name__ == '__main__':
    main()
