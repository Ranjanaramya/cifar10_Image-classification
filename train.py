# train.py - Train custom CNN on CIFAR-10 (Colab friendly)
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import models, layers, callbacks, optimizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import cifar10

CLASS_NAMES = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

def build_model(input_shape=(32,32,3), num_classes=10):
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=input_shape),
        layers.Conv2D(32, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),

        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),

        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def plot_history(history, out_dir='plots'):
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(8,4))
    plt.plot(history.history.get('accuracy', []), label='train_acc')
    plt.plot(history.history.get('val_accuracy', []), label='val_acc')
    plt.legend(); plt.title('Accuracy')
    plt.savefig(os.path.join(out_dir, 'accuracy.png'))
    plt.close()

    plt.figure(figsize=(8,4))
    plt.plot(history.history.get('loss', []), label='train_loss')
    plt.plot(history.history.get('val_loss', []), label='val_loss')
    plt.legend(); plt.title('Loss')
    plt.savefig(os.path.join(out_dir, 'loss.png'))
    plt.close()

def main(args):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # split small validation set
    val_split = int(len(x_train) * 0.1)
    x_val = x_train[:val_split]
    y_val = y_train[:val_split]
    x_train_sub = x_train[val_split:]
    y_train_sub = y_train[val_split:]

    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )
    datagen.fit(x_train_sub)

    y_train_cat = to_categorical(y_train_sub, 10)
    y_val_cat = to_categorical(y_val, 10)
    y_test_cat = to_categorical(y_test, 10)

    model = build_model(input_shape=(32,32,3), num_classes=10)
    model.compile(optimizer=optimizers.Adam(args.lr),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    checkpoint_cb = callbacks.ModelCheckpoint(args.model_path, save_best_only=True, monitor='val_accuracy')
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

    steps_per_epoch = max(1, len(x_train_sub) // args.batch_size)

    history = model.fit(datagen.flow(x_train_sub, y_train_cat, batch_size=args.batch_size),
                        epochs=args.epochs,
                        validation_data=(x_val, y_val_cat),
                        steps_per_epoch=steps_per_epoch,
                        callbacks=[checkpoint_cb, reduce_lr])

    plot_history(history, out_dir=args.plots_dir)

    # Evaluate on test
    results = model.evaluate(x_test, y_test_cat, verbose=1)
    print('Test loss / acc:', results)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--model_path', type=str, default='models/model_colab.h5')
    parser.add_argument('--plots_dir', type=str, default='plots')
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.model_path) or '.', exist_ok=True)
    main(args)
