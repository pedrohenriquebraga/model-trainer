import os
from datetime import datetime

import tensorflow as tf
from keras.applications import MobileNetV2
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.models import Model
from keras.regularizers import l2

from augmentations import (
    build_augment_image,
    preprocess,
)

batch_size = 32
img_size = (300, 300)
epochs = 100
epochs_frozen = 10
epochs_fine_tune_initial = 20
early_stopping_patience = 10
early_stopping_min_delta = 0.01

base_lr = 1e-3
fine_tune_lr_initial = 2e-5
fine_tune_lr = 1e-5
dropout_rate = 0.4
classifier_l2 = 1e-4
initial_fine_tune_layers = 30
fine_tune_layers = 100
shuffle_buffer = 1000 

tf.keras.utils.set_random_seed(42)

gpus = tf.config.list_physical_devices('GPU')
base_dir = './data'
train_dir = './data/train'
val_dir = './data/val'

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPUs disponíveis: {gpus}")
    except RuntimeError as e:
        print(f"Erro ao configurar GPU: {e}")

def list_class_names(directory):
    if not os.path.isdir(directory):
        return []
    return sorted(
        entry for entry in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, entry)) and not entry.startswith('.')
    )

def save_classes_file(base_dir, output_file):
    train_classes = list_class_names(os.path.join(base_dir, 'train'))
    val_classes = list_class_names(os.path.join(base_dir, 'val'))
    all_classes = sorted(set(train_classes) | set(val_classes))

    with open(output_file, 'w') as f:
        for class_name in all_classes:
            f.write(f"{class_name}\n")

    print(f"classes.txt atualizado com {len(all_classes)} classes.")
    return all_classes

def resize_image(image, label):
    image = tf.image.resize_with_pad(image, img_size[0], img_size[1])
    return image, label

all_class_names = save_classes_file(base_dir, 'classes.txt')
if not all_class_names:
    raise ValueError('Nenhuma classe encontrada em ./data/train.')

num_classes = len(all_class_names)

train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir, batch_size=batch_size, label_mode='categorical', class_names=all_class_names
).map(resize_image)

val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    val_dir, batch_size=batch_size, label_mode='categorical', class_names=all_class_names
).map(resize_image)

rotation_layer = tf.keras.layers.RandomRotation(0.1, fill_mode='reflect')
augment_image = build_augment_image(img_size, rotation_layer)

train_dataset = train_dataset.cache().shuffle(
    shuffle_buffer,
    reshuffle_each_iteration=True,
)

train_dataset = train_dataset.map(augment_image)
train_dataset = train_dataset.map(preprocess).prefetch(1)

val_dataset = (
    val_dataset
    .cache()
    .map(preprocess)
    .prefetch(1)
)

base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(img_size[0], img_size[1], 3),
)
base_model.trainable = False

x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(dropout_rate)(x)

output = Dense(
    num_classes,
    activation='softmax',
    kernel_regularizer=l2(classifier_l2),
    dtype='float32',
)(x)

model = Model(inputs=base_model.input, outputs=output)

loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=base_lr),
    loss=loss_fn,
    metrics=[tf.keras.metrics.CategoricalAccuracy(name='accuracy')],
)

os.makedirs('models', exist_ok=True)
os.makedirs('logs', exist_ok=True)
run_id = datetime.now().strftime('%Y%m%d-%H%M%S')

callbacks_list = [
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6, verbose=1),
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=early_stopping_min_delta, patience=early_stopping_patience, mode='min', restore_best_weights=True, verbose=1),
    tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join('models', f'best_{run_id}.keras'), monitor='val_loss', save_best_only=True, verbose=1),
    tf.keras.callbacks.TensorBoard(log_dir=f"logs/fit/{run_id}", histogram_freq=1)
]

print("\n--- Fase 1: Treinando Top Layers (Backbone Congelado) ---")
history_phase1 = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs_frozen,
    callbacks=callbacks_list
)

print("\n--- Fase 2: Fine-Tuning (Descongelando Backbone) ---")
base_model.trainable = True

for layer in base_model.layers[:-initial_fine_tune_layers]:
    layer.trainable = False

for layer in base_model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=fine_tune_lr_initial),
    loss=loss_fn,
    metrics=[tf.keras.metrics.CategoricalAccuracy(name='accuracy')],
)

history_phase2 = model.fit(
    train_dataset,
    validation_data=val_dataset,
    initial_epoch=epochs_frozen,
    epochs=min(epochs, epochs_frozen + epochs_fine_tune_initial),
    callbacks=callbacks_list
)

print("\n--- Fase 3: Fine-Tuning Ampliado ---")
for layer in base_model.layers[:-fine_tune_layers]:
    layer.trainable = False

for layer in base_model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=fine_tune_lr),
    loss=loss_fn,
    metrics=[tf.keras.metrics.CategoricalAccuracy(name='accuracy')],
)

history_phase3 = model.fit(
    train_dataset,
    validation_data=val_dataset,
    initial_epoch=epochs_frozen + epochs_fine_tune_initial,
    epochs=epochs,
    callbacks=callbacks_list
)