import tensorflow as tf  # Import TensorFlow
import numpy as np  # Import NumPy for numerical operations
import matplotlib.pyplot as plt  # Import Matplotlib for plotting
import sklearn  # Import scikit-learn for machine learning utilities
from sklearn.metrics import confusion_matrix, roc_curve  # Import metrics for evaluation
import seaborn as sns  # Import Seaborn for data visualization
import datetime
import io
import os
import random
from PIL import Image
import tensorflow_datasets as tfds  # Import TensorFlow Datasets for dataset management
from tensorflow.keras.models import Model  # Import Model from Keras
from tensorflow.keras.layers import Layer  # Import Layer from Keras
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, InputLayer, BatchNormalization, Input, Dropout, RandomFlip, RandomRotation, Resizing, Rescaling  # Import various layers from Keras
from tensorflow.keras.losses import BinaryCrossentropy  # Import Binary Crossentropy loss
from tensorflow.keras.metrics import BinaryAccuracy, FalsePositives, FalseNegatives, TruePositives, TrueNegatives, Precision, Recall, AUC, binary_accuracy  # Import metrics for evaluation
from tensorflow.keras.optimizers import Adam  # Import Adam optimizer
from tensorflow.keras.callbacks import Callback, CSVLogger, EarlyStopping, LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau  # Import callbacks
from tensorflow.keras.regularizers import L2, L1  # Import regularizers
from tensorboard.plugins.hparams import api as hp  # Import TensorBoard for hyperparameter tuning

# Suppress TensorFlow logs except errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Disable oneDNN optimizations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Load the Malaria dataset
dataset, dataset_info = tfds.load('malaria', with_info=True, as_supervised=True, shuffle_files=True, split=['train'])

# Function to split the dataset into train, validation, and test sets
def splits(dataset, TRAIN_RATIO, VAL_RATIO, TEST_RATIO):
    DATA_SIZE = len(dataset)
    train_dataset = dataset.take(int(TRAIN_RATIO * DATA_SIZE))
    val_test_dataset = dataset.skip(int(TRAIN_RATIO * DATA_SIZE))
    val_dataset = val_test_dataset.take(int(VAL_RATIO * DATA_SIZE))
    test_dataset = val_test_dataset.skip(int(VAL_RATIO * DATA_SIZE))
    return train_dataset, val_dataset, test_dataset

# Define train, validation, and test ratios
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# Split the dataset
train_dataset, val_dataset, test_dataset = splits(dataset[0], TRAIN_RATIO, VAL_RATIO, TEST_RATIO)

# Resize images to a fixed size
IM_SIZE = 224
def resizing(image, label):
    return tf.image.resize(image, (IM_SIZE, IM_SIZE)) / 255.0, label

# Apply resizing to datasets
train_dataset = train_dataset.map(resizing)
val_dataset = val_dataset.map(resizing)
test_dataset = test_dataset.map(resizing)

# Prepare datasets for training
train_dataset = train_dataset.shuffle(buffer_size=8, reshuffle_each_iteration=True).batch(32).prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.shuffle(buffer_size=8, reshuffle_each_iteration=True).batch(32).prefetch(tf.data.AUTOTUNE)

# Define the LeNet model
lenet_model = tf.keras.Sequential([
    InputLayer(input_shape=(IM_SIZE, IM_SIZE, 3)),
    Conv2D(filters=6, kernel_size=3, strides=1, padding="valid", activation='relu'),
    BatchNormalization(),
    MaxPool2D(pool_size=2, strides=2),
    Conv2D(filters=16, kernel_size=3, strides=1, padding="valid", activation='relu'),
    BatchNormalization(),
    MaxPool2D(pool_size=2, strides=2),
    Flatten(),
    Dense(100, activation="relu"),
    BatchNormalization(),
    Dense(10, activation="relu"),
    BatchNormalization(),
    Dense(1, activation="sigmoid"),
])

# Compile the model
lenet_model.compile(optimizer=Adam(learning_rate=0.01),
                    loss=BinaryCrossentropy(),
                    metrics=[BinaryAccuracy(), Precision(), Recall(), AUC()])

# Train the model
history = lenet_model.fit(train_dataset, validation_data=val_dataset, epochs=20, verbose=1)

# Plot training and validation loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(['Train Loss', 'Validation Loss'])
plt.show()

# Prepare test dataset for evaluation
test_dataset = test_dataset.batch(1)

# Evaluate the model on the test dataset
lenet_model.evaluate(test_dataset)

# Predict on a single example from the test dataset
prediction = lenet_model.predict(test_dataset.take(1))[0][0]

# Function to classify predictions
def parasite_or_not(x=prediction):
    if x < 0.5:
        return 'Parasitized'
    else:
        return 'Uninfected'

# Classify a single example
print(parasite_or_not())

# Plot predictions for a few test examples
for i, (image, label) in enumerate(test_dataset.take(9)):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(image[0])
    plt.title(f'True: {parasite_or_not(label.numpy()[0])}\nPred: {parasite_or_not(lenet_model.predict(image)[0][0])}')
    plt.axis('off')
plt.show()
