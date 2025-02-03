import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import tensorflow as tf

# GPU Configuration
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Enable memory growth for GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Found {len(gpus)} GPU(s). GPU acceleration enabled.")
        # Restrict TensorFlow to only use the first GPU
        tf.config.set_visible_devices(gpus[0], 'GPU')
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f"Using {len(logical_gpus)} Logical GPU(s)")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")
else:
    print("No GPU found. Running on CPU")

# Paths to the datasets
train_dir = 'split_dataset/train'
val_dir = 'split_dataset/val'
test_dir = 'split_dataset/test'
graph_dir = 'graph'
checkpoint_dir = 'checkpoint'
best_model_dir = os.path.join(checkpoint_dir, 'best_model')
epoch_checkpoints_dir = os.path.join(checkpoint_dir, 'epoch_checkpoints')

# Create directories if they don't exist
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(best_model_dir, exist_ok=True)
os.makedirs(epoch_checkpoints_dir, exist_ok=True)
os.makedirs(graph_dir, exist_ok=True)

# Parameters
img_height, img_width = 224, 224
batch_size = 32  # Increased batch size for GPU
epochs = 10
learning_rate = 1e-4

# Configure mixed precision training for better GPU performance
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Function to find the latest checkpoint
def find_latest_checkpoint():
    checkpoints = []
    # Check epoch checkpoints
    if os.path.exists(epoch_checkpoints_dir):
        checkpoints.extend([os.path.join(epoch_checkpoints_dir, f) for f in os.listdir(epoch_checkpoints_dir) if f.endswith('.keras')])
    # Check best model checkpoints
    if os.path.exists(best_model_dir):
        checkpoints.extend([os.path.join(best_model_dir, f) for f in os.listdir(best_model_dir) if f.endswith('.keras')])
    
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        return latest_checkpoint
    return None

# Function to get the starting epoch from checkpoint filename
def get_starting_epoch(checkpoint_path):
    if checkpoint_path is None:
        return 0
    try:
        filename = os.path.basename(checkpoint_path)
        epoch_str = filename.split('_')[2]  # Assuming format: model_epoch_XX_...
        return int(epoch_str) + 1
    except:
        return 0

# Create data generators without augmentation
train_datagen = ImageDataGenerator(rescale=1.0/255.0)
val_datagen = ImageDataGenerator(rescale=1.0/255.0)
test_datagen = ImageDataGenerator(rescale=1.0/255.0)

# Create train, validation, and test generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary'
)

# Print class indices to ensure correct mapping
class_indices = train_generator.class_indices
print(class_indices)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

# Build the model and load weights if available
latest_checkpoint = find_latest_checkpoint()
if latest_checkpoint:
    print(f"Loading weights from checkpoint: {latest_checkpoint}")
    with tf.device('/GPU:0'):
        model = tf.keras.models.load_model(latest_checkpoint)
    initial_epoch = get_starting_epoch(latest_checkpoint)
    print(f"Resuming from epoch {initial_epoch}")
else:
    print("No checkpoint found. Training from scratch.")
    with tf.device('/GPU:0'):
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
        x = base_model.output
        x = Flatten()(x)
        x = Dense(512, activation='relu', kernel_regularizer='l2')(x)
        x = Dropout(0.5)(x)
        x = Dense(512, activation='relu', kernel_regularizer='l2')(x)
        x = Dropout(0.5)(x)
        predictions = Dense(1, activation='sigmoid')(x)

        model = Model(inputs=base_model.input, outputs=predictions)

        # Freeze some of the base model layers
        for layer in base_model.layers[:-4]:
            layer.trainable = False

    # Compile the model with mixed precision optimizer
    optimizer = Adam(learning_rate=learning_rate)
    optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
    model.compile(optimizer=optimizer, 
                 loss='binary_crossentropy', 
                 metrics=['accuracy'])
    initial_epoch = 0

# Define the callbacks
# Checkpoint for best model
best_model_checkpoint = ModelCheckpoint(
    filepath=os.path.join(best_model_dir, 'model_epoch_{epoch:02d}_val_accuracy_{val_accuracy:.4f}.keras'),
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

# Checkpoint after each epoch
epoch_checkpoint = ModelCheckpoint(
    filepath=os.path.join(epoch_checkpoints_dir, 'model_epoch_{epoch:02d}_val_accuracy_{val_accuracy:.4f}.keras'),
    monitor='val_accuracy',
    save_best_only=False,
    mode='max',
    verbose=1
)

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)

if __name__ == '__main__':
    # Train the model
    with tf.device('/GPU:0'):
        history = model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // batch_size,
            epochs=epochs,
            validation_data=val_generator,
            validation_steps=val_generator.samples // batch_size,
            callbacks=[best_model_checkpoint, epoch_checkpoint, early_stopping, reduce_lr],
            initial_epoch=initial_epoch,
            workers=4,  # Increased worker processes for data loading
            use_multiprocessing=False,  # Disable multiprocessing for data loading
            max_queue_size=10  # Prefetch queue size
        )

    # Clear GPU memory after training
    tf.keras.backend.clear_session()

    # Save model history plots
    def save_plot(history, metric, filename):
        plt.figure()
        plt.plot(history.history[metric])
        plt.plot(history.history[f'val_{metric}'])
        plt.title(f'Model {metric.capitalize()}')
        plt.xlabel('Epoch')
        plt.ylabel(metric.capitalize())
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig(os.path.join(graph_dir, filename))
        plt.close()

    save_plot(history, 'accuracy', 'model_accuracy.png')
    save_plot(history, 'loss', 'model_loss.png')

    # Evaluate the model
    test_loss, test_acc = model.evaluate(test_generator)

    # Predict using the model
    Y_pred = model.predict(test_generator)
    y_pred = np.round(Y_pred).astype(int).flatten()
    y_true = test_generator.classes

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=test_generator.class_indices.keys())
    disp.plot(cmap=plt.cm.Blues)
    plt.savefig(os.path.join(graph_dir, 'confusion_matrix.png'))
    plt.close()

    # Train data distribution
    labels, counts = np.unique(train_generator.classes, return_counts=True)
    class_labels = [key for key, value in train_generator.class_indices.items()]
    sns.barplot(x=class_labels, y=counts)
    plt.title('Train Data Distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.savefig(os.path.join(graph_dir, 'train_data_distribution.png'))
    plt.close()

    print(f'Test Accuracy: {test_acc:.4f}')