import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.models import Sequential, load_model # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from sklearn.metrics import classification_report, confusion_matrix

# --- Part 1: Foundational Setup and Data Ingestion ->

# Define paths to the dataset
dataset_path = '/mnt/d/PROJECTS/Deep Learning Projects/Parkinsons-Disease-Detection/dataset'
train_path = os.path.join(dataset_path, 'train')
valid_path = os.path.join(dataset_path, 'valid')
test_path = os.path.join(dataset_path, 'test')

# Define image dimensions and batch size 
IMG_HEIGHT = 256
IMG_WIDTH = 256
BATCH_SIZE = 16

# Create a data generator for the training set with augmentation
# Try with less aggressive augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=5,      
    width_shift_range=0.05,  
    height_shift_range=0.05, 
    zoom_range=0.05,     
    fill_mode='nearest'
)

# Create a data generator for validation and testing (only rescaling)
test_datagen = ImageDataGenerator(rescale=1./255)

# Create the data generators from directories
train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',        # For binary (Healthy/Parkinson) classification
    color_mode='grayscale'      # images are grayscale 
)

validation_generator = test_datagen.flow_from_directory(
    valid_path,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    color_mode='grayscale',
    shuffle=False               # Important for validation/testing to not shuffle 
)

test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    color_mode='grayscale',
    shuffle=False               # Important for evaluation 
)

print("Class indices:", train_generator.class_indices)


# Part 2: Constructing the Parkinson's Detection CNN ->

model = Sequential([
    # Input layer
    Conv2D(64, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)), # Start with 64 filters
    MaxPooling2D((2, 2)),
    
    # Second block
    Conv2D(128, (3, 3), activation='relu'), # Increased to 128 filters
    MaxPooling2D((2, 2)),
    Dropout(0.3), 

    # New third block
    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.4),

    # Classifier Head
    Flatten(),
    Dense(256, activation='relu'), # More neurons in the Dense layer
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

custom_optimizer = Adam(learning_rate=0.0001)

# Compile the model
model.compile(optimizer=custom_optimizer,       # Adam optimizer
              loss='binary_crossentropy',       # Loss function for binary classification
              metrics=['accuracy'])             # Monitor accuracy

# Print the model summary
model.summary()


# Part 3: The Training and Evaluation Protocol ->

# Define callbacks for training
model_checkpoint = ModelCheckpoint(
    'D:/PROJECTS/Deep Learning Projects/Parkinsons-Disease-Detection/best_model.h5',          # Filepath to save the model
    monitor='val_loss',         # Monitor validation loss
    save_best_only=True,        # Save only the best model
    verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,                 # Stop if no improvement after 5 epochs 
    restore_best_weights=True,  # Restore weights from the best epoch 
    verbose=1
)

# Train the model->
history = model.fit(
    train_generator,
    epochs=50,
    validation_data=validation_generator,
    callbacks=[model_checkpoint, early_stopping] # Use the defined callbacks
)


# Plot training & validation history
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(len(acc))


plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')


plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# Final Model Assessment on the Test Set
final_model = load_model('best_model.h5') 

# Evaluate the model on the test set
test_loss, test_acc = final_model.evaluate(test_generator, verbose=2)
print(f'\nTest accuracy: {test_acc:.4f}')

# Generate predictions
predictions = final_model.predict(test_generator)
predicted_classes = (predictions > 0.5).astype(int).flatten()

# Get true labels and class names
true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

# Print classification report
print('\nClassification Report:')
print(classification_report(true_classes, predicted_classes, target_names=class_labels))

# Display the confusion matrix
cm = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix')
plt.ylabel('Actual Class')
plt.xlabel('Predicted Class')
plt.show()

# Saved the final model for deployment
final_model.save('parkinsons_detection_model.h5')
print("\nFinal model saved to parkinsons_detection_model.h5")