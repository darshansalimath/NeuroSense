import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import kagglehub

# Define paths and hyperparameters
BATCH_SIZE = 32
IMAGE_SIZE = (256, 256)
EPOCHS = 20

POTATO_CLASSES = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

def build_model(input_shape, num_classes):
    """Builds a Convolutional Neural Network (CNN) model."""
    model = models.Sequential([
        layers.Input(shape=input_shape),
        # Rescaling pixel values to 0-1
        layers.Rescaling(1./255),
        
        # Convolutional Block 1
        layers.Conv2D(32, kernel_size=(3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        
        # Convolutional Block 2
        layers.Conv2D(64, kernel_size=(3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        
        # Convolutional Block 3
        layers.Conv2D(64, kernel_size=(3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        
        # Convolutional Block 4
        layers.Conv2D(128, kernel_size=(3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        
        # Flattening and Dense Layers
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5), # Dropout to prevent overfitting
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )
    
    return model

def main():
    print("Downloading/Locating dataset via kagglehub...")
    path = kagglehub.dataset_download("arjuntejaswi/plant-village")
    
    # The dataset usually contains a 'PlantVillage' subfolder
    dataset_dir = os.path.join(path, 'PlantVillage')
    if not os.path.exists(dataset_dir):
        dataset_dir = path # Fallback to the root path if the subfolder isn't there
        
    print(f"Dataset ready at: {dataset_dir}")

    # Data Augmentation for training
    train_datagen = ImageDataGenerator(
        validation_split=0.2, # Use 20% of the data for validation
        rotation_range=20,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.2
    )

    print("Loading Training Data...")
    train_generator = train_datagen.flow_from_directory(
        dataset_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='sparse',
        classes=POTATO_CLASSES,
        subset='training',
        shuffle=True
    )

    print("Loading Validation Data...")
    val_generator = train_datagen.flow_from_directory(
        dataset_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='sparse',
        classes=POTATO_CLASSES,
        subset='validation',
        shuffle=False
    )

    class_names = list(train_generator.class_indices.keys())
    print(f"Detected classes: {class_names}")
    
    # Save class names for the Streamlit app
    with open('class_names.txt', 'w') as f:
        f.write('\n'.join(class_names))

    # Build the model
    input_shape = (IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
    model = build_model(input_shape, num_classes=len(class_names))
    model.summary()

    # Train the model
    print("Starting Model Training...")
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator
    )

    # Save the trained model
    model.save('potato_disease_model.h5')
    print("Model saved to 'potato_disease_model.h5'")

    # Plot Training History
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs_range = range(EPOCHS)

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
    plt.savefig('training_history.png')
    print("Training history plot saved as 'training_history.png'")

    # Evaluate the model on validation set
    print("Evaluating Model...")
    # Reset validation generator before predicting
    val_generator.reset()
    predictions = model.predict(val_generator)
    y_pred = np.argmax(predictions, axis=1)
    y_true = val_generator.classes

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    print("Confusion matrix saved as 'confusion_matrix.png'")

if __name__ == '__main__':
    main()
