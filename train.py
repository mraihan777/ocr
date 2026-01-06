"""
Training script untuk Handwriting Recognition Model
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from model import create_model
from prepare_data import load_processed_data
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


def plot_training_history(history, save_path='training_history.png'):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nTraining history plot saved to: {save_path}")
    plt.close()


def train_model(epochs=30, batch_size=128):
    """
    Train handwriting recognition model
    
    Args:
        epochs: Number of training epochs
        batch_size: Batch size for training
    """
    print("="*60)
    print("HANDWRITING RECOGNITION - TRAINING")
    print("="*60)
    
    # Load data
    print("\n[1/5] Loading processed data...")
    try:
        X_train, X_val, X_test, y_train, y_val, y_test = load_processed_data()
    except Exception as e:
        print(f"\nError loading data: {e}")
        print("Pastikan Anda sudah menjalankan 'python prepare_data.py' terlebih dahulu!")
        return
    
    # Create model
    print("\n[2/5] Creating CNN model...")
    model = create_model()
    print(f"Model created with {model.count_params():,} parameters")
    
    # Setup callbacks
    print("\n[3/5] Setting up callbacks...")
    os.makedirs('models', exist_ok=True)
    
    callbacks = [
        ModelCheckpoint(
            'models/best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Data augmentation
    print("\n[4/5] Setting up data augmentation...")
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        shear_range=0.1
    )
    datagen.fit(X_train)
    
    # Train model
    print("\n[5/5] Training model...")
    print(f"Epochs: {epochs}, Batch size: {batch_size}")
    print("-" * 60)
    
    start_time = datetime.now()
    
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=batch_size),
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    end_time = datetime.now()
    training_time = (end_time - start_time).total_seconds()
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED!")
    print("="*60)
    print(f"Training time: {training_time/60:.2f} minutes")
    
    # Evaluate on test set
    print("\n[EVALUATION] Testing on test set...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy*100:.2f}%")
    
    # Save final model
    model.save('models/final_model.h5')
    print("\nModel saved:")
    print("  - Best model: models/best_model.h5")
    print("  - Final model: models/final_model.h5")
    
    # Plot training history
    plot_training_history(history)
    
    # Print final metrics
    print("\n" + "="*60)
    print("FINAL METRICS")
    print("="*60)
    best_val_acc = max(history.history['val_accuracy'])
    best_epoch = history.history['val_accuracy'].index(best_val_acc) + 1
    print(f"Best Validation Accuracy: {best_val_acc*100:.2f}% (Epoch {best_epoch})")
    print(f"Final Test Accuracy: {test_accuracy*100:.2f}%")
    print("="*60)
    
    return model, history


if __name__ == "__main__":
    # Train model
    model, history = train_model(epochs=10, batch_size=128)
