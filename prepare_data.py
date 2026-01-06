"""
Script untuk memproses dataset A-Z Handwritten Alphabets dari Kaggle
Download manual dataset dari: https://www.kaggle.com/datasets/sachinpatel21/az-handwritten-alphabets-in-csv-format
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import pickle


def load_and_prepare_data(csv_path='A_Z Handwritten Data.csv', save_dir='data'):
    """
    Load dan prepare dataset dari CSV
    
    Args:
        csv_path: Path ke file CSV dataset
        save_dir: Directory untuk save processed data
    
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    print("Loading dataset dari CSV...")
    
    # Check if file exists
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"File {csv_path} tidak ditemukan!\n"
            "Silakan download dataset dari:\n"
            "https://www.kaggle.com/datasets/sachinpatel21/az-handwritten-alphabets-in-csv-format\n"
            "Dan letakkan file 'A_Z Handwritten Data.csv' di folder project ini."
        )
    
    # Load CSV
    df = pd.read_csv(csv_path)
    print(f"Dataset loaded: {df.shape[0]:,} sampel")
    
    # Separate features and labels
    # Column 0 adalah label (0-25 untuk A-Z)
    # Column 1-784 adalah pixel values (28x28 = 784)
    y = df.iloc[:, 0].values
    X = df.iloc[:, 1:].values
    
    # Reshape X dari (n, 784) menjadi (n, 28, 28, 1)
    X = X.reshape(-1, 28, 28, 1)
    
    # Normalize pixel values ke range [0, 1]
    X = X.astype('float32') / 255.0
    
    print(f"Data shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Labels range: {y.min()} to {y.max()}")
    
    # Convert labels ke categorical (one-hot encoding)
    from tensorflow.keras.utils import to_categorical
    y = to_categorical(y, num_classes=26)
    
    # Split data: 70% train, 15% validation, 15% test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y.argmax(axis=1)
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.176, random_state=42, 
        stratify=y_train_val.argmax(axis=1)  # 0.176 * 0.85 ≈ 0.15 of total
    )
    
    print(f"\nData split:")
    print(f"  Training: {X_train.shape[0]:,} sampel")
    print(f"  Validation: {X_val.shape[0]:,} sampel")
    print(f"  Test: {X_test.shape[0]:,} sampel")
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Save processed data
    print(f"\nSaving processed data ke folder '{save_dir}'...")
    np.save(os.path.join(save_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(save_dir, 'X_val.npy'), X_val)
    np.save(os.path.join(save_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(save_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(save_dir, 'y_val.npy'), y_val)
    np.save(os.path.join(save_dir, 'y_test.npy'), y_test)
    
    print("Data berhasil diproses dan disimpan!")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def load_processed_data(save_dir='data'):
    """
    Load processed data dari file .npy
    
    Args:
        save_dir: Directory tempat processed data disimpan
    
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    print(f"Loading processed data dari folder '{save_dir}'...")
    
    X_train = np.load(os.path.join(save_dir, 'X_train.npy'))
    X_val = np.load(os.path.join(save_dir, 'X_val.npy'))
    X_test = np.load(os.path.join(save_dir, 'X_test.npy'))
    y_train = np.load(os.path.join(save_dir, 'y_train.npy'))
    y_val = np.load(os.path.join(save_dir, 'y_val.npy'))
    y_test = np.load(os.path.join(save_dir, 'y_test.npy'))
    
    print(f"Data loaded:")
    print(f"  Training: {X_train.shape[0]:,} sampel")
    print(f"  Validation: {X_val.shape[0]:,} sampel")
    print(f"  Test: {X_test.shape[0]:,} sampel")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    # Process dataset
    try:
        X_train, X_val, X_test, y_train, y_val, y_test = load_and_prepare_data()
        
        # Show sample
        print("\n" + "="*50)
        print("Sample data:")
        print(f"First training image shape: {X_train[0].shape}")
        print(f"First training label: {y_train[0].argmax()} (Letter: {chr(65 + y_train[0].argmax())})")
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
