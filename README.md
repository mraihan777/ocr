# Handwriting Recognition Application

Aplikasi web untuk mengenali tulisan tangan huruf A-Z menggunakan Deep Learning (CNN) dengan Python, TensorFlow, dan Flask.

## 🌟 Fitur

- ✍️ **Canvas Interaktif**: Gambar huruf langsung di browser menggunakan mouse atau touchscreen
- 📤 **Upload Gambar**: Upload gambar tulisan tangan dari perangkat Anda
- 🤖 **CNN Model**: Model Deep Learning yang akurat untuk klasifikasi huruf A-Z
- 📊 **Hasil Detail**: Menampilkan prediksi utama dengan confidence score dan top 3 prediksi
- 🎨 **UI Modern**: Interface yang menarik dengan dark mode dan animasi smooth
- 📱 **Responsive**: Bekerja dengan baik di desktop dan mobile

## 📋 Prerequisites

- Python 3.8 atau lebih tinggi
- pip (Python package manager)
- Dataset A-Z Handwritten Alphabets dari Kaggle

## 🚀 Instalasi

### 1. Clone atau Download Repository

```bash
cd "d:\Citra Digital (UAS)"
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Dataset

Dataset yang digunakan: **A-Z Handwritten Alphabets in CSV format**

**Link Download**: https://www.kaggle.com/datasets/sachinpatel21/az-handwritten-alphabets-in-csv-format

**Langkah download:**
1. Buka link di atas
2. Login ke Kaggle (buat akun jika belum punya)
3. Klik tombol "Download" untuk mendownload `A_Z Handwritten Data.csv`
4. Letakkan file `A_Z Handwritten Data.csv` di folder project ini (`d:\Citra Digital (UAS)`)

**Informasi Dataset:**
- Format: CSV
- Ukuran: ~375,000 sampel tulisan tangan
- Ukuran gambar: 28x28 pixel
- Jumlah kelas: 26 (A-Z)

## 📚 Cara Menggunakan

### Step 1: Prepare Data

Jalankan script untuk memproses dataset:

```bash
python prepare_data.py
```

Script ini akan:
- Load data dari CSV
- Normalize pixel values
- Split data menjadi training, validation, dan test sets (70-15-15)
- Save processed data ke folder `data/`

**Output:**
```
Dataset loaded: 372,450 sampel
Data split:
  Training: 222,382 sampel
  Validation: 55,534 sampel
  Test: 55,534 sampel
```

### Step 2: Train Model

Train CNN model dengan data yang sudah diproses:

```bash
python train.py
```

**Training Configuration:**
- Epochs: 30
- Batch size: 128
- Optimizer: Adam
- Data augmentation: Rotation, shift, zoom, shear

**Output:**
- Model tersimpan di folder `models/`
  - `best_model.h5` - Model dengan validation accuracy terbaik
  - `final_model.h5` - Model setelah training selesai
- Training history plot tersimpan sebagai `training_history.png`

**Expected Performance:**
- Training accuracy: >95%
- Validation accuracy: >90%
- Test accuracy: >88%

⏱️ **Waktu training**: ~15-30 menit (tergantung hardware)

### Step 3: Run Web Application

Jalankan Flask server:

```bash
python app.py
```

Buka browser dan akses:
```
http://localhost:5000
```

## 🎯 Cara Menggunakan Aplikasi

### Metode 1: Gambar di Canvas
1. Gunakan mouse atau touchscreen untuk menggambar huruf di canvas putih
2. Klik tombol **"Prediksi"** untuk mendapatkan hasil
3. Lihat prediksi huruf dengan confidence score
4. Klik **"Clear"** untuk menghapus dan mencoba huruf lain

### Metode 2: Upload Gambar
1. Klik tombol **"Upload Gambar"**
2. Pilih file gambar tulisan tangan dari perangkat Anda
3. Gambar akan muncul di canvas
4. Klik tombol **"Prediksi"** untuk mendapatkan hasil

**Tips untuk hasil terbaik:**
- Tulis huruf dengan jelas dan tegas
- Gunakan huruf kapital (A-Z)
- Pastikan huruf cukup besar dan terpusat
- Untuk upload, gunakan gambar dengan background putih dan tulisan hitam

## 📁 Struktur Project

```
d:\Citra Digital (UAS)\
├── model.py                 # CNN model architecture
├── prepare_data.py          # Data preprocessing script
├── train.py                 # Training script
├── app.py                   # Flask web application
├── requirements.txt         # Python dependencies
├── README.md               # Dokumentasi (file ini)
│
├── templates/
│   └── index.html          # HTML frontend
│
├── static/
│   ├── css/
│   │   └── style.css       # Styling
│   └── js/
│       └── script.js       # Canvas & prediction logic
│
├── data/                   # Processed dataset (generated)
│   ├── X_train.npy
│   ├── X_val.npy
│   ├── X_test.npy
│   ├── y_train.npy
│   ├── y_val.npy
│   └── y_test.npy
│
└── models/                 # Trained models (generated)
    ├── best_model.h5
    └── final_model.h5
```

## 🧠 Arsitektur Model

**CNN Architecture:**
```
Input (28x28x1)
    ↓
Conv2D (32) + BatchNorm + Conv2D (32) + BatchNorm + MaxPool + Dropout(0.25)
    ↓
Conv2D (64) + BatchNorm + Conv2D (64) + BatchNorm + MaxPool + Dropout(0.25)
    ↓
Conv2D (128) + BatchNorm + Conv2D (128) + BatchNorm + MaxPool + Dropout(0.4)
    ↓
Flatten
    ↓
Dense (512) + BatchNorm + Dropout(0.5)
    ↓
Dense (256) + BatchNorm + Dropout(0.5)
    ↓
Dense (26, softmax)
    ↓
Output (A-Z)
```

**Total Parameters**: ~4.5 million

## 🔧 Troubleshooting

### Error: "Model tidak ditemukan"
- Pastikan Anda sudah menjalankan `python train.py`
- Check folder `models/` ada file `best_model.h5`

### Error: "File A_Z Handwritten Data.csv tidak ditemukan"
- Download dataset dari Kaggle (lihat section Download Dataset)
- Pastikan file CSV ada di folder project

### Prediksi tidak akurat
- Coba tulis huruf lebih jelas dan besar
- Pastikan huruf terpusat di canvas
- Train model lebih lama (tingkatkan epochs)

### Server tidak bisa diakses
- Pastikan Flask server running (tidak ada error di terminal)
- Check firewall tidak block port 5000
- Coba akses `http://127.0.0.1:5000` jika `localhost` tidak work

## 🛠️ Tech Stack

- **Backend**: Python, Flask, TensorFlow/Keras
- **Frontend**: HTML5, CSS3, JavaScript (Vanilla)
- **Model**: Convolutional Neural Network (CNN)
- **Dataset**: Kaggle A-Z Handwritten Alphabets

## 📊 Performance Metrics

Setelah training selesai, Anda dapat melihat:
- Training/Validation accuracy plot di `training_history.png`
- Test accuracy di output terminal
- Real-time prediction confidence di web interface

## 📝 License

This project is for educational purposes.

## 👨‍💻 Author

Created for Citra Digital (UAS) Project

---

**Happy Coding! ✨**
