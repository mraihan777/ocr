"""
Script untuk download trained model dari Google Drive atau URL lain
Dijalankan otomatis saat aplikasi startup di Railway
"""

import os
import requests
from tqdm import tqdm

def download_file_from_google_drive(file_id, destination):
    """
    Download file dari Google Drive
    
    Args:
        file_id: Google Drive file ID
        destination: Path tujuan untuk save file
    """
    URL = "https://drive.google.com/uc?export=download"
    
    session = requests.Session()
    
    response = session.get(URL, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)
    
    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
    
    save_response_content(response, destination)

def get_confirm_token(response):
    """Get confirmation token dari Google Drive"""
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination):
    """Save content dari response ke file dengan progress bar"""
    CHUNK_SIZE = 32768
    
    # Get total file size if available
    total_size = int(response.headers.get('content-length', 0))
    
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    with open(destination, "wb") as f:
        if total_size == 0:
            # No content-length header
            print(f"Downloading to {destination}...")
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:
                    f.write(chunk)
        else:
            # Show progress bar
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=destination) as pbar:
                for chunk in response.iter_content(CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

def download_model(model_path='models/best_model.h5'):
    """
    Download model jika belum ada
    
    Args:
        model_path: Path ke model file
    
    Returns:
        bool: True jika model ada atau berhasil di-download
    """
    if os.path.exists(model_path):
        print(f"✓ Model sudah ada di {model_path}")
        return True
    
    print(f"Model tidak ditemukan di {model_path}")
    
    # Coba ambil URL dari environment variable
    model_url = os.environ.get('MODEL_DOWNLOAD_URL')
    google_drive_id = os.environ.get('MODEL_GDRIVE_ID')
    
    if google_drive_id:
        print(f"Downloading model dari Google Drive (ID: {google_drive_id})...")
        try:
            download_file_from_google_drive(google_drive_id, model_path)
            print(f"✓ Model berhasil di-download ke {model_path}")
            return True
        except Exception as e:
            print(f"✗ Gagal download dari Google Drive: {e}")
            return False
    
    elif model_url:
        print(f"Downloading model dari URL: {model_url}")
        try:
            response = requests.get(model_url, stream=True)
            response.raise_for_status()
            save_response_content(response, model_path)
            print(f"✓ Model berhasil di-download ke {model_path}")
            return True
        except Exception as e:
            print(f"✗ Gagal download dari URL: {e}")
            return False
    
    else:
        print("✗ Tidak ada MODEL_DOWNLOAD_URL atau MODEL_GDRIVE_ID di environment variables")
        print("\nCara setup:")
        print("1. Upload best_model.h5 ke Google Drive")
        print("2. Set file menjadi 'Anyone with the link can view'")
        print("3. Copy file ID dari URL (https://drive.google.com/file/d/FILE_ID/view)")
        print("4. Set environment variable di Railway: MODEL_GDRIVE_ID=your_file_id")
        print("\nAtau gunakan URL langsung:")
        print("5. Set environment variable di Railway: MODEL_DOWNLOAD_URL=your_url")
        return False

if __name__ == '__main__':
    download_model()
