"""
Script untuk download dataset A-Z Handwriting dari Kaggle secara otomatis
Mendukung 2 metode: kagglehub dan kaggle API
"""

import os
import sys

def check_kaggle_credentials():
    """Check apakah Kaggle credentials sudah di-setup"""
    kaggle_json_path = os.path.expanduser('~/.kaggle/kaggle.json')
    if os.path.exists(kaggle_json_path):
        print("✓ Kaggle credentials ditemukan")
        return True
    else:
        print("✗ Kaggle credentials tidak ditemukan")
        return False


def download_with_kagglehub():
    """Download menggunakan kagglehub (metode baru)"""
    try:
        print("\n[Method 1] Downloading dengan kagglehub...")
        import kagglehub
        
        # Download dataset A-Z Handwritten Alphabets
        path = kagglehub.dataset_download("sachinpatel21/az-handwritten-alphabets-in-csv-format")
        
        print(f"✓ Dataset downloaded to: {path}")
        
        # Find CSV file
        csv_file = None
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith('.csv'):
                    csv_file = os.path.join(root, file)
                    break
            if csv_file:
                break
        
        if csv_file:
            # Copy to current directory
            import shutil
            dest = os.path.join(os.getcwd(), 'A_Z Handwritten Data.csv')
            shutil.copy(csv_file, dest)
            print(f"✓ CSV file copied to: {dest}")
            return True
        else:
            print("✗ CSV file tidak ditemukan di downloaded dataset")
            return False
            
    except ImportError:
        print("✗ kagglehub tidak terinstall")
        print("  Install dengan: pip install kagglehub")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def download_with_kaggle_api():
    """Download menggunakan Kaggle API tradisional"""
    try:
        print("\n[Method 2] Downloading dengan Kaggle API...")
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        api = KaggleApi()
        api.authenticate()
        
        # Download dataset
        print("Downloading dataset...")
        api.dataset_download_files(
            'sachinpatel21/az-handwritten-alphabets-in-csv-format',
            path='.',
            unzip=True
        )
        
        print("✓ Dataset downloaded successfully!")
        
        # Check if CSV exists
        csv_file = 'A_Z Handwritten Data.csv'
        if os.path.exists(csv_file):
            print(f"✓ CSV file ready: {csv_file}")
            return True
        else:
            print("✗ CSV file tidak ditemukan setelah download")
            return False
            
    except ImportError:
        print("✗ kaggle tidak terinstall")
        print("  Install dengan: pip install kaggle")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def show_manual_instructions():
    """Tampilkan instruksi manual download"""
    print("\n" + "="*70)
    print("MANUAL DOWNLOAD INSTRUCTIONS")
    print("="*70)
    print("\nKarena download otomatis gagal, silakan download manual:")
    print("\n1. Buka link ini di browser:")
    print("   https://www.kaggle.com/datasets/sachinpatel21/az-handwritten-alphabets-in-csv-format")
    print("\n2. Login ke Kaggle (buat akun gratis jika belum punya)")
    print("\n3. Klik tombol 'Download' di pojok kanan atas")
    print("\n4. File 'A_Z Handwritten Data.csv' akan terdownload")
    print("\n5. Pindahkan file tersebut ke folder ini:")
    print(f"   {os.getcwd()}")
    print("\n6. Setelah itu, jalankan: python prepare_data.py")
    print("="*70)


def show_kaggle_api_setup():
    """Tampilkan cara setup Kaggle API credentials"""
    print("\n" + "="*70)
    print("SETUP KAGGLE API CREDENTIALS")
    print("="*70)
    print("\nUntuk download otomatis, Anda perlu Kaggle API credentials:")
    print("\n1. Login ke Kaggle.com")
    print("\n2. Klik profile picture Anda (pojok kanan atas)")
    print("\n3. Pilih 'Settings'")
    print("\n4. Scroll ke bagian 'API'")
    print("\n5. Klik 'Create New Token'")
    print("\n6. File 'kaggle.json' akan terdownload")
    print("\n7. Letakkan file kaggle.json di:")
    if os.name == 'nt':  # Windows
        kaggle_dir = os.path.expanduser('~/.kaggle/')
        print(f"   {kaggle_dir}")
        print("\n   Buat folder .kaggle jika belum ada:")
        print(f"   mkdir {kaggle_dir}")
    else:  # Linux/Mac
        print("   ~/.kaggle/")
        print("\n   Buat folder .kaggle jika belum ada:")
        print("   mkdir -p ~/.kaggle/")
        print("   chmod 600 ~/.kaggle/kaggle.json")
    
    print("\n8. Install library yang diperlukan:")
    print("   pip install kagglehub kaggle")
    print("\n9. Jalankan script ini lagi: python download_dataset.py")
    print("="*70)


def main():
    print("="*70)
    print("KAGGLE DATASET DOWNLOADER")
    print("Dataset: A-Z Handwritten Alphabets")
    print("="*70)
    
    # Check if CSV already exists
    csv_file = 'A_Z Handwritten Data.csv'
    if os.path.exists(csv_file):
        print(f"\n✓ Dataset sudah ada: {csv_file}")
        print("Anda bisa langsung jalankan: python prepare_data.py")
        return
    
    # Check credentials
    has_credentials = check_kaggle_credentials()
    
    if not has_credentials:
        print("\nKaggle API credentials tidak ditemukan.")
        print("Pilih salah satu:")
        print("  [1] Setup Kaggle API credentials dulu (recommended)")
        print("  [2] Download manual")
        
        choice = input("\nPilihan Anda (1/2): ").strip()
        
        if choice == '1':
            show_kaggle_api_setup()
        else:
            show_manual_instructions()
        return
    
    # Try automatic download
    print("\nMencoba download otomatis...")
    
    # Try kagglehub first
    success = download_with_kagglehub()
    
    # If failed, try kaggle API
    if not success:
        success = download_with_kaggle_api()
    
    # If still failed, show manual instructions
    if not success:
        show_manual_instructions()
    else:
        print("\n" + "="*70)
        print("✓ DOWNLOAD BERHASIL!")
        print("="*70)
        print("\nLangkah selanjutnya:")
        print("1. python prepare_data.py")
        print("2. python train.py")
        print("3. python app.py")
        print("="*70)


if __name__ == "__main__":
    main()
