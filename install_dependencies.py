import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def upgrade_pip():
    # Upgrade pip first
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])

def upgrade_setuptools():
    # Upgrade setuptools (which also includes distutils)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "setuptools"])

def main():
    # Upgrade pip
    upgrade_pip()
    
    # Upgrade setuptools
    upgrade_setuptools()
    
    # Install required packages
    packages = [
        'streamlit==1.37.1',
        'gdown==5.2.0',
        'Pillow==9.5.0',
        'pandas==1.5.3',
        'scikit-learn==1.3.2',
        'scikit-surprise==1.1.4',
        'numpy==1.24.3',
        'dask==2023.5.0',
        'fuzzywuzzy==0.18.0',
        'boto3==1.35.0',
        'python-levenshtein==0.25.1',
        "joblib==1.4.2"
    ]
    
    for package in packages:
        install(package)

if __name__ == "__main__":
    main()
