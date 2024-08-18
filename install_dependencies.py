import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def upgrade_pip():
    # Upgrade pip first
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])

def ensure_setuptools():
    try:
        import setuptools
    except ImportError:
        install('setuptools')

def main():
    # Ensure setuptools is installed
    ensure_setuptools()

    # Upgrade pip
    upgrade_pip()
    
    # Install required packages
    packages = [
        'Pillow==9.5.0',
        'streamlit==1.25.0',
        'gdown==5.2.0',
        'pandas==1.5.3',  # Compatible version for Python 3.8
        'scikit-learn==1.3.2',
        'scikit-surprise==1.1.4',
        'numpy==1.24.3',
        'dask==2023.5.0',
        'fuzzywuzzy==0.18.0',
        'setuptools==68.0.0',
        'boto3==1.35.0',
        'python-levenshtein==0.25.1',
        "joblib==1.4.2"
    ]
    
    for package in packages:
        install(package)

if __name__ == "__main__":
    main()
