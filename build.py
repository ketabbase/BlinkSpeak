import PyInstaller.__main__
import os
import shutil
import sys
from PyInstaller.utils.hooks import copy_metadata

def build_executable():
    # Clean up previous builds
    if os.path.exists('dist'):
        shutil.rmtree('dist')
    if os.path.exists('build'):
        shutil.rmtree('build')

    # Get the path to the mediapipe package
    # Use sys.executable to ensure we are using the correct python environment
    try:
        import mediapipe
        mediapipe_path = os.path.dirname(mediapipe.__file__)
    except ImportError:
        print("Error: mediapipe not found. Please install it first.")
        sys.exit(1)

    # Define the source paths for mediapipe data files
    # Include the modules directory, which likely contains the models
    mediapipe_data_path = os.path.join(mediapipe_path, 'modules')

    # Collect streamlit metadata
    streamlit_metadata_datas = copy_metadata('streamlit')

    # Get absolute path for the icon file
    icon_path = os.path.abspath('app_icon.ico')

    # Build the eye tracking system executable
    a = PyInstaller.__main__.run([
        'eyetrackcam.py',
        '--name=BlinkSpeak',
        '--onefile',
        '--console',
        '--add-data=README.md;.',
        '--hidden-import=mediapipe',
        '--hidden-import=opencv-python',
        '--hidden-import=numpy',
        '--hidden-import=pandas',
        # Add mediapipe data files by including the modules directory
        '--add-data=' + mediapipe_data_path + ';mediapipe/modules',
        '--icon=%s' % icon_path,
    ])

    # Create installer directory
    if not os.path.exists('installer'):
        os.makedirs('installer')

    # Copy files to installer directory
    shutil.copy('dist/BlinkSpeak.exe', 'installer/')
    shutil.copy('README.md', 'installer/')

    print("\nBuild completed successfully!")
    print("Executable is located in the 'dist' directory")
    print("Installer files are located in the 'installer' directory")

if __name__ == "__main__":
    build_executable() 