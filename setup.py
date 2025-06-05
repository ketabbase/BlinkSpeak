from setuptools import setup, find_packages

setup(
    name="blinkspeak",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "opencv-python>=4.5.0",
        "numpy>=1.26.4",
        "pandas>=2.2.1",
        "mediapipe>=0.8.9",
        "streamlit>=1.32.0",
        "plotly>=5.19.0"
    ],
    entry_points={
        'console_scripts': [
            'blinkspeak=eyetrackcam:main',
        ],
    },
    author="Your Name",
    author_email="your.email@example.com",
    description="A professional blink detection and speech analysis system",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/BlinkSpeak",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Microsoft :: Windows",
    ],
    python_requires=">=3.8",
) 