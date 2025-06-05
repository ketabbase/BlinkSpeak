# BlinkSpeak

A professional blink and mouth movemen detection system.
To purchase the software (no coding needs) contact: mahsatorabi515@gmail.com

![icon](https://github.com/user-attachments/assets/655d4376-3219-45ae-984d-fcee827244e3)


## Features

- Real-time blink detection using webcam
- Mouth movement tracking
- Research-grade data collection and analysis

## Installation

### Windows (Easiest Method)

1. Download the latest release from the [Releases page](https://github.com/ketabbase/BlinkSpeak)
2. Run the installer (BlinkSpeak-Setup.exe)
3. Follow the installation wizard
4. The application will be installed with a desktop shortcut

### Windows (Manual Installation)

1. Download the latest release
2. Extract the zip file
3. Double-click `BlinkSpeak.exe` to run the tracking system

### From Source (For Developers)

1. Clone the repository:
```bash
git clone https://github.com/ketabbase/BlinkSpeak.git
cd BlinkSpeak
```

2. Install the package:
```bash
pip install -e .
```

3. Run the application:
```bash
blinkspeak
```

## Usage

### Tracking System

1. Launch the tracking system:
   - Double-click `BlinkSpeak.exe` (Windows installer)
   - Or run `blinkspeak` (command line)

2. Controls:
- Press 'R' to start recording
- Press 'S' to stop recording
- Press 'Q' to quit

## System Requirements

- Windows 10 or later
- Webcam
- 4GB RAM minimum
- 500MB free disk space

## Data Files

The system generates the following data files in the current directory:
- `blinks.csv`: Blink detection data
- `mouth_movements.csv`: Mouth movement data

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support, please open an issue on the [GitHub repository](https://github.com/ketabbase/BlinkSpeak/issues). 
