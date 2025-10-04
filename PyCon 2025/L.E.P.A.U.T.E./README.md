# L.E..A.U.T.E.

A Python package for processing webcam images with a Lie group-based Transformer model and accessing the resulting data.

## Installation

### Install from The Python Package Index (PyPI)(currently unavailable)

```
pip3 install lepaute
```

### Install from source(recommended)

1. Make it executable:
```
chmod +x install_dependencies.sh
```

2. Run the script:
```
./install_dependencies.sh
```

### Check system equirement

1. Make it executable:
```
chmod +x check_system_requirements.sh
```

2. Run the script:
```
./check_system_requirements.sh
```

## Usage

### Install from The Python Package Index (PyPI)(currently unavailable)

1. Run the [example_pip.py](example_pip.py) file.
```
python3 example_pip.py
```

### Install from source


1. Run the [example.py](example.py) file.
```
python3 example.py
```

## Requirements

- "torch>=2.3.0",
- "kornia>=0.7.2",
- "opencv-python>=4.8.0",
- "numpy>=1.24.0",
- "torchvision>=0.15.0",
- "pytorch-metric-learning>=2.0.0",
- "scipy>=1.10.0",
- "matplotlib>=3.8.0",

## Notes

- Ensure webcam access for real-time data collection.
- In Pyodide, data is stored in memory only.
- Debug logs can be enabled by setting `logging.basicConfig(level=logging.DEBUG)`.
