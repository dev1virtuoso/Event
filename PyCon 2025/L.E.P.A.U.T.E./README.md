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

### Modes

1. **GUI/Realtime Mode**:
   ```bash:disable-run
   python LEPAUTE.py
   ```
   - Displays the camera feed with parameters shown in the top-right corner.
   - Shows a red border for rotation and a blue border for translation.
   - Saves data to `lepaute_data.json`.
   - Saves images to the `frames` directory.

2. **JSON Mode**:
   ```python
   from LEPAUTE import run_main
   run_main(display_mode="json", save_json=True)
   ```
   - Does not display a window; saves data to `lepaute_data.json`.

3. **Save Images**:
   ```python
   from LEPAUTE import run_main
   run_main(display_mode="gui,realtime", save_image=True, frames_dir="frames")
   ```
   - Saves images to the `frames` directory.

4. **Using `example.py`**:
   ```bash
   python example.py
   ```
   - Runs in realtime mode, saves JSON and images, and prints collected data.

5. **Using `example_pip.py`**:
   ```bash
   python example_pip.py
   ```
   - Runs with default settings and prints Lie parameters and loss.

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
- Due to priority issues, the following issues will be addressed as soon as possible. It is expected that it will take a month to implement all the following functions. At the same time, there are some minor issues in the current code, which I hope to fix on or before Friday. The issues include: 
  - data dependency issues. The current implementation relies on placeholder datasets (such as randomly generated validation datasets) and cyclic labels, lacking support from real data, which limits the effectiveness of training and performance in actual applications. The code also clearly states that it needs to be replaced with real data, otherwise the evaluation results may not be representative. 
  - Functional implementation is incomplete. Some functions (such as validation dataset generation and label assignment) are temporary and have not yet been fully implemented. For example, the training process uses simple cyclic labels, which cannot reflect the complexity of real scenes. In addition, support for SE(3) or Sim(3) does not seem to be fully implemented, and is limited to SE(2). 
  - Error handling basics. Although there is basic error logging, the handling of some key errors (such as the camera cannot be turned on, low-variance frame skipping) is relatively simple, lacking a more robust response strategy, which may affect system stability.
