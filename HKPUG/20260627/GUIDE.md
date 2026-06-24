# L.E.P.A.U.T.E. Installation and Usage Guide

This guide provides step-by-step instructions for setting up, configuring, and running the L.E.P.A.U.T.E. Framework, including both inference (GUI mode) and original dataset training.

## System Requirements

| Parameter | Minimum Requirements | Recommended Requirements |
|---|---|---|
| Operating System | Linux, macOS, or Windows | Linux or macOS                               |
| Python Version | 3.8+ | 3.10–3.12 |
| PyTorch Version | >= 2.3.0 | 2.4+ |
| Memory (RAM) | 8 GB | 16 GB or more |
| Processor (CPU) | 4 cores | 6–8 cores or more |
| Graphics (GPU) | None required (CPU execution supported) | Nvidia GPU or Apple Silicon GPU (Avoid Intel)* |
| Storage Space | ~400–500 MB (for main pipeline only) | 90 GB (for training pipeline) |
| Network | Active connection required | Stable broadband connection |

This is a clean Markdown table based on the provided data. Copy and paste it directly into your `.md` file.

> * **Note on Apple Silicon:** The PyTorch MPS (Metal Performance Shaders) backend is supported but may display instability depending on your specific environment environment.

## Installation and Setup

### 1. Environment Configuration

Navigate to the project directory, initialize a isolated python virtual environment, and install the required dependencies.

```bash
cd L.E.P.A.U.T.E.
python -m venv lepaute
```

Activate the environment based on your operating system:

* **Linux / macOS:**
```bash
source lepaute/bin/activate
```
*   **Windows (Command Prompt):**
```cmd
lepaute\Scripts\activate.bat
```

* **Windows (PowerShell):**
```powershell
.\lepaute\Scripts\Activate.ps1
```

Once activated, install the environment dependencies:

```bash
pip install -r requirements.txt
```

### 2. Asset Procurement

Download the required model and dataset elements from the official release page:

* **Release Link:** [20260627 hands on assets](https://github.com/dev1virtuoso/Event/releases/tag/20260627)

Ensure you download all three essential files:

1. `siglip-model`
2. `best_model.pth`
3. `lepaute_ycbv_test`

### 3. File Deployment and Extraction

Extract the files locally and place them into their designated project directories:

* **`siglip-model`**

* *Manual Option:* Extract and move the directory to `~/.cache/huggingface/hub/` (Linux/macOS).
* *Automatic Option:* Leave the network connected and run `main.py` directly to handle setup automatically.

* **`best_model.pth`**
* Extract and place inside: `L.E.P.A.U.T.E./checkpoints/`

* **`lepaute_ycbv_test`**
* Extract and place inside the root project directory: `L.E.P.A.U.T.E./`

## Running the Application

Once the asset pipeline configuration is complete, execute the main runtime file to launch the Graphical User Interface.

```bash
python main.py --mode gui
```

## Dataset Training (Optional)

Follow this section if you wish to build the model from scratch using the original dataset.

### Training Prerequisites

* **Network:** Persistent, high-speed connection for asset downloads.
* **Storage Space:** Minimum **90 GB** available local disk space.
* **Memory (RAM):** 16 GB minimum (32 GB or higher strongly recommended).
* **Hardware:** Nvidia GPU or Apple Silicon hardware setup recommended.

### Training Pipeline Execution Steps

#### Step 1: Purge Pre-trained Weights

Remove conflicting weight records from your checkpoints folder.

```bash
rm L.E.P.A.U.T.E./checkpoints/best_model.pth
```

#### Step 2: Download YCB-V Data

```bash
python ycb-v_download.py
```

#### Step 3: Format and Preprocess Dataset

Convert the raw BOP format data into the project native file tree.

```bash
python convert_bop_to_lepaute.py --bop_dir ./bop_datasets/ycbv --output_dir ./lepaute_dataset
```

#### Step 4: Initiate Model Training

Run the training engine against the generated dataset paths.

```bash
python train.py --dataset_dir ./lepaute_dataset --checkpoint_dir ./checkpoints --epochs 15
```