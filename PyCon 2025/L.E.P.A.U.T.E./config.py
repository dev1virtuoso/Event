from collections import defaultdict
import torch
import logging
import platform
from typing import Dict
import kornia as K

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s:%(name)s:%(levelname)s:%(message)s',
    handlers=[
        logging.FileHandler("lepaute_debug.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Check for pytorch-metric-learning
try:
    from pytorch_metric_learning import losses
    HAS_PML = True
except ImportError:
    HAS_PML = False
    logger.warning("pytorch-metric-learning not found. Using MSE fallback. Install with `pip install pytorch-metric-learning`.")

# Global variables
DATA_STORE: Dict = defaultdict(list)  # Stores pipeline data
DTYPE: torch.dtype = torch.float32  # Tensor data type
OBJECT_NAMES: Dict[int, str] = {
    0: "Book", 1: "Pen", 2: "Phone", 3: "Mug", 4: "Keyboard",
    5: "Mouse", 6: "Notebook", 7: "Bottle", 8: "Glasses", 9: "Wallet",
    10: "Chair", 11: "Table",
}  # Object label mappings

def get_device() -> torch.device:
    """Detect available GPU for acceleration."""
    if torch.cuda.is_available():
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    elif torch.backends.mps.is_available() and platform.system() == "Darwin":
        logger.info("Using MPS device for acceleration.")
        return torch.device("mps")
    logger.info("Using CPU as no suitable GPU found.")
    return torch.device("cpu")

DEVICE: torch.device = get_device()  # Device for computation