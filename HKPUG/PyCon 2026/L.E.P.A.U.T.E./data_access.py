import json
import platform
from typing import List, Dict
from utils import get_collected_data
from config import logger

def load_data(file_path: str = "lepaute_data.json", format: str = "json") -> List[Dict]:
    """Load pipeline data from memory or file."""
    try:
        data = get_collected_data()
        if data:
            return data
        
        if file_path and platform.system() != "Emscripten":
            if format == "json":
                with open(file_path, "r") as f:
                    return json.load(f)
        logger.warning(f"No data found in memory or file: {file_path}")
        return []
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        return []