import json
import platform
from typing import List, Dict
from LEPAUTE import get_collected_data
import logging

logger = logging.getLogger(__name__)

def load_data(file_path: str = None, format: str = "json") -> List[Dict]:
    try:
        data = get_collected_data()
        if data:
            return data
        
        if file_path and platform.system() != "Emscripten":
            if format == "json":
                with open(file_path, "r") as f:
                    return json.load(f)
        return []
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return []