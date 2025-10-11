import asyncio
import os
import cv2
import numpy as np
from main import main
from utils import get_collected_data
from data_access import load_data
import multiprocessing
import logging
import kornia as K

logger = logging.getLogger(__name__)

async def run_pipeline_and_access_data(display_mode: str = "realtime", frames_dir: str = "frames", save_json: bool = True, save_image: bool = True):
    print(f"Resource usage: Dynamic adjustment to maintain 10 FPS, targeting ~50% CPU ({multiprocessing.cpu_count()//2} threads initially).")
    print("Monitor usage in Activity Monitor (macOS) under CPU tab.")
    print(f"\nStarting the LEPAUTE pipeline in {display_mode} mode...")
    if save_image:
        print(f"Saving frames to {frames_dir}...")
    
    # Run the main pipeline with real-time display
    await main(display_mode=display_mode, frames_dir=frames_dir, unlimited=True, save_json=save_json, save_image=save_image)
    
    # Access and print collected data
    print("\nAccessing collected data from memory...")
    data = get_collected_data()
    
    print(f"Total data entries: {len(data)}")
    if len(data) == 0:
        print("No data collected. Check webcam, feature extraction, or image texture.")
    for i, item in enumerate(data):
        print(f"\nData entry {i + 1}:")
        print(f"  Image1 shape: {item['image1'].shape}")
        print(f"  Image2 shape: {item['image2'].shape}")
        print(f"  SO(2) theta: {item['lie_params'][0][0]:.4f}")
        print(f"  SE(2) params (tx, ty): ({item['lie_params'][0][1]:.2f}, {item['lie_params'][0][2]:.2f})")
        print(f"  Model output: {[f'{x:.2f}' for x in item['output'][0][:3]]}...")
        print(f"  Loss: {item['loss']:.4f}")
        print(f"  Label: {item['label']}")
        print(f"  Detected Object: {item['detected_object']}")

    if display_mode == "json" and save_json:
        print("\nAccessing data from file...")
        file_data = load_data("lepaute_data.json")
        print(f"Total file data entries: {len(file_data)}")
        if len(file_data) == 0:
            print("No data in file. Ensure pipeline ran successfully in json mode.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    frames_dir = "frames"
    os.makedirs(frames_dir, exist_ok=True)
    
    mode = "realtime"
    save_json = True
    save_image = True
    
    print(f"Running in {mode.upper()} mode...")
    print(f"Save JSON: {save_json}, Save Image: {save_image}")
    asyncio.run(run_pipeline_and_access_data(display_mode=mode, frames_dir=frames_dir, save_json=save_json, save_image=save_image))