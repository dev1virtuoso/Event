import torch
import cv2
import asyncio
import argparse
import os
import time
from retrying import retry
from config import DEVICE, logger
from data import DataPreprocessing, DataAugmentation, load_real_dataset
from training import PreTraining, Training, Hyperparameter
from models import TransformerModel
from utils import DynamicResourceManager, MultiModalProcessor, display_raw_frame, display_or_save_data
from config import OBJECT_NAMES

@retry(stop_max_attempt_number=3, wait_fixed=1000)
def open_camera(index: int = 0) -> cv2.VideoCapture:
    """Attempt to open camera with retries."""
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    logger.info(f"Camera opened successfully. Settings: width={cap.get(cv2.CAP_PROP_FRAME_WIDTH)}, height={cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}, fps={cap.get(cv2.CAP_PROP_FPS)}")
    return cap

async def main(display_mode: str = "json", frames_dir: str = "frames", unlimited: bool = False, save_json: bool = False, save_image: bool = False, group_type: str = "SE(2)", data_dir: str = None, use_cifar10: bool = False):
    modes: List[str] = [m.strip().lower() for m in display_mode.split(',')]
    show_window: bool = 'gui' in modes or 'realtime' in modes
    save_to_json: bool = 'json' in modes or save_json
    
    if save_image:
        os.makedirs(frames_dir, exist_ok=True)
    
    cap: cv2.VideoCapture = open_camera()
    
    FPS: int = 10
    resource_manager: DynamicResourceManager = DynamicResourceManager(target_fps=FPS)
    frame_count: int = 0
    prev_frame: Optional[torch.Tensor] = None
    train_dataset: List = load_real_dataset(data_dir=data_dir, use_cifar10=use_cifar10, num_samples=100)
    val_dataset: List = load_real_dataset(data_dir=data_dir, use_cifar10=use_cifar10, num_samples=20)
    
    preprocess: DataPreprocessing = DataPreprocessing()
    augment: DataAugmentation = DataAugmentation()
    multi_modal: MultiModalProcessor = MultiModalProcessor()
    model: TransformerModel = TransformerModel(group_type=group_type).to(DEVICE)
    pre_training: PreTraining = PreTraining(model, train_dataset, val_dataset)
    training: Training = Training(pre_training)
    hyperparam: Hyperparameter = Hyperparameter(training)
    
    logger.info(f"Running in {'REALTIME' if show_window else 'JSON'} mode...")
    logger.info(f"Save JSON: {save_to_json}, Save Image: {save_image}")
    logger.info(f"Resource usage: Dynamic adjustment to maintain {FPS} FPS, targeting ~50% CPU ({resource_manager.current_threads} threads initially).")
    logger.info("Monitor usage in Activity Monitor (macOS) under CPU tab.")
    logger.info(f"Starting the LEPAUTE pipeline in {'realtime' if show_window else 'json'} mode...")
    if save_image:
        logger.info(f"Saving frames to {frames_dir}...")
    
    try:
        best_lr: float = await hyperparam.tune([1e-5, 1e-4, 1e-3])
        logger.info(f"Best learning rate: {best_lr}")
    except Exception as e:
        logger.error(f"Hyperparameter tuning failed: {e}")
        return

    try:
        label_counter: int = 0
        while unlimited or frame_count < FPS * 10:
            start_time: float = time.time()
            ret, frame = cap.read()
            if not ret:
                logger.warning(f"Frame {frame_count}: Failed to capture frame from camera")
                if save_to_json:
                    display_or_save_data({}, save_json=True, frame_count=frame_count, frame=None)
                frame_count += 1
                continue
            logger.debug(f"Frame {frame_count}: Captured frame shape: {frame.shape}")
            frame = preprocess.normalize_2d(frame)
            logger.debug(f"Frame {frame_count}: Preprocessed frame shape: {frame.shape}")
            frame = multi_modal.process(frame)
            
            if prev_frame is not None:
                frame1 = prev_frame
                frame2 = frame
                try:
                    if frame1.std() < 0.02 or frame2.std() < 0.02:
                        logger.debug(f"Frame {frame_count}: Skipping low-variance frame")
                        if save_to_json:
                            display_or_save_data({}, save_json=True, frame_count=frame_count, frame=frame.cpu().numpy())
                        continue
                    
                    frame1_aug = augment.apply(frame1)
                    frame2_aug = augment.apply(frame2)
                    label = torch.tensor([label_counter % len(OBJECT_NAMES)], device=DEVICE, dtype=torch.long)
                    train_dataset.append((frame1_aug, frame2_aug, label))
                    loss, data = await pre_training.train_step(frame1_aug, frame2_aug, label)
                    logger.debug(f"Frame {frame_count}: Train step completed, loss: {loss:.4f}, data: {data}")
                    if data:
                        frame_np = (frame.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                        frame_np = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
                        if show_window:
                            display_raw_frame(frame_np, frame_count, data)
                        if save_to_json:
                            display_or_save_data(data, save_json=True, frame_count=frame_count, frame=frame_np)
                    else:
                        logger.warning(f"Frame {frame_count}: No data returned from train_step")
                        if save_to_json:
                            display_or_save_data({}, save_json=True, frame_count=frame_count, frame=frame.cpu().numpy())
                    label_counter += 1
                except Exception as e:
                    logger.warning(f"Frame {frame_count}: Frame processing failed: {e}")
                    if save_to_json:
                        display_or_save_data({}, save_json=True, frame_count=frame_count, frame=frame.cpu().numpy())
                    continue
            
            if save_image:
                try:
                    frame_path = os.path.join(frames_dir, f"frame_{frame_count:04d}.png")
                    frame_np = (frame.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    frame_np = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(frame_path, frame_np)
                    logger.debug(f"Frame {frame_count}: Saved image to {frame_path}")
                except Exception as e:
                    logger.error(f"Frame {frame_count}: Failed to save image to {frame_path}: {e}")
            
            prev_frame = frame
            frame_count += 1
            frame_time = time.time() - start_time
            dynamic_delay = resource_manager.update(frame_time)
            await asyncio.sleep(max(1.0 / FPS - frame_time, 0) + dynamic_delay)
    
    except Exception as e:
        logger.error(f"Main loop error: {e}")
    
    finally:
        if 'cap' in locals():
            cap.release()
        if show_window:
            cv2.destroyAllWindows()
        if str(DEVICE).startswith("mps"):
            torch.mps.empty_cache()

def run_main(display_mode: str = "json", frames_dir: str = "frames", unlimited: bool = False, save_json: bool = False, save_image: bool = False, group_type: str = "SE(2)", data_dir: str = None, use_cifar10: bool = False):
    loop = asyncio.get_event_loop()
    try:
        if loop.is_running():
            loop.create_task(main(display_mode=display_mode, frames_dir=frames_dir, unlimited=unlimited, save_json=save_json, save_image=save_image, group_type=group_type, data_dir=data_dir, use_cifar10=use_cifar10))
        else:
            loop.run_until_complete(main(display_mode=display_mode, frames_dir=frames_dir, unlimited=unlimited, save_json=save_json, save_image=save_image, group_type=group_type, data_dir=data_dir, use_cifar10=use_cifar10))
    except Exception as e:
        logger.error(f"Failed to run main: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LEPAUTE Pipeline")
    parser.add_argument("--data_dir", type=str, default=None, help="Directory containing CIFAR-10 or image pairs")
    parser.add_argument("--use_cifar10", action="store_true", help="Use CIFAR-10 dataset")
    parser.add_argument("--group_type", type=str, default="SE(2)", choices=["SE(2)", "SE(3)", "Sim(3)"], help="Lie group type")
    parser.add_argument("--display_mode", type=str, default="gui,realtime", help="Display mode: gui,realtime,json")
    parser.add_argument("--frames_dir", type=str, default="frames", help="Directory to save frames")
    parser.add_argument("--unlimited", action="store_true", help="Run indefinitely")
    parser.add_argument("--save_json", action="store_true", help="Save data to JSON")
    parser.add_argument("--save_image", action="store_true", help="Save frames to disk")
    args = parser.parse_args()
    
    run_main(
        display_mode=args.display_mode,
        frames_dir=args.frames_dir,
        unlimited=args.unlimited,
        save_json=args.save_json,
        save_image=args.save_image,
        group_type=args.group_type,
        data_dir=args.data_dir,
        use_cifar10=args.use_cifar10
    )