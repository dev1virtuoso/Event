import torch
import numpy as np
import cv2
import os
import pickle
from typing import List, Tuple
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import kornia as K
from config import DEVICE, DTYPE, OBJECT_NAMES, logger
import random

def unpickle(file: str) -> dict:
    """Read CIFAR-10 pickle file."""
    try:
        with open(file, 'rb') as fo:
            dict_data = pickle.load(fo, encoding='bytes')
        return dict_data
    except Exception as e:
        logger.error(f"Failed to unpickle file {file}: {e}")
        raise

class DataPreprocessing:
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        self.target_size: Tuple[int, int] = target_size
        self.clahe: cv2.CLAHE = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(8, 8))
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def normalize_2d(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess and normalize image."""
        try:
            if len(image.shape) == 2 or image.shape[-1] != 3:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            image = cv2.convertScaleAbs(image, alpha=1.5, beta=0)
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image_gray = self.clahe.apply(image_gray)
            image_gray = cv2.GaussianBlur(image_gray, (3, 3), 0)
            image = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
            image = cv2.resize(image, self.target_size)
            return self.transform(image).to(DEVICE)
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            raise

class DataAugmentation:
    def __init__(self):
        self.transform = K.augmentation.AugmentationSequential(
            K.augmentation.RandomRotation(degrees=30.0),
            K.augmentation.RandomAffine(degrees=15.0, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=15.0),
            K.augmentation.RandomHorizontalFlip(p=0.5),
            K.augmentation.RandomGaussianNoise(mean=0.0, std=0.05, p=0.5),
            data_keys=["input"]
        ).to(DEVICE)

    def apply(self, image: torch.Tensor) -> torch.Tensor:
        """Apply augmentation to image."""
        return self.transform(image.unsqueeze(0))[0].squeeze(0)

def load_real_dataset(data_dir: str = None, use_cifar10: bool = True, num_samples: int = 100) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Load a real dataset (CIFAR-10 or from local directory)."""
    dataset: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    preprocess = DataPreprocessing()
    augment = DataAugmentation()
    
    if use_cifar10:
        logger.info("Loading CIFAR-10 dataset from pickle files")
        cifar_dir = data_dir if data_dir else "data/cifar-10-batches-py"
        if not os.path.exists(cifar_dir):
            logger.warning(f"CIFAR-10 directory not found: {cifar_dir}. Attempting to download...")
            try:
                CIFAR10(root="./data", train=True, download=True)
                logger.info("CIFAR-10 downloaded successfully")
            except Exception as e:
                logger.error(f"Failed to download CIFAR-10: {e}. Falling back to placeholder dataset.")
                return generate_placeholder_val_dataset(num_samples=num_samples)
        
        for batch_id in range(1, 6):
            batch_file = os.path.join(cifar_dir, f"data_batch_{batch_id}")
            if not os.path.exists(batch_file):
                logger.warning(f"Batch file not found: {batch_file}")
                continue
            try:
                batch_data = unpickle(batch_file)
                images = batch_data[b'data']
                labels = batch_data[b'labels']
                
                for i in range(min(num_samples - len(dataset), len(images))):
                    img = images[i].reshape(3, 32, 32).transpose(1, 2, 0).astype(np.uint8)
                    img_processed = preprocess.normalize_2d(img)
                    img2_tensor = augment.apply(torch.from_numpy(img).permute(2, 0, 1).float() / 255.0)
                    img2_np = (img2_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    img2_processed = preprocess.normalize_2d(img2_np)
                    mapped_label = random.randint(0, len(OBJECT_NAMES) - 1)  # Random label for robustness
                    dataset.append((img_processed, img2_processed, torch.tensor([mapped_label], dtype=torch.long, device=DEVICE)))
                    if len(dataset) >= num_samples:
                        break
                if len(dataset) >= num_samples:
                    break
            except Exception as e:
                logger.error(f"Failed to process batch {batch_id}: {e}")
                continue
    elif data_dir:
        logger.info(f"Loading dataset from local directory: {data_dir}")
        image_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        for i in range(0, min(num_samples * 2, len(image_files)), 2):
            if i + 1 >= len(image_files):
                break
            img_path = image_files[i]
            img2_path = image_files[i + 1]
            img = cv2.imread(img_path)
            img2 = cv2.imread(img2_path)
            if img is None or img2 is None:
                logger.warning(f"Failed to load images: {img_path} or {img2_path}")
                continue
            img = preprocess.normalize_2d(img)
            img2 = preprocess.normalize_2d(img2)
            label = torch.tensor([random.randint(0, len(OBJECT_NAMES) - 1)], device=DEVICE, dtype=torch.long)
            dataset.append((img, img2, label))
    else:
        logger.warning("No data directory provided and CIFAR-10 not used. Falling back to placeholder dataset.")
        return generate_placeholder_val_dataset(num_samples=num_samples)
    
    logger.info(f"Loaded {len(dataset)} samples for dataset")
    return dataset

def generate_placeholder_val_dataset(num_samples: int = 10, img_size: Tuple[int, int] = (224, 224)) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Generate a placeholder validation dataset."""
    val_dataset: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    preprocess = DataPreprocessing()
    for i in range(num_samples):
        img1 = np.random.rand(img_size[0], img_size[1], 3).astype(np.float32) * 255
        img2 = np.random.rand(img_size[0], img_size[1], 3).astype(np.float32) * 255
        img1 = preprocess.normalize_2d(img1)
        img2 = preprocess.normalize_2d(img2)
        label = torch.tensor([random.randint(0, len(OBJECT_NAMES) - 1)], device=DEVICE, dtype=torch.long)
        val_dataset.append((img1, img2, label))
    return val_dataset