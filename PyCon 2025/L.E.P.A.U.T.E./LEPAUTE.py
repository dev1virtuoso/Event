import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from typing import Tuple, List, Dict, Optional
import kornia as K
import platform
import json
import os
from collections import defaultdict
import multiprocessing
import time
import logging
import asyncio
from torch.optim import AdamW
import torchvision.transforms as transforms

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from pytorch_metric_learning import losses
    HAS_PML = True
except ImportError:
    HAS_PML = False
    logger.warning("pytorch-metric-learning not found. Using MSE fallback. Install with `pip install pytorch-metric-learning`.")

DATA_STORE = defaultdict(list)
DTYPE = torch.float32

OBJECT_NAMES = {
    0: "Book", 1: "Pen", 2: "Phone", 3: "Mug", 4: "Keyboard",
    5: "Mouse", 6: "Notebook", 7: "Bottle", 8: "Glasses", 9: "Wallet",
    10: "Chair", 11: "Table",
}

def get_device():
    """Detect available GPU for acceleration."""
    if torch.cuda.is_available():
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    elif torch.backends.mps.is_available() and platform.system() == "Darwin":
        logger.info("Using MPS device for acceleration.")
        return torch.device("mps")
    logger.info("Using CPU as no suitable GPU found.")
    return torch.device("cpu")

DEVICE = torch.device("cpu")

class DynamicResourceManager:
    def __init__(self, target_fps: float = 10.0, max_threads: int = multiprocessing.cpu_count()):
        self.target_fps = target_fps
        self.max_threads = max(max_threads, 1)
        self.min_threads = 1
        self.current_threads = max(1, self.max_threads // 2)
        self.base_delay = 0.01
        self.dynamic_delay = self.base_delay
        self.fps_history = []
        self.max_history = 20
        torch.set_num_threads(self.current_threads)

    def update(self, frame_time: float):
        actual_fps = 1.0 / frame_time if frame_time > 0 else float('inf')
        self.fps_history.append(actual_fps)
        if len(self.fps_history) > self.max_history:
            self.fps_history.pop(0)
        
        avg_fps = sum(self.fps_history) / len(self.fps_history) if self.fps_history else actual_fps
        
        if avg_fps < self.target_fps * 0.9:
            if self.current_threads > self.min_threads:
                self.current_threads -= 1
                torch.set_num_threads(self.current_threads)
            if self.dynamic_delay > 0.001:
                self.dynamic_delay *= 0.8
        elif avg_fps > self.target_fps * 1.1:
            if self.current_threads < self.max_threads:
                self.current_threads += 1
                torch.set_num_threads(self.current_threads)
            self.dynamic_delay = min(self.dynamic_delay * 1.2, 0.1)
        
        if str(DEVICE).startswith("cuda"):
            try:
                gpu_util = torch.cuda.utilization(DEVICE)
                if gpu_util > 90:
                    self.dynamic_delay *= 1.1
            except Exception as e:
                logger.warning(f"GPU utilization check failed: {e}")
        elif str(DEVICE).startswith("mps"):
            try:
                torch.mps.empty_cache()
            except Exception as e:
                logger.warning(f"MPS cache clear failed: {e}")
        
        return self.dynamic_delay

def get_collected_data() -> List[dict]:
    return DATA_STORE["pipeline_data"]

def create_meshgrid(height: int, width: int, device: torch.device = DEVICE) -> torch.Tensor:
    x = torch.linspace(0, width - 1, width, device=device, dtype=DTYPE)
    y = torch.linspace(0, height - 1, height, device=device, dtype=DTYPE)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    grid = torch.stack([grid_x, grid_y], dim=-1)
    return grid

def custom_match_nn(descriptors1: torch.Tensor, descriptors2: torch.Tensor, threshold: float = 0.9) -> torch.Tensor:
    descriptors1 = descriptors1.cpu()
    descriptors2 = descriptors2.cpu()
    dists = torch.cdist(descriptors1[0], descriptors2[0], p=2)
    dists1, idx1 = torch.min(dists, dim=1)
    dists2, idx2 = torch.min(dists, dim=0)
    matches = []
    for i in range(dists1.shape[0]):
        if idx2[idx1[i]] == i and dists1[i] < threshold:
            matches.append([i, idx1[i]])
    return torch.tensor(matches, dtype=torch.long, device=DEVICE) if matches else torch.empty((0, 2), dtype=torch.long, device=DEVICE)

class LieGroupRepresentation(nn.Module):
    def __init__(self, group_type: str = "SE(2)"):
        super().__init__()
        self.group_type = group_type
        self.param_dim = 6 if group_type == "SE(3)" else 7 if group_type == "Sim(3)" else 3

    def lie_algebra_to_params(self, alg: torch.Tensor) -> torch.Tensor:
        if self.group_type == "SE(2)":
            theta, tx, ty = alg[..., 0], alg[..., 1], alg[..., 2]
            return torch.stack([theta, tx, ty], dim=-1)
        elif self.group_type in ["SE(3)", "Sim(3)"]:
            return alg
        raise ValueError(f"Unsupported group: {self.group_type}")

    def params_to_matrix(self, params: torch.Tensor) -> torch.Tensor:
        if self.group_type == "SE(2)":
            theta, tx, ty = params[..., 0], params[..., 1], params[..., 2]
            cos_t = torch.cos(theta)
            sin_t = torch.sin(theta)
            matrix = torch.stack([
                cos_t, -sin_t, tx,
                sin_t, cos_t, ty,
                torch.zeros_like(tx), torch.zeros_like(tx), torch.ones_like(tx)
            ], dim=-1).view(*params.shape[:-1], 3, 3)
            return matrix
        raise ValueError(f"Unsupported group: {self.group_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lie_algebra_to_params(x)

class LieGroupConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, group_type: str = "SE(2)"):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2).to(DEVICE)
        self.lie_rep = LieGroupRepresentation(group_type)
        self.group_samples = 4  # Reduced for memory efficiency

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        x = x.to(DEVICE)
        g = g.to(DEVICE)
        outputs = []
        for i in range(self.group_samples):
            g_sample = g * (i / self.group_samples)
            transformed_x = self.apply_group_action(x, g_sample)
            out = self.conv(transformed_x)
            outputs.append(out)
        return torch.mean(torch.stack(outputs), dim=0)

    def apply_group_action(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        matrix = self.lie_rep.params_to_matrix(g)
        grid = create_meshgrid(x.shape[2], x.shape[3], x.device)
        warped_grid = K.geometry.transform_points(matrix.unsqueeze(0), grid.unsqueeze(0))
        return F.grid_sample(x, warped_grid, align_corners=True)

class LieGroupAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, group_type: str = "SE(2)"):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads).to(DEVICE)
        self.lie_rep = LieGroupRepresentation(group_type)
        self.embed_dim = embed_dim
        self.group_projection = nn.Linear(3, embed_dim).to(DEVICE)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        q, k, v, g = q.to(DEVICE), k.to(DEVICE), v.to(DEVICE), g.to(DEVICE)
        rel_g = self.compute_relative_g(g)
        rel_g = self.group_projection(rel_g)
        q = q + rel_g
        k = k + rel_g
        attn_output, _ = self.attn(q, k, v)
        return attn_output

    def compute_relative_g(self, g: torch.Tensor) -> torch.Tensor:
        return g.unsqueeze(1) - g.unsqueeze(0)

class TransformerModel(nn.Module):
    def __init__(self, num_classes: int = len(OBJECT_NAMES), embed_dim: int = 64, num_heads: int = 2, num_layers: int = 2):
        super().__init__()
        self.embed_dim = embed_dim
        self.lie_conv = LieGroupConv(3, embed_dim, kernel_size=3)
        self.lie_attn_layers = nn.ModuleList([LieGroupAttention(embed_dim, num_heads) for _ in range(num_layers)])
        self.fc = nn.Linear(embed_dim, num_classes).to(DEVICE)
        self.norm = nn.LayerNorm(embed_dim).to(DEVICE)
        self.lie_rep = LieGroupRepresentation()

    def forward(self, x: torch.Tensor, lie_params: torch.Tensor) -> torch.Tensor:
        x = self.lie_conv(x, lie_params)
        x = x.flatten(2).permute(2, 0, 1)
        for attn in self.lie_attn_layers:
            res = x
            x = attn(x, x, x, lie_params)
            x = self.norm(x + res)
        x = x.mean(0)
        return self.fc(x.unsqueeze(0))

class GeometricLoss(nn.Module):
    def __init__(self, group_samples: int = 4):
        super().__init__()
        self.group_samples = group_samples
        self.ce_loss = nn.CrossEntropyLoss()
        self.inv_loss_weight = 0.5

    def forward(self, outputs: torch.Tensor, labels: torch.Tensor, model: nn.Module, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        outputs, labels, x, g = outputs.to(DEVICE), labels.to(DEVICE), x.to(DEVICE), g.to(DEVICE)
        ce = self.ce_loss(outputs, labels)
        inv_loss = 0
        for i in range(self.group_samples):
            g_sample = g * (i / self.group_samples)
            transformed_x = model.lie_conv.apply_group_action(x, g_sample)
            transformed_out = model(transformed_x, g_sample)
            inv_loss += F.mse_loss(outputs, transformed_out)
        return ce + self.inv_loss_weight * (inv_loss / self.group_samples)

class SelfSupervisedTrainer:
    def __init__(self, model: nn.Module, temperature: float = 0.5):
        super().__init__()
        self.model = model
        self.temperature = temperature
        if HAS_PML:
            self.contrastive_loss = losses.NTXentLoss(temperature=temperature)
        else:
            logger.warning("Using MSE fallback for self-supervised learning")

    async def train_step(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1, x2 = x1.to(DEVICE), x2.to(DEVICE)
        emb1 = self.model(x1.unsqueeze(0), torch.zeros(3, device=DEVICE))
        emb2 = self.model(x2.unsqueeze(0), torch.rand(3, device=DEVICE))
        if HAS_PML:
            labels = torch.arange(emb1.size(0), device=DEVICE)
            loss = self.contrastive_loss(torch.cat([emb1, emb2]), torch.cat([labels, labels]))
        else:
            loss = F.mse_loss(emb1, emb2)
        return loss

class DataPreprocessing:
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        self.target_size = target_size
        self.clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(8, 8))  # Increased clipLimit for better contrast
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def normalize_2d(self, image: np.ndarray) -> torch.Tensor:
        if len(image.shape) == 2 or image.shape[-1] != 3:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        image = cv2.convertScaleAbs(image, alpha=1.5, beta=0)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_gray = self.clahe.apply(image_gray)
        image_gray = cv2.equalizeHist(image_gray)  # Added histogram equalization
        image_gray = cv2.GaussianBlur(image_gray, (3, 3), 0)
        image = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
        image = cv2.resize(image, self.target_size)
        return self.transform(image).to(DEVICE)

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
        return self.transform(image.unsqueeze(0))[0].squeeze(0)

class MultiModalProcessor:
    def __init__(self):
        self.point_cloud_proj = nn.Linear(3, 64).to(DEVICE)  # Match embed_dim

    def process(self, rgb: torch.Tensor, point_cloud: Optional[torch.Tensor] = None) -> torch.Tensor:
        rgb = rgb.to(DEVICE)
        if point_cloud is not None:
            point_cloud = point_cloud.to(DEVICE)
            pc_feat = self.point_cloud_proj(point_cloud)
            rgb = torch.cat([rgb, pc_feat.unsqueeze(2).unsqueeze(3).repeat(1, 1, rgb.shape[2], rgb.shape[3])], dim=1)
        return rgb

class GeometricTransformationExtraction:
    def __init__(self):
        self.local_feature = K.feature.LocalFeature(
            detector=K.feature.DISK().to(DEVICE),
            descriptor=K.feature.SIFTDescriptor(patch_size=32).to(DEVICE)
        ).to(DEVICE)
        self.orb = cv2.ORB_create(nfeatures=5000, scaleFactor=1.2, nlevels=8, edgeThreshold=5)
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def extract_2d_transform(self, image1: torch.Tensor, image2: torch.Tensor) -> torch.Tensor:
        image1, image2 = image1.to(DEVICE, DTYPE), image2.to(DEVICE, DTYPE)
        rgb1, rgb2 = image1.unsqueeze(0), image2.unsqueeze(0)
        
        img1_np = (image1.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        img2_np = (image2.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        img1_var, img2_var = img1_np.std(), img2_np.std()
        logger.debug(f"Image variances - img1: {img1_var:.4f}, img2: {img2_var:.4f}")
        if img1_var < 0.02 or img2_var < 0.02:
            logger.debug("Low image variance detected, switching to ORB fallback")
            return self._orb_fallback(img1_np, img2_np)
        
        try:
            # Extract features and log details
            feats1 = self.local_feature(rgb1)
            feats2 = self.local_feature(rgb2)
            
            # Detailed logging for feature output
            logger.debug(f"feats1 type: {type(feats1)}")
            logger.debug(f"feats1 content: {feats1}")
            logger.debug(f"feats2 type: {type(feats2)}")
            logger.debug(f"feats2 content: {feats2}")
            
            # Handle possible output formats
            if isinstance(feats1, (tuple, list)):
                logger.debug(f"feats1 is a tuple/list with length: {len(feats1)}")
                if len(feats1) == 3:  # Handle (lafs, responses, descriptors)
                    keypoints1 = K.feature.get_laf_center(feats1[0])
                    descriptors1 = feats1[2]
                    logger.debug(f"Extracted keypoints1 shape: {keypoints1.shape if keypoints1 is not None else 'None'}")
                    logger.debug(f"Extracted descriptors1 shape: {descriptors1.shape if descriptors1 is not None else 'None'}")
                elif len(feats1) == 2:  # Handle (lafs, descriptors)
                    keypoints1 = K.feature.get_laf_center(feats1[0])
                    descriptors1 = feats1[1]
                    logger.debug(f"Extracted keypoints1 shape: {keypoints1.shape if keypoints1 is not None else 'None'}")
                    logger.debug(f"Extracted descriptors1 shape: {descriptors1.shape if descriptors1 is not None else 'None'}")
                else:
                    logger.warning(f"Unexpected tuple/list length for feats1: {len(feats1)}, using ORB fallback")
                    return self._orb_fallback(img1_np, img2_np)
            elif hasattr(feats1, 'keypoints') and hasattr(feats1, 'descriptors'):
                keypoints1 = feats1.keypoints
                descriptors1 = feats1.descriptors
                logger.debug(f"Extracted keypoints1 shape (from attributes): {keypoints1.shape if keypoints1 is not None else 'None'}")
                logger.debug(f"Extracted descriptors1 shape (from attributes): {descriptors1.shape if descriptors1 is not None else 'None'}")
            else:
                logger.warning(f"Unexpected feature output format for feats1: {type(feats1)}, using ORB fallback")
                return self._orb_fallback(img1_np, img2_np)
            
            if isinstance(feats2, (tuple, list)):
                logger.debug(f"feats2 is a tuple/list with length: {len(feats2)}")
                if len(feats2) == 3:
                    keypoints2 = K.feature.get_laf_center(feats2[0])
                    descriptors2 = feats2[2]
                    logger.debug(f"Extracted keypoints2 shape: {keypoints2.shape if keypoints2 is not None else 'None'}")
                    logger.debug(f"Extracted descriptors2 shape: {descriptors2.shape if descriptors2 is not None else 'None'}")
                elif len(feats2) == 2:
                    keypoints2 = K.feature.get_laf_center(feats2[0])
                    descriptors2 = feats2[1]
                    logger.debug(f"Extracted keypoints2 shape: {keypoints2.shape if keypoints2 is not None else 'None'}")
                    logger.debug(f"Extracted descriptors2 shape: {descriptors2.shape if descriptors2 is not None else 'None'}")
                else:
                    logger.warning(f"Unexpected tuple/list length for feats2: {len(feats2)}, using ORB fallback")
                    return self._orb_fallback(img1_np, img2_np)
            elif hasattr(feats2, 'keypoints') and hasattr(feats2, 'descriptors'):
                keypoints2 = feats2.keypoints
                descriptors2 = feats2.descriptors
                logger.debug(f"Extracted keypoints2 shape (from attributes): {keypoints2.shape if keypoints2 is not None else 'None'}")
                logger.debug(f"Extracted descriptors2 shape (from attributes): {descriptors2.shape if descriptors2 is not None else 'None'}")
            else:
                logger.warning(f"Unexpected feature output format for feats2: {type(feats2)}, using ORB fallback")
                return self._orb_fallback(img1_np, img2_np)
            
            if keypoints1 is None or descriptors1 is None or keypoints2 is None or descriptors2 is None:
                logger.warning("Invalid feature output (None detected), using ORB fallback")
                return self._orb_fallback(img1_np, img2_np)
            
            if len(keypoints1.shape) < 2 or keypoints1.shape[1] < 4 or len(descriptors1.shape) < 2:
                logger.warning(f"Insufficient keypoints1: shape={keypoints1.shape}, descriptors1 shape={descriptors1.shape}, using ORB fallback")
                return self._orb_fallback(img1_np, img2_np)
            if len(keypoints2.shape) < 2 or keypoints2.shape[1] < 4 or len(descriptors2.shape) < 2:
                logger.warning(f"Insufficient keypoints2: shape={keypoints2.shape}, descriptors2 shape={descriptors2.shape}, using ORB fallback")
                return self._orb_fallback(img1_np, img2_np)
            
            torch.mps.empty_cache()  # Clear MPS memory after feature extraction
        except Exception as e:
            logger.warning(f"Feature extraction failed: {e}, using ORB fallback")
            return self._orb_fallback(img1_np, img2_np)
        
        matches = custom_match_nn(descriptors1, descriptors2, threshold=0.9)
        logger.debug(f"Number of matches: {matches.shape[0]}")
        if matches.shape[0] < 4:
            logger.debug("Insufficient matches, using ORB fallback")
            return self._orb_fallback(img1_np, img2_np)
        
        src_pts = keypoints1[0][matches[:, 0]].cpu().numpy()
        dst_pts = keypoints2[0][matches[:, 1]].cpu().numpy()
        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H is None:
            logger.debug("Homography estimation failed, using ORB fallback")
            return self._orb_fallback(img1_np, img2_np)
        
        H = torch.from_numpy(H).to(DEVICE, DTYPE)
        det = H[0, 0] * H[1, 1] - H[0, 1] * H[1, 0]
        if abs(det) < 1e-8:
            logger.debug("Singular homography, returning zero transform")
            return torch.zeros(3, device=DEVICE, dtype=DTYPE)
        theta = torch.atan2(H[1, 0], H[0, 0])
        tx, ty = H[0, 2], H[1, 2]
        logger.debug(f"Computed transform - theta: {theta:.4f}, tx: {tx:.2f}, ty: {ty:.2f}")
        return torch.tensor([theta, tx, ty], device=DEVICE, dtype=DTYPE)

    def _orb_fallback(self, img1: np.ndarray, img2: np.ndarray) -> torch.Tensor:
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(8, 8))
        img1_gray = clahe.apply(img1_gray)
        img2_gray = clahe.apply(img2_gray)
        kp1, des1 = self.orb.detectAndCompute(img1_gray, None)
        kp2, des2 = self.orb.detectAndCompute(img2_gray, None)
        logger.debug(f"ORB keypoints - img1: {len(kp1) if kp1 else 0}, img2: {len(kp2) if kp2 else 0}")
        if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
            logger.debug("ORB fallback failed: insufficient keypoints")
            return torch.zeros(3, device=DEVICE, dtype=DTYPE)
        matches = self.bf_matcher.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)[:500]
        logger.debug(f"ORB matches: {len(matches)}")
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches])
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches])
        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H is None:
            logger.debug("ORB homography estimation failed")
            return torch.zeros(3, device=DEVICE, dtype=DTYPE)
        H = torch.from_numpy(H).to(DEVICE, DTYPE)
        theta = torch.atan2(H[1, 0], H[0, 0])
        tx, ty = H[0, 2], H[1, 2]
        logger.debug(f"ORB transform - theta: {theta:.4f}, tx: {tx:.2f}, ty: {ty:.2f}")
        return torch.tensor([theta, tx, ty], device=DEVICE, dtype=DTYPE)

class PreTraining:
    def __init__(self, model: nn.Module, train_dataset: List, val_dataset: List):
        self.model = model.to(DEVICE)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.optimizer = AdamW(self.model.parameters(), lr=1e-4)
        self.loss_fn = GeometricLoss()
        self.self_sup = SelfSupervisedTrainer(self.model)

    async def train_step(self, image1: torch.Tensor, image2: torch.Tensor, label: torch.Tensor) -> Tuple[float, dict]:
        try:
            if image1.shape != (3, 224, 224) or image2.shape != (3, 224, 224):
                logger.warning(f"Invalid image shapes: image1={image1.shape}, image2={image2.shape}")
                return 0.0, {}
            self.optimizer.zero_grad()
            lie_params = GeometricTransformationExtraction().extract_2d_transform(image1, image2).unsqueeze(0)
            output = self.model(image1.unsqueeze(0), lie_params)
            if output.shape[0] == 0 or output.shape[1] == 0:
                logger.warning(f"Invalid model output shape: {output.shape}")
                return 0.0, {}
            label = label.unsqueeze(0) if label.dim() == 0 else label
            loss = self.loss_fn(output, label, self.model, image1.unsqueeze(0), lie_params.squeeze(0))
            ss_loss = await self.self_sup.train_step(image1, image2)
            total_loss = loss + 0.3 * ss_loss
            total_loss.backward()
            self.optimizer.step()
            predicted = torch.argmax(output, dim=1).item()
            detected_object = OBJECT_NAMES.get(predicted, "Unknown")
            data = {
                "image1": image1.detach().cpu().numpy(),
                "image2": image2.detach().cpu().numpy(),
                "lie_params": lie_params.detach().cpu().numpy().tolist(),
                "output": output.detach().cpu().numpy().tolist(),
                "loss": total_loss.item(),
                "predicted": predicted,
                "label": label.item(),
                "detected_object": detected_object
            }
            if str(DEVICE).startswith("mps"):
                torch.mps.empty_cache()
            return total_loss.item(), data
        except Exception as e:
            logger.error(f"Train step failed: {e}")
            if str(DEVICE).startswith("mps"):
                torch.mps.empty_cache()
            return 0.0, {}

class Training:
    def __init__(self, pre_training: PreTraining):
        self.pre_training = pre_training

    async def train(self, num_epochs: int):
        total_loss = 0
        for epoch in range(num_epochs):
            for image1, image2, label in self.pre_training.train_dataset:
                loss, _ = await self.pre_training.train_step(image1, image2, label)
                total_loss += loss
                await asyncio.sleep(0.01)
            if len(self.pre_training.train_dataset) > 0:
                avg_loss = total_loss / len(self.pre_training.train_dataset)
                logger.info(f"Epoch {epoch+1}/{num_epochs}, Avg Loss: {avg_loss:.4f}")
                total_loss = 0

class Evaluation:
    def __init__(self, model: nn.Module, val_dataset: List[torch.Tensor]):
        self.model = model.to(DEVICE)
        self.val_dataset = val_dataset
        self.transform_extract = GeometricTransformationExtraction()
        self.lie_rep = LieGroupRepresentation()

    def evaluate(self) -> float:
        if not self.val_dataset:
            logger.warning("Validation dataset is empty. Skipping evaluation.")
            return 0.0
        correct = 0
        total = 0
        inv_error = 0
        processed_samples = 0
        with torch.no_grad():
            for image1, image2, label in self.val_dataset:
                try:
                    if image1.shape != (3, 224, 224) or image2.shape != (3, 224, 224):
                        logger.warning(f"Invalid validation image shapes: image1={image1.shape}, image2={image2.shape}")
                        continue
                    if label.dim() == 0:
                        label = label.unsqueeze(0)
                    image1 = image1.to(DEVICE, DTYPE)
                    image2 = image2.to(DEVICE, DTYPE)
                    label = label.to(DEVICE)
                    lie_params = self.transform_extract.extract_2d_transform(image1, image2).unsqueeze(0)
                    output = self.model(image1.unsqueeze(0), lie_params)
                    if output.shape[0] == 0 or output.shape[1] == 0:
                        logger.warning(f"Invalid model output shape: {output.shape}")
                        continue
                    _, predicted = torch.max(output, 1)
                    total += label.size(0)
                    correct += (predicted == label).sum().item()
                    transformed_out = self.model(self.model.lie_conv.apply_group_action(image1.unsqueeze(0), lie_params.squeeze(0)), lie_params)
                    inv_error += F.mse_loss(output, transformed_out).item()
                    processed_samples += 1
                except Exception as e:
                    logger.warning(f"Evaluation failed for sample: {e}")
                    continue
                finally:
                    if str(DEVICE).startswith("mps"):
                        torch.mps.empty_cache()
        acc = correct / total if total > 0 else 0.0
        avg_inv_error = inv_error / processed_samples if processed_samples > 0 else 0.0
        logger.info(f"Evaluation - Accuracy: {acc:.4f}, Inv Error: {avg_inv_error:.4f}")
        return acc

class Hyperparameter:
    def __init__(self, training: Training):
        self.training = training

    async def tune(self, lr_values: List[float]) -> float:
        best_lr = lr_values[0]
        best_acc = 0
        eval_module = Evaluation(self.training.pre_training.model, self.training.pre_training.val_dataset)
        for lr in lr_values:
            try:
                self.training.pre_training.optimizer = AdamW(
                    self.training.pre_training.model.parameters(), lr=lr
                )
                await self.training.train(1)
                acc = eval_module.evaluate()
                if acc > best_acc:
                    best_acc = acc
                    best_lr = lr
            except Exception as e:
                logger.warning(f"Hyperparameter tuning failed for lr={lr}: {e}")
                continue
        return best_lr

def display_raw_frame(frame: np.ndarray, frame_count: int, data: dict = None):
    try:
        display_frame = frame.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color = (0, 255, 0)  # Green text
        thickness = 1
        text_x = display_frame.shape[1] - 300  # Top-right alignment
        
        cv2.putText(display_frame, f"Frame: {frame_count}", (text_x, 20), font, font_scale, color, thickness)
        
        if data:
            y = 40
            line_spacing = 20
            cv2.putText(display_frame, f"SO(2) theta: {data['lie_params'][0][0]:.4f}",
                        (text_x, y), font, font_scale, color, thickness)
            cv2.putText(display_frame, f"SE(2) tx, ty: ({data['lie_params'][0][1]:.2f}, {data['lie_params'][0][2]:.2f})",
                        (text_x, y + line_spacing), font, font_scale, color, thickness)
            cv2.putText(display_frame, f"Model output: {[f'{x:.2f}' for x in data['output'][0][:3]]}...",
                        (text_x, y + 2 * line_spacing), font, font_scale, color, thickness)
            cv2.putText(display_frame, f"Loss: {data['loss']:.4f}",
                        (text_x, y + 3 * line_spacing), font, font_scale, color, thickness)
            cv2.putText(display_frame, f"Label: {data['label']}",
                        (text_x, y + 4 * line_spacing), font, font_scale, color, thickness)
            cv2.putText(display_frame, f"Detected: {data['detected_object']}",
                        (text_x, y + 5 * line_spacing), font, font_scale, color, thickness)
            
            # Movement detection with borders
            theta, tx, ty = data['lie_params'][0]
            if abs(theta) > 0.01:
                cv2.rectangle(display_frame, (10, 10),
                             (display_frame.shape[1] - 10, display_frame.shape[0] - 10),
                             (0, 0, 255), 2)  # Red border for rotation
                cv2.putText(display_frame, "Rotation",
                            (text_x, y + 6 * line_spacing), font, font_scale, (0, 0, 255), thickness)
            if abs(tx) > 5 or abs(ty) > 5:
                cv2.rectangle(display_frame, (10, 10),
                             (display_frame.shape[1] - 10, display_frame.shape[0] - 10),
                             (255, 0, 0), 2)  # Blue border for translation
                cv2.putText(display_frame, "Translation",
                            (text_x, y + 7 * line_spacing), font, font_scale, (255, 0, 0), thickness)
        
        cv2.imshow("Camera Feed", display_frame)
        cv2.waitKey(1)
    except Exception as e:
        logger.warning(f"Failed to display raw frame: {e}")

def display_or_save_data(data: dict, save_json: bool = False):
    try:
        if save_json:
            DATA_STORE["pipeline_data"].append(data)
            with open("lepaute_data.json", "w") as f:
                json.dump(DATA_STORE["pipeline_data"], f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save JSON data: {e}")

def generate_placeholder_val_dataset(num_samples: int = 10, img_size: Tuple[int, int] = (224, 224)) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Generate a placeholder validation dataset for testing."""
    val_dataset = []
    preprocess = DataPreprocessing()
    for i in range(num_samples):
        img1 = np.random.rand(img_size[0], img_size[1], 3).astype(np.float32) * 255  # Ensure proper range
        img2 = np.random.rand(img_size[0], img_size[1], 3).astype(np.float32) * 255
        img1 = preprocess.normalize_2d(img1)
        img2 = preprocess.normalize_2d(img2)
        label = torch.tensor([i % len(OBJECT_NAMES)], device=DEVICE, dtype=torch.long)  # Ensure 1D tensor
        val_dataset.append((img1, img2, label))
    return val_dataset

async def main(display_mode: str = "json", frames_dir: str = "frames", unlimited: bool = False, save_json: bool = False, save_image: bool = False):
    modes = [m.strip().lower() for m in display_mode.split(',')]
    show_window = 'gui' in modes or 'realtime' in modes
    save_to_json = 'json' in modes or save_json
    
    if save_image:
        os.makedirs(frames_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Cannot open camera")
        raise RuntimeError("Cannot open camera")

    FPS = 10
    resource_manager = DynamicResourceManager(target_fps=FPS)
    frame_count = 0
    prev_frame = None
    train_dataset = []
    val_dataset = generate_placeholder_val_dataset()  # TODO: Replace with actual validation data
    logger.warning("Using placeholder validation dataset. Replace with actual data for meaningful evaluation.")
    
    preprocess = DataPreprocessing()
    augment = DataAugmentation()
    multi_modal = MultiModalProcessor()
    model = TransformerModel().to(DEVICE)
    pre_training = PreTraining(model, train_dataset, val_dataset)
    training = Training(pre_training)
    hyperparam = Hyperparameter(training)
    
    logger.info(f"Running in {'REALTIME' if show_window else 'JSON'} mode...")
    logger.info(f"Save JSON: {save_to_json}, Save Image: {save_image}")
    logger.info(f"Resource usage: Dynamic adjustment to maintain {FPS} FPS, targeting ~50% CPU ({resource_manager.current_threads} threads initially).")
    logger.info("Monitor usage in Activity Monitor (macOS) under CPU tab.")
    logger.info(f"Starting the LEPAUTE pipeline in {'realtime' if show_window else 'json'} mode...")
    if save_image:
        logger.info(f"Saving frames to {frames_dir}...")
    
    try:
        best_lr = await hyperparam.tune([1e-5, 1e-4, 1e-3])
        logger.info(f"Best learning rate: {best_lr}")
    except Exception as e:
        logger.error(f"Hyperparameter tuning failed: {e}")
        return

    try:
        label_counter = 0
        logger.warning("Using cycling labels for training. Replace with actual labels for accurate training.")
        while unlimited or frame_count < FPS * 10:
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to capture frame")
                break
            frame = preprocess.normalize_2d(frame)
            frame = multi_modal.process(frame)
            
            if prev_frame is not None:
                frame1 = prev_frame
                frame2 = frame
                try:
                    if frame1.std() < 0.02 or frame2.std() < 0.02:
                        logger.debug("Skipping low-variance frame")
                        continue
                    
                    frame1_aug = augment.apply(frame1)
                    frame2_aug = augment.apply(frame2)
                    label = torch.tensor([label_counter % len(OBJECT_NAMES)], device=DEVICE, dtype=torch.long)  # Ensure 1D tensor
                    train_dataset.append((frame1_aug, frame2_aug, label))
                    loss, data = await pre_training.train_step(frame1_aug, frame2_aug, label)
                    if data:
                        frame_np = (frame.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                        frame_np = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
                        if show_window:
                            display_raw_frame(frame_np, frame_count, data)
                        if save_to_json:
                            display_or_save_data(data, save_to_json)
                    label_counter += 1
                except Exception as e:
                    logger.warning(f"Frame processing failed: {e}")
                    continue
            
            if save_image:
                frame_path = os.path.join(frames_dir, f"frame_{frame_count:04d}.png")
                frame_np = (frame.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                frame_np = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
                cv2.imwrite(frame_path, frame_np)
            
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

def run_main(display_mode: str = "json", **kwargs):
    loop = asyncio.get_event_loop()
    try:
        if loop.is_running():
            loop.create_task(main(display_mode=display_mode, **kwargs))
        else:
            loop.run_until_complete(main(display_mode=display_mode, **kwargs))
    except Exception as e:
        logger.error(f"Failed to run main: {e}")

if platform.system() == "Emscripten":
    asyncio.ensure_future(main())
else:
    if __name__ == "__main__":
        run_main(display_mode="gui,realtime", save_json=True, save_image=True)
