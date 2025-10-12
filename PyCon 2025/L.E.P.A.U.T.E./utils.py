import torch
import torch.nn as nn
import cv2
import numpy as np
from typing import List, Optional, Dict
import json
import multiprocessing
from config import DEVICE, DTYPE, DATA_STORE, logger
from models import LieGroupRepresentation
import kornia as K

def custom_match_nn(descriptors1: torch.Tensor, descriptors2: torch.Tensor, threshold: float = 0.9) -> torch.Tensor:
    """Match descriptors for feature extraction."""
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

def get_collected_data() -> List[dict]:
    """Return collected pipeline data."""
    return DATA_STORE["pipeline_data"]

class DynamicResourceManager:
    def __init__(self, target_fps: float = 10.0, max_threads: int = multiprocessing.cpu_count()):
        self.target_fps: float = target_fps
        self.max_threads: int = max(max_threads, 1)
        self.min_threads: int = 1
        self.current_threads: int = max(1, self.max_threads // 2)
        self.base_delay: float = 0.01
        self.dynamic_delay: float = self.base_delay
        self.fps_history: List[float] = []
        self.max_history: int = 20
        torch.set_num_threads(self.current_threads)

    def update(self, frame_time: float) -> float:
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

class GeometricTransformationError(Exception):
    """Base exception for transformation estimation failures."""
    pass

class GeometricTransformationExtraction:
    def __init__(self, var_threshold: float = 0.02, match_threshold: float = 0.9):
        """
        Initializes the transformation extraction module.

        Args:
            var_threshold (float): Minimum variance for images to be processed by the GPU pipeline.
            match_threshold (float): Threshold for feature matching.
        """
        self.var_threshold = var_threshold
        self.match_threshold = match_threshold
        
        # GPU-native feature extractor and matcher
        self.local_feature = K.feature.DISK().to(DEVICE).eval()
        self.matcher = K.feature.DescriptorMatcher('snn', self.match_threshold).to(DEVICE).eval()
        self.ransac = K.geometry.RANSAC('homography', inliers_threshold=5.0, max_iterations=50).to(DEVICE).eval()

        # CPU-based fallback components
        self.orb = cv2.ORB_create(nfeatures=2000, scaleFactor=1.2, nlevels=8)
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def _parse_kornia_features(self, features: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Parses the output of a Kornia feature detector."""
        if not all(k in features for k in ['keypoints', 'descriptors']):
            raise GeometricTransformationError("Kornia feature output missing 'keypoints' or 'descriptors'.")
        
        keypoints = features['keypoints']
        descriptors = features['descriptors']

        if keypoints.shape[1] < 4 or descriptors.shape[1] < 4:
            raise GeometricTransformationError(f"Insufficient features found: {keypoints.shape[1]} keypoints.")
            
        return keypoints, descriptors

    def _extract_gpu_transform(self, image1: torch.Tensor, image2: torch.Tensor) -> torch.Tensor:
        """
        Performs transformation estimation entirely on the GPU using Kornia.
        Raises a TransformationError subclass on failure.
        """
        # 1. Pre-check: Image Variance
        if image1.std() < self.var_threshold or image2.std() < self.var_threshold:
            raise GeometricTransformationError(f"Image variance below threshold {self.var_threshold}.")

        # 2. Feature Detection (GPU)
        # Kornia expects grayscale images for DISK, shape (B, 1, H, W)
        gray1 = K.color.rgb_to_grayscale(image1.unsqueeze(0))
        gray2 = K.color.rgb_to_grayscale(image2.unsqueeze(0))
        
        with torch.no_grad():
            feats1 = self.local_feature(gray1)
            feats2 = self.local_feature(gray2)
        
        # 3. Feature Parsing & Matching (GPU)
        kps1, descs1 = self._parse_kornia_features(feats1)
        kps2, descs2 = self._parse_kornia_features(feats2)

        distances, indices = self.matcher(descs1, descs2)
        if indices.shape[0] < 4:
            raise GeometricTransformationError(f"Insufficient matches found: {indices.shape[0]}.")
            
        # 4. Homography Estimation with RANSAC (GPU)
        src_pts = kps1[indices[:, 0], :, 2]  # Kornia keypoints format: (B, N, 4) -> (angle, scale, x, y)
        dst_pts = kps2[indices[:, 1], :, 2]  # We need the xy coordinates at index 2

        try:
            H, _ = self.ransac(src_pts, dst_pts)
            if H is None:
                raise GeometricTransformationError("Kornia RANSAC returned None.")
        except Exception as e:
            raise GeometricTransformationError(f"Kornia RANSAC failed with error: {e}") from e

        # 5. Decompose Homography to SE(2) parameters
        det = H[0, 0] * H[1, 1] - H[0, 1] * H[1, 0]
        if abs(det) < 1e-8:
            # A singular homography is a failure case
            raise GeometricTransformationError("Singular homography matrix cannot be decomposed.")
            
        theta = torch.atan2(H[1, 0], H[0, 0])
        tx, ty = H[0, 2], H[1, 2]
        
        return torch.tensor([theta, tx, ty], device=DEVICE, dtype=DTYPE)

    def _orb_fallback(self, image1_tensor: torch.Tensor, image2_tensor: torch.Tensor) -> torch.Tensor:
        """CPU-based fallback using OpenCV ORB."""
        # --- Conversion from Tensor to NumPy happens ONLY here ---
        img1_np = (image1_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        img2_np = (image2_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

        img1_gray = cv2.cvtColor(img1_np, cv2.COLOR_RGB2GRAY)
        img2_gray = cv2.cvtColor(img2_np, cv2.COLOR_RGB2GRAY)
        
        kp1, des1 = self.orb.detectAndCompute(img1_gray, None)
        kp2, des2 = self.orb.detectAndCompute(img2_gray, None)

        if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
            logger.debug("ORB fallback failed: insufficient keypoints.")
            return torch.zeros(3, device=DEVICE, dtype=DTYPE)
        
        matches = self.bf_matcher.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)[:min(500, len(matches))]
        
        if len(matches) < 4:
            logger.debug("ORB fallback failed: insufficient matches after sorting.")
            return torch.zeros(3, device=DEVICE, dtype=DTYPE)

        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches])
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches])
        
        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if H is None:
            logger.debug("ORB homography estimation failed.")
            return torch.zeros(3, device=DEVICE, dtype=DTYPE)
            
        H = torch.from_numpy(H).to(device=DEVICE, dtype=DTYPE)
        theta = torch.atan2(H[1, 0], H[0, 0])
        tx, ty = H[0, 2], H[1, 2]
        
        return torch.tensor([theta, tx, ty], device=DEVICE, dtype=DTYPE)

    def extract_2d_transform(self, image1: torch.Tensor, image2: torch.Tensor) -> torch.Tensor:
        """
        Estimates the 2D transformation (SE(2)) between two images.
        
        Tries a fast, GPU-native approach first. If it fails, falls back to a 
        CPU-based ORB method.

        Args:
            image1 (torch.Tensor): The first image tensor (C, H, W) on DEVICE.
            image2 (torch.Tensor): The second image tensor (C, H, W) on DEVICE.

        Returns:
            torch.Tensor: A 3-element tensor [theta, tx, ty] representing the transformation.
                          Returns a zero tensor on complete failure.
        """
        try:
            # --- Happy Path: Attempt GPU-native extraction ---
            lie_params = self._extract_gpu_transform(image1, image2)
            logger.debug("GPU-native transform extraction successful.")
            return lie_params
        except GeometricTransformationError as e:
            # --- Sad Path: Controlled fallback on known errors ---
            logger.debug(f"GPU transform failed: {e}. Falling back to CPU/ORB.")
            return self._orb_fallback(image1, image2)
        except Exception as e:
            # --- Catastrophic Path: Catch unexpected errors ---
            logger.warning(
                f"An unexpected error occurred during GPU transform: {e}. "
                "Falling back to CPU/ORB."
            )
            return self._orb_fallback(image1, image2)

class MultiModalProcessor:
    def __init__(self):
        self.point_cloud_proj: nn.Linear = nn.Linear(3, 64).to(DEVICE)

    def process(self, rgb: torch.Tensor, point_cloud: Optional[torch.Tensor] = None) -> torch.Tensor:
        rgb = rgb.to(DEVICE)
        if point_cloud is not None:
            point_cloud = point_cloud.to(DEVICE)
            pc_feat = self.point_cloud_proj(point_cloud)
            rgb = torch.cat([rgb, pc_feat.unsqueeze(2).unsqueeze(3).repeat(1, 1, rgb.shape[2], rgb.shape[3])], dim=1)
        return rgb

def display_raw_frame(frame: np.ndarray, frame_count: int, data: dict = None):
    """Display frame with annotations in GUI."""
    try:
        display_frame = frame.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color = (0, 255, 0)
        thickness = 1
        text_x = display_frame.shape[1] - 300
        
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
            
            theta, tx, ty = data['lie_params'][0]
            if abs(theta) > 0.01:
                cv2.rectangle(display_frame, (10, 10), 
                             (display_frame.shape[1] - 10, display_frame.shape[0] - 10), 
                             (0, 0, 255), 2)
                cv2.putText(display_frame, "Rotation", 
                            (text_x, y + 6 * line_spacing), font, font_scale, (255, 0, 0), thickness)
            if abs(tx) > 5 or abs(ty) > 5:
                cv2.rectangle(display_frame, (10, 10), 
                             (display_frame.shape[1] - 10, display_frame.shape[0] - 10), 
                             (255, 0, 0), 2)
                cv2.putText(display_frame, "Translation", 
                            (text_x, y + 7 * line_spacing), font, font_scale, (255, 0, 0), thickness)
        
        cv2.imshow("Camera Feed", display_frame)
        cv2.waitKey(1)
    except Exception as e:
        logger.warning(f"Failed to display raw frame: {e}")

def display_or_save_data(data: dict, save_json: bool = False, frame_count: int = 0, frame: np.ndarray = None):
    """Save pipeline data to JSON."""
    try:
        if save_json:
            data["frame_count"] = frame_count
            if frame is not None:
                data["frame_shape"] = list(frame.shape)
            DATA_STORE["pipeline_data"].append(data)
            with open("lepaute_data.json", "w") as f:
                json.dump(DATA_STORE["pipeline_data"], f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save JSON data: {e}")