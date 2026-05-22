import torch
import torch.nn as nn
import torch.nn.functional as F
import asyncio
from typing import Tuple, List
from config import DEVICE, HAS_PML, OBJECT_NAMES, logger
from models import GeometricLoss, TransformerModel, LieGroupRepresentation
from utils import GeometricTransformationExtraction
from torch.optim import AdamW

class SelfSupervisedTrainer:
    def __init__(self, model: nn.Module, temperature: float = 0.5):
        self.model: nn.Module = model
        self.temperature: float = temperature
        self.contrastive_loss = None
        if HAS_PML:
            from pytorch_metric_learning import losses
            self.contrastive_loss = losses.NTXentLoss(temperature=temperature)
        else:
            logger.warning("Using MSE fallback for self-supervised learning")

    async def train_step(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1, x2 = x1.to(DEVICE), x2.to(DEVICE)
        emb1 = self.model(x1.unsqueeze(0), torch.zeros(3, device=DEVICE))
        emb2 = self.model(x2.unsqueeze(0), torch.rand(3, device=DEVICE))
        if HAS_PML and self.contrastive_loss is not None:
            labels = torch.arange(emb1.size(0), device=DEVICE)
            loss = self.contrastive_loss(torch.cat([emb1, emb2]), torch.cat([labels, labels]))
        else:
            loss = F.mse_loss(emb1, emb2)
        return loss

class PreTraining:
    def __init__(self, model: nn.Module, train_dataset: List, val_dataset: List):
        self.model: nn.Module = model.to(DEVICE)
        self.train_dataset: List = train_dataset
        self.val_dataset: List = val_dataset
        self.optimizer: AdamW = AdamW(self.model.parameters(), lr=1e-4)
        self.loss_fn: GeometricLoss = GeometricLoss()
        self.self_sup: SelfSupervisedTrainer = SelfSupervisedTrainer(self.model)

    async def train_step(self, image1: torch.Tensor, image2: torch.Tensor, label: torch.Tensor) -> Tuple[float, dict]:
        try:
            self.optimizer.zero_grad()
            lie_params = GeometricTransformationExtraction().extract_2d_transform(image1, image2).unsqueeze(0)
            output = self.model(image1.unsqueeze(0), lie_params)
            # Ensure label is 1D tensor of shape [batch_size]
            if label.dim() > 1:
                label = label.squeeze()
            if label.dim() == 0:
                label = label.unsqueeze(0)
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
        self.pre_training: PreTraining = pre_training

    async def train(self, num_epochs: int):
        total_loss: float = 0
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
        self.model: nn.Module = model.to(DEVICE)
        self.val_dataset: List[torch.Tensor] = val_dataset
        self.transform_extract: GeometricTransformationExtraction = GeometricTransformationExtraction()
        self.lie_rep: LieGroupRepresentation = LieGroupRepresentation()

    def evaluate(self) -> float:
        if not self.val_dataset:
            logger.warning("Validation dataset is empty. Skipping evaluation.")
            return 0.0
        correct: int = 0
        total: int = 0
        inv_error: float = 0
        processed_samples: int = 0
        with torch.no_grad():
            for image1, image2, label in self.val_dataset:
                try:
                    image1 = image1.to(DEVICE, DTYPE)
                    image2 = image2.to(DEVICE, DTYPE)
                    # Ensure label is 1D tensor of shape [batch_size]
                    label = label.to(DEVICE)
                    if label.dim() > 1:
                        label = label.squeeze()
                    if label.dim() == 0:
                        label = label.unsqueeze(0)
                    lie_params = self.transform_extract.extract_2d_transform(image1, image2).unsqueeze(0)
                    output = self.model(image1.unsqueeze(0), lie_params)
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
        self.training: Training = training

    async def tune(self, lr_values: List[float]) -> float:
        best_lr: float = lr_values[0]
        best_acc: float = 0
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