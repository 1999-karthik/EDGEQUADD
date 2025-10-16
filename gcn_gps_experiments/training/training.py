from utils import accuracy, TotalMeter, count_params, isfloat
import torch
import numpy as np
from pathlib import Path
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, confusion_matrix, recall_score, average_precision_score
from sklearn.metrics import precision_recall_fscore_support, classification_report
from utils import continus_mixup_data
from omegaconf import DictConfig
from typing import List
import torch.utils.data as utils
from .lr_scheduler import LRScheduler


def macro_auc_ovr(y_true, proba):
    """
    Compute macro AUC using one-vs-rest approach.
    
    Args:
        y_true: integer class labels of shape (n_samples,)
        proba: predicted probabilities of shape (n_samples, n_classes)
    
    Returns:
        auc_macro: macro averaged AUC
    """
    try:
        # ROC AUC macro using one-vs-rest
        auc_macro = roc_auc_score(y_true, proba, multi_class="ovr", average="macro")
        return auc_macro
    except ValueError:
        # Handle case where only one class is present
        return 0.5


def macro_recall_specificity(y_true, proba, thresholds=0.5):
    """
    Compute macro recall and specificity using one-vs-rest approach.
    
    Args:
        y_true: integer class labels of shape (n_samples,)
        proba: predicted probabilities of shape (n_samples, n_classes)
        thresholds: float or list/array of length n_classes
    
    Returns:
        macro_recall, macro_specificity, per_class_dict
    """
    classes = np.unique(y_true)
    K = len(classes)
    
    if np.isscalar(thresholds):
        thresholds = np.array([thresholds] * K)
    thresholds = np.asarray(thresholds)
    
    recalls = []
    specs = []
    per_class = {}
    
    for idx, c in enumerate(classes):
        # one-vs-rest targets for class c
        y_pos = (y_true == c).astype(int)
        scores = proba[:, idx]
        y_pred = (scores >= thresholds[idx]).astype(int)
        
        # recall (sensitivity) for class c
        rec = recall_score(y_pos, y_pred, zero_division=0)
        
        # specificity for class c = TN / (TN + FP)
        try:
            tn, fp, fn, tp = confusion_matrix(y_pos, y_pred, labels=[0,1]).ravel()
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        except ValueError:
            spec = 0.0
        
        recalls.append(rec)
        specs.append(spec)
        per_class[c] = {"recall": rec, "specificity": spec, "threshold": thresholds[idx]}
    
    macro_recall = np.nanmean(recalls) if len(recalls) > 0 else 0.0
    macro_specificity = np.nanmean(specs) if len(specs) > 0 else 0.0
    
    return macro_recall, macro_specificity, per_class


def compute_macro_metrics(y_true, proba, dataset_name="unknown"):
    """
    Compute macro metrics for imbalanced datasets like PPMI.
    
    Args:
        y_true: integer class labels
        proba: predicted probabilities
        dataset_name: name of dataset for logging
    
    Returns:
        dict with macro metrics
    """
    # Convert to numpy if needed
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy()
    if torch.is_tensor(proba):
        proba = proba.cpu().numpy()
    
    # Ensure y_true is integer labels
    if y_true.ndim > 1:
        y_true = np.argmax(y_true, axis=1)
    
    # Compute macro AUC
    auc_macro = macro_auc_ovr(y_true, proba)
    
    # Compute macro recall and specificity
    macro_rec, macro_spec, per_class = macro_recall_specificity(y_true, proba, thresholds=0.5)
    
    # Also compute macro AUPRC for imbalanced datasets
    try:
        auprc_macro = average_precision_score(y_true, proba, average="macro")
    except ValueError:
        auprc_macro = 0.0
    
    return {
        "auc_macro": auc_macro,
        "recall_macro": macro_rec,
        "specificity_macro": macro_spec,
        "auprc_macro": auprc_macro,
        "per_class": per_class
    }


class Train:

    def __init__(self, cfg: DictConfig,
                 model: torch.nn.Module,
                 optimizers: List[torch.optim.Optimizer],
                 lr_schedulers: List[LRScheduler],
                 dataloaders: List[utils.DataLoader],
                 logger) -> None:

        self.config = cfg
        self.logger = logger
        self.model = model
        if self.logger:
            self.logger.info(f'#model params: {count_params(self.model)}')
        else:
            print(f'#model params: {count_params(self.model)}')
        self.train_dataloader, self.val_dataloader, self.test_dataloader = dataloaders
        self.epochs = cfg.training.epochs
        self.total_steps = cfg.total_steps
        self.optimizers = optimizers
        self.lr_schedulers = lr_schedulers
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
        # Result storage removed - only CSV results like BioBGT

        self.init_meters()

    def init_meters(self):
        self.train_loss, self.val_loss,\
            self.test_loss, self.train_accuracy,\
            self.val_accuracy, self.test_accuracy = [
                TotalMeter() for _ in range(6)]

    def reset_meters(self):
        for meter in [self.train_accuracy, self.val_accuracy,
                      self.test_accuracy, self.train_loss,
                      self.val_loss, self.test_loss]:
            meter.reset()

    def train_per_epoch(self, optimizer, lr_scheduler):
        """Train the model for one epoch with improved error handling."""
        self.model.train()
        
        try:
            for batch_idx, (time_series, node_feature, label) in enumerate(self.train_dataloader):
                # Validate input shapes
                if time_series.dim() != 3 or node_feature.dim() != 3:
                    print(f"Warning: Unexpected input shape at batch {batch_idx}")
                    continue
                
                # Convert one-hot labels to class indices for CrossEntropyLoss
                label_indices = torch.argmax(label, dim=1).long()
                self.current_step += 1

                # Update learning rate
                lr_scheduler.update(optimizer=optimizer, step=self.current_step)

                # Move to GPU with error handling
                try:
                    time_series = time_series.cuda()
                    node_feature = node_feature.cuda()
                    label_indices = label_indices.cuda()
                except RuntimeError as e:
                    print(f"CUDA error at batch {batch_idx}: {e}")
                    continue

                # Apply data augmentation if configured
                if hasattr(self.config.preprocess, 'continus') and self.config.preprocess.continus:
                    time_series, node_feature, label = continus_mixup_data(
                        time_series, node_feature, y=label)
                    # Re-convert labels after augmentation
                    label_indices = torch.argmax(label, dim=1).long().cuda()

                # Forward pass with gradient computation
                optimizer.zero_grad()
                
                try:
                    predict = self.model(time_series, node_feature)
                    loss = self.loss_fn(predict, label_indices)
                    
                    # Check for NaN or infinite values
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"Warning: Invalid loss at batch {batch_idx}: {loss.item()}")
                        continue
                    
                    loss.backward()
                    
                    # Gradient clipping for stability
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    
                    # Update metrics
                    self.train_loss.update_with_weight(loss.item(), label_indices.shape[0])
                    top1 = accuracy(predict, label_indices)[0]
                    self.train_accuracy.update_with_weight(top1, label_indices.shape[0])
                    
                except RuntimeError as e:
                    print(f"Forward/backward pass error at batch {batch_idx}: {e}")
                    continue
                    
        except Exception as e:
            print(f"Training epoch failed: {e}")
            raise

    def test_per_epoch(self, dataloader, loss_meter, acc_meter):
        labels = []
        result = []
        outputs = []

        self.model.eval()

        for time_series, node_feature, label in dataloader:
            time_series, node_feature, label = time_series.cuda(), node_feature.cuda(), label.cuda()
            output = self.model(time_series, node_feature)

            # Convert one-hot labels to class indices for CrossEntropyLoss
            label_indices = torch.argmax(label, dim=1).long()

            loss = self.loss_fn(output, label_indices)
            loss_meter.update_with_weight(
                loss.item(), label_indices.shape[0])
            top1 = accuracy(output, label_indices)[0]
            acc_meter.update_with_weight(top1, label_indices.shape[0])
            result += F.softmax(output, dim=1)[:, 1].tolist()
            labels += label_indices.tolist()
            outputs.append(output.cpu())

        # Concatenate all outputs for macro metrics computation
        all_outputs = torch.cat(outputs, dim=0)
        all_labels = torch.tensor(labels).float()

        # Compute metrics based on dataset type
        if self.config.dataset.name.lower().startswith('ppmi'):
            # Use macro metrics for PPMI (multi-class, imbalanced)
            y_prob_multi = torch.softmax(all_outputs, dim=1).detach().numpy()  # [prob_class0, prob_class1, prob_class2, prob_class3]
            
            # Compute macro metrics
            macro_metrics = compute_macro_metrics(all_labels.numpy(), y_prob_multi, self.config.dataset.name)
            
            auc = macro_metrics['auc_macro']
            sensitivity = macro_metrics['recall_macro']
            specificity = macro_metrics['specificity_macro']
            
            print(f"PPMI Macro Metrics:")
            print(f"  Macro AUC: {auc:.4f}")
            print(f"  Macro Recall: {sensitivity:.4f}")
            print(f"  Macro Specificity: {specificity:.4f}")
            print(f"  Macro AUPRC: {macro_metrics['auprc_macro']:.4f}")
            print(f"  Per-class details: {macro_metrics['per_class']}")
            
            # For compatibility with existing code, return in expected format
            return [auc, 0, 0, 0, sensitivity, specificity]  # [auc, precision, recall, f1, sensitivity, specificity]
        else:
            # For binary datasets, use the original approach
            auc = roc_auc_score(labels, result)
            result, labels = np.array(result), np.array(labels)
            result[result > 0.5] = 1
            result[result <= 0.5] = 0
            metric = precision_recall_fscore_support(
                labels, result, average='micro')

            report = classification_report(
                labels, result, output_dict=True, zero_division=0)

            recall = [0, 0]
            for k in report:
                if isfloat(k):
                    recall[int(float(k))] = report[k]['recall']
            return [auc] + list(metric) + recall

    def generate_save_learnable_matrix(self):
        # Learnable matrix saving removed - only CSV results like BioBGT
        pass

    def save_result(self, results: torch.Tensor):
        # Result saving removed - only CSV results like BioBGT
        pass

    def train(self):
        self.current_step = 0
        best_val_loss = float('inf')
        best_val_auc = 0.0  # For PPMI macro AUC tracking
        best_model_state = None
        best_epoch = 0
        for epoch in range(self.epochs):
            self.reset_meters()
            self.train_per_epoch(self.optimizers[0], self.lr_schedulers[0])
            val_result = self.test_per_epoch(self.val_dataloader,
                                             self.val_loss, self.val_accuracy)

            test_result = self.test_per_epoch(self.test_dataloader,
                                              self.test_loss, self.test_accuracy)

            # Log metrics based on dataset type
            if self.config.dataset.name.lower().startswith('ppmi'):
                print(" | ".join([
                    f'Epoch[{epoch}/{self.epochs}]',
                    f'Train Loss:{self.train_loss.avg: .3f}',
                    f'Train Accuracy:{self.train_accuracy.avg: .3f}%',
                    f'Test Loss:{self.test_loss.avg: .3f}',
                    f'Test Accuracy:{self.test_accuracy.avg: .3f}%',
                    f'Val Macro AUC:{val_result[0]:.4f}',
                    f'Test Macro AUC:{test_result[0]:.4f}',
                    f'Test Macro Sen:{test_result[-1]:.4f}',
                    f'LR:{self.lr_schedulers[0].lr:.4f}'
                ]))
            else:
                print(" | ".join([
                    f'Epoch[{epoch}/{self.epochs}]',
                    f'Train Loss:{self.train_loss.avg: .3f}',
                    f'Train Accuracy:{self.train_accuracy.avg: .3f}%',
                    f'Test Loss:{self.test_loss.avg: .3f}',
                    f'Test Accuracy:{self.test_accuracy.avg: .3f}%',
                    f'Val AUC:{val_result[0]:.4f}',
                    f'Test AUC:{test_result[0]:.4f}',
                    f'Test Sen:{test_result[-1]:.4f}',
                    f'LR:{self.lr_schedulers[0].lr:.4f}'
                ]))

            # Track best model based on dataset type
            if self.config.dataset.name.lower().startswith('ppmi'):
                # For PPMI: Select epoch with highest validation Macro AUC
                val_auc = val_result[0]
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_model_state = self.model.state_dict().copy()
                    best_epoch = epoch
            else:
                # For other datasets: Select epoch with lowest validation loss
                if self.val_loss.avg < best_val_loss:
                    best_val_loss = self.val_loss.avg
                    best_model_state = self.model.state_dict().copy()
                    best_epoch = epoch
            
            # Training process tracking removed - only CSV results like BioBGT

        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            if self.config.dataset.name.lower().startswith('ppmi'):
                print(f"Loaded best model from epoch {best_epoch} with val Macro AUC: {best_val_auc:.4f}")
            else:
                print(f"Loaded best model from epoch {best_epoch} with val_loss: {best_val_loss:.4f}")
        
        # Get final test metrics with best model
        final_test_result = self.test_per_epoch(self.test_dataloader,
                                               self.test_loss, self.test_accuracy)
        
        # Result saving removed - only CSV results like BioBGT
        
        # Return final metrics
        return {
            'test_auc': final_test_result[0],
            'test_sensitivity': final_test_result[-1],  # recall for class 1
            'test_specificity': final_test_result[-2],  # recall for class 0
            'test_accuracy': self.test_accuracy.avg
        }
