import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import continues_mixup_data, accuracy, optimizer_update
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, classification_report, confusion_matrix, recall_score, average_precision_score


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


def train(model, optimizer, args, train_loader, epoch):
    """
    model train
    """
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    # train
    criterion = nn.CrossEntropyLoss(reduction='sum')
    model = model.to(device)
    model.train()
    train_loss = 0
    train_acc_list = []
    step, total_steps = 0 + epoch * len(train_loader), len(train_loader) * args.epochs
    total_samples = 0
    
    for batch in train_loader:
        step += 1
        # Handle both 2-value (ADHD) and 3-value (ABIDE with timeseries) dataloaders
        if len(batch) == 2:
            corr_mat, label = batch
        elif len(batch) == 3:
            timeseries, corr_mat, label = batch
            # For now, we only use correlation matrices (timeseries not used in current model)
        else:
            raise ValueError(f"Unexpected batch size: {len(batch)}")
        
        corr_mat, label = corr_mat.to(device), label.to(device)
        
        # MIXUP THE CORRELATION MATRICES
        corr_mat, label = continues_mixup_data(corr_mat, y=label, device=device)
        label = label.float()
        
        # Pass correlation matrix through: CNN → ASPP → Quadratic → Atrous → Pooling → Classification
        output = model(corr_mat)

        optimizer_update(optimizer=optimizer, step=step, total_steps=total_steps, args=args)
        optimizer.zero_grad()
        loss = criterion(output, label.argmax(dim=1))  # Standard CE with hard labels
        loss.backward()
        optimizer.step()

        
        # Atrous weight updates removed during model simplification
        
        train_loss += loss.item()
        total_samples += corr_mat.shape[0]
        
        # Monitor accuracy roughly: compare logits argmax vs. mixed-label argmax
        with torch.no_grad():
            gt_argmax = torch.argmax(label, dim=1)
            top1 = accuracy(output, gt_argmax)[0] / 100
            train_acc_list.append(top1)

    train_loss = train_loss / total_samples  # Use actual number of samples
    train_acc = np.mean(train_acc_list)

    return {"train_loss": train_loss, "train_acc": train_acc}


def val_test(model, args, val_loader, test_loader):
    """
    model validation on valid dataset and acc&roc on test dataset
    """
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)  # Ensure model is on correct device

    # Use standard CE on hard labels for val/test
    ce = nn.CrossEntropyLoss(reduction='sum')

    # ----- VALID -----
    val_loss = 0
    val_acc_list, val_outputs, val_labels = [], [], []
    val_samples = 0
    
    with torch.no_grad():  # Disable gradient computation for validation
        for batch in val_loader:
            # Handle both 2-value (ADHD) and 3-value (ABIDE with timeseries) dataloaders
            if len(batch) == 2:
                corr_mat, label = batch
            elif len(batch) == 3:
                timeseries, corr_mat, label = batch
                # For now, we only use correlation matrices (timeseries not used in current model)
            else:
                raise ValueError(f"Unexpected batch size: {len(batch)}")
            
            corr_mat, label = corr_mat.to(device), label.to(device).float()
            val_samples += corr_mat.shape[0]

            # Pass correlation matrix through: CNN → ASPP → Quadratic → Atrous → Pooling → Classification
            output = model(corr_mat)
            y_idx = label.argmax(dim=1)

            loss = ce(output, y_idx)
            val_loss += loss.item()

            top1 = accuracy(output, y_idx)[0] / 100
            val_acc_list.append(top1)

            # Store full outputs and labels for macro metrics
            val_outputs.append(F.softmax(output, dim=1))
            val_labels.append(label)

    val_loss = val_loss / val_samples  # Use actual number of samples
    val_acc = np.mean(val_acc_list)
    
    # Concatenate all validation outputs and labels
    val_outputs = torch.cat(val_outputs, dim=0)
    val_labels = torch.cat(val_labels, dim=0)
    
    # Compute macro metrics for validation
    if args.dataset.lower().startswith('ppmi'):
        val_metrics = compute_macro_metrics(val_labels, val_outputs, args.dataset)
        val_roc = val_metrics["auc_macro"]
    else:
        # For binary datasets, use the original approach
        prob1 = val_outputs[:, 1].cpu().numpy()
        labels_binary = val_labels[:, 1].cpu().numpy()
        try:
            val_roc = roc_auc_score(labels_binary, prob1)
        except ValueError:
            val_roc = 0.5

    # ----- TEST -----
    test_loss = 0
    test_acc_list, test_outputs, test_labels = [], [], []
    test_samples = 0
    
    with torch.no_grad():  # Disable gradient computation for testing
        for batch in test_loader:
            # Handle both 2-value (ADHD) and 3-value (ABIDE with timeseries) dataloaders
            if len(batch) == 2:
                corr_mat, label = batch
            elif len(batch) == 3:
                timeseries, corr_mat, label = batch
                # For now, we only use correlation matrices (timeseries not used in current model)
            else:
                raise ValueError(f"Unexpected batch size: {len(batch)}")
            
            corr_mat, label = corr_mat.to(device), label.to(device).float()
            test_samples += corr_mat.shape[0]

            # Pass correlation matrix through: CNN → ASPP → Quadratic → Atrous → Pooling → Classification
            output = model(corr_mat)
            y_idx = label.argmax(dim=1)

            loss = ce(output, y_idx)
            test_loss += loss.item()

            top1 = accuracy(output, y_idx)[0] / 100
            test_acc_list.append(top1)

            # Store full outputs and labels for macro metrics
            test_outputs.append(F.softmax(output, dim=1))
            test_labels.append(label)

    test_loss = test_loss / test_samples  # Use actual number of samples
    test_acc = np.mean(test_acc_list)
    
    # Concatenate all test outputs and labels
    test_outputs = torch.cat(test_outputs, dim=0)
    test_labels = torch.cat(test_labels, dim=0)
    
    # Compute metrics based on dataset type
    if args.dataset.lower().startswith('ppmi'):
        # Use macro metrics for PPMI (multi-class, imbalanced)
        test_metrics = compute_macro_metrics(test_labels, test_outputs, args.dataset)
        test_roc = test_metrics["auc_macro"]
        test_sensitivity = test_metrics["recall_macro"]
        test_specificity = test_metrics["specificity_macro"]
        
        print(f"PPMI Macro Metrics:")
        print(f"  Macro AUC: {test_roc:.4f}")
        print(f"  Macro Recall: {test_sensitivity:.4f}")
        print(f"  Macro Specificity: {test_specificity:.4f}")
        print(f"  Macro AUPRC: {test_metrics['auprc_macro']:.4f}")
        print(f"  Per-class details: {test_metrics['per_class']}")
        
    else:
        # For binary datasets, use the original approach
        prob1 = test_outputs[:, 1].cpu().numpy()
        labels_binary = test_labels[:, 1].cpu().numpy()
        
        try:
            test_roc = roc_auc_score(labels_binary, prob1)
        except ValueError:
            test_roc = 0.5
        
        # Binary classification metrics
        result = (prob1 > 0.5).astype(int)
        labels = labels_binary
        
        report = classification_report(labels, result, output_dict=True, zero_division=0)
        
        recall = [0, 0]
        for k in report:
            try:
                class_idx = int(float(k))
                if 0 <= class_idx < len(recall):
                    recall[class_idx] = report[k]['recall']
            except (ValueError, IndexError):
                continue
        
        test_sensitivity = recall[-1]
        test_specificity = recall[-2]

    return {"val_loss": val_loss, "val_acc": val_acc, "val_roc": val_roc,
            "test_loss": test_loss, "test_acc": test_acc, "test_roc": test_roc,
            "test_sensitivity": test_sensitivity, "test_specificity": test_specificity}