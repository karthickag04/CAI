"""
Metrics Utilities for OCR Evaluation
====================================

This module provides evaluation metrics for OCR models including:
- Character Error Rate (CER)
- Word Error Rate (WER)
- Exact Match Accuracy
- BLEU Score
- Edit Distance calculations

These metrics are essential for evaluating the performance of OCR models
and comparing different approaches.

Author: OCR Project Extended
Date: July 2025
"""

import editdistance
import numpy as np
from typing import List, Tuple
import string
import re


def normalize_text(text: str) -> str:
    """
    Normalize text for fair comparison.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Normalized text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove punctuation (optional - can be configurable)
    # text = text.translate(str.maketrans('', '', string.punctuation))
    
    return text


def calculate_cer(predictions: List[str], targets: List[str], normalize: bool = True) -> float:
    """
    Calculate Character Error Rate (CER).
    
    CER = (Substitutions + Insertions + Deletions) / Total Characters in Target
    
    Args:
        predictions (List[str]): List of predicted texts
        targets (List[str]): List of target texts
        normalize (bool): Whether to normalize texts before comparison
        
    Returns:
        float: Character Error Rate (0.0 to 1.0+)
    """
    if len(predictions) != len(targets):
        raise ValueError("Number of predictions and targets must match")
    
    total_chars = 0
    total_errors = 0
    
    for pred, target in zip(predictions, targets):
        if normalize:
            pred = normalize_text(pred)
            target = normalize_text(target)
        
        # Calculate edit distance (number of character-level operations)
        errors = editdistance.eval(pred, target)
        
        # Update totals
        total_errors += errors
        total_chars += len(target)
    
    # Avoid division by zero
    if total_chars == 0:
        return 0.0 if total_errors == 0 else float('inf')
    
    return total_errors / total_chars


def calculate_wer(predictions: List[str], targets: List[str], normalize: bool = True) -> float:
    """
    Calculate Word Error Rate (WER).
    
    WER = (Substitutions + Insertions + Deletions) / Total Words in Target
    
    Args:
        predictions (List[str]): List of predicted texts
        targets (List[str]): List of target texts
        normalize (bool): Whether to normalize texts before comparison
        
    Returns:
        float: Word Error Rate (0.0 to 1.0+)
    """
    if len(predictions) != len(targets):
        raise ValueError("Number of predictions and targets must match")
    
    total_words = 0
    total_errors = 0
    
    for pred, target in zip(predictions, targets):
        if normalize:
            pred = normalize_text(pred)
            target = normalize_text(target)
        
        # Split into words
        pred_words = pred.split()
        target_words = target.split()
        
        # Calculate edit distance at word level
        errors = editdistance.eval(pred_words, target_words)
        
        # Update totals
        total_errors += errors
        total_words += len(target_words)
    
    # Avoid division by zero
    if total_words == 0:
        return 0.0 if total_errors == 0 else float('inf')
    
    return total_errors / total_words


def calculate_accuracy(predictions: List[str], targets: List[str], normalize: bool = True) -> float:
    """
    Calculate exact match accuracy.
    
    Args:
        predictions (List[str]): List of predicted texts
        targets (List[str]): List of target texts
        normalize (bool): Whether to normalize texts before comparison
        
    Returns:
        float: Accuracy (0.0 to 1.0)
    """
    if len(predictions) != len(targets):
        raise ValueError("Number of predictions and targets must match")
    
    if len(predictions) == 0:
        return 0.0
    
    correct = 0
    
    for pred, target in zip(predictions, targets):
        if normalize:
            pred = normalize_text(pred)
            target = normalize_text(target)
        
        if pred == target:
            correct += 1
    
    return correct / len(predictions)


def calculate_bleu_score(predictions: List[str], targets: List[str], n_gram: int = 4) -> float:
    """
    Calculate BLEU score for OCR evaluation.
    
    Args:
        predictions (List[str]): List of predicted texts
        targets (List[str]): List of target texts
        n_gram (int): Maximum n-gram order
        
    Returns:
        float: BLEU score (0.0 to 1.0)
    """
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        
        if len(predictions) != len(targets):
            raise ValueError("Number of predictions and targets must match")
        
        if len(predictions) == 0:
            return 0.0
        
        smoothing = SmoothingFunction().method1
        total_score = 0
        
        for pred, target in zip(predictions, targets):
            # Tokenize
            pred_tokens = pred.split()
            target_tokens = [target.split()]  # BLEU expects list of reference lists
            
            # Calculate BLEU score for this pair
            score = sentence_bleu(target_tokens, pred_tokens, 
                                smoothing_function=smoothing,
                                weights=[1/n_gram] * n_gram)
            total_score += score
        
        return total_score / len(predictions)
    
    except ImportError:
        print("NLTK not available. Install with: pip install nltk")
        return 0.0


def calculate_detailed_metrics(predictions: List[str], targets: List[str]) -> dict:
    """
    Calculate comprehensive metrics for OCR evaluation.
    
    Args:
        predictions (List[str]): List of predicted texts
        targets (List[str]): List of target texts
        
    Returns:
        dict: Dictionary containing all metrics
    """
    metrics = {
        'accuracy': calculate_accuracy(predictions, targets),
        'cer': calculate_cer(predictions, targets),
        'wer': calculate_wer(predictions, targets),
        'bleu_score': calculate_bleu_score(predictions, targets),
        'num_samples': len(predictions)
    }
    
    # Calculate per-sample metrics for analysis
    sample_metrics = []
    for pred, target in zip(predictions, targets):
        sample_cer = calculate_cer([pred], [target])
        sample_wer = calculate_wer([pred], [target])
        sample_acc = calculate_accuracy([pred], [target])
        
        sample_metrics.append({
            'prediction': pred,
            'target': target,
            'cer': sample_cer,
            'wer': sample_wer,
            'accuracy': sample_acc,
            'match': pred.lower().strip() == target.lower().strip()
        })
    
    metrics['sample_metrics'] = sample_metrics
    
    # Calculate statistics
    cer_values = [m['cer'] for m in sample_metrics]
    wer_values = [m['wer'] for m in sample_metrics]
    
    metrics['cer_std'] = np.std(cer_values)
    metrics['wer_std'] = np.std(wer_values)
    metrics['cer_median'] = np.median(cer_values)
    metrics['wer_median'] = np.median(wer_values)
    
    return metrics


def print_metrics_report(metrics: dict, title: str = "OCR Evaluation Report"):
    """
    Print a formatted metrics report.
    
    Args:
        metrics (dict): Metrics dictionary from calculate_detailed_metrics
        title (str): Report title
    """
    print(f"\n{'='*60}")
    print(f"{title:^60}")
    print(f"{'='*60}")
    
    print(f"\n📊 Overall Metrics:")
    print(f"   Number of samples: {metrics['num_samples']}")
    print(f"   Exact Match Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"   Character Error Rate (CER): {metrics['cer']:.4f}")
    print(f"   Word Error Rate (WER): {metrics['wer']:.4f}")
    print(f"   BLEU Score: {metrics['bleu_score']:.4f}")
    
    print(f"\n📈 Statistical Analysis:")
    print(f"   CER - Mean: {metrics['cer']:.4f}, Std: {metrics['cer_std']:.4f}, Median: {metrics['cer_median']:.4f}")
    print(f"   WER - Mean: {metrics['wer']:.4f}, Std: {metrics['wer_std']:.4f}, Median: {metrics['wer_median']:.4f}")
    
    # Show worst performing samples
    sample_metrics = metrics['sample_metrics']
    worst_samples = sorted(sample_metrics, key=lambda x: x['cer'], reverse=True)[:3]
    
    print(f"\n❌ Worst Performing Samples (by CER):")
    for i, sample in enumerate(worst_samples, 1):
        print(f"   {i}. CER: {sample['cer']:.4f}")
        print(f"      Target:     '{sample['target']}'")
        print(f"      Prediction: '{sample['prediction']}'")
        print()
    
    # Show best performing samples
    best_samples = [s for s in sample_metrics if s['match']][:3]
    if best_samples:
        print(f"✅ Perfect Matches (examples):")
        for i, sample in enumerate(best_samples, 1):
            print(f"   {i}. '{sample['target']}'")


class MetricsTracker:
    """
    Class for tracking metrics over training epochs.
    """
    
    def __init__(self):
        """Initialize metrics tracker."""
        self.epoch_metrics = []
        self.best_metrics = None
        self.best_epoch = -1
    
    def update(self, epoch: int, predictions: List[str], targets: List[str]):
        """
        Update metrics for current epoch.
        
        Args:
            epoch (int): Current epoch number
            predictions (List[str]): Predicted texts
            targets (List[str]): Target texts
        """
        metrics = calculate_detailed_metrics(predictions, targets)
        metrics['epoch'] = epoch
        
        self.epoch_metrics.append(metrics)
        
        # Update best metrics
        if self.best_metrics is None or metrics['accuracy'] > self.best_metrics['accuracy']:
            self.best_metrics = metrics.copy()
            self.best_epoch = epoch
    
    def get_best_metrics(self) -> dict:
        """Get best metrics achieved so far."""
        return self.best_metrics
    
    def print_progress(self, epoch: int):
        """Print progress for current epoch."""
        if not self.epoch_metrics:
            return
        
        current_metrics = self.epoch_metrics[-1]
        print(f"\nEpoch {epoch} Metrics:")
        print(f"  Accuracy: {current_metrics['accuracy']:.4f}")
        print(f"  CER: {current_metrics['cer']:.4f}")
        print(f"  WER: {current_metrics['wer']:.4f}")
        
        if self.best_metrics:
            print(f"  Best Accuracy: {self.best_metrics['accuracy']:.4f} (Epoch {self.best_epoch})")


def compare_models(model_results: dict, metric: str = 'accuracy'):
    """
    Compare multiple models based on specified metric.
    
    Args:
        model_results (dict): Dictionary of model names to metrics
        metric (str): Metric to compare by ('accuracy', 'cer', 'wer', 'bleu_score')
    """
    print(f"\n🏆 Model Comparison (by {metric.upper()}):")
    print("=" * 60)
    
    # Sort models by metric (higher is better for accuracy, bleu; lower for cer, wer)
    reverse_sort = metric in ['accuracy', 'bleu_score']
    sorted_models = sorted(model_results.items(), 
                          key=lambda x: x[1][metric], 
                          reverse=reverse_sort)
    
    for rank, (model_name, metrics) in enumerate(sorted_models, 1):
        print(f"{rank}. {model_name}")
        print(f"   {metric.upper()}: {metrics[metric]:.4f}")
        print(f"   Accuracy: {metrics['accuracy']:.4f}")
        print(f"   CER: {metrics['cer']:.4f}")
        print(f"   WER: {metrics['wer']:.4f}")
        print()


if __name__ == "__main__":
    # Example usage and testing
    print("Testing OCR Metrics")
    print("=" * 40)
    
    # Example predictions and targets
    predictions = [
        "hello world",
        "machine learning",
        "deep neural networks",
        "computer vison",  # Error: "vision" -> "vison"
        "artificial intelligence"
    ]
    
    targets = [
        "hello world",
        "machine learning",
        "deep neural networks", 
        "computer vision",
        "artificial intelligence"
    ]
    
    # Calculate metrics
    metrics = calculate_detailed_metrics(predictions, targets)
    
    # Print report
    print_metrics_report(metrics, "Example OCR Evaluation")
    
    print("\n✅ Metrics utilities test completed successfully!")
