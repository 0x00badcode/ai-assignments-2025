#!/usr/bin/env python3
"""
CNN Training Results Visualization
==================================
This script creates comprehensive visualizations of CNN training results
for reports and presentations.

Generated graphs:
1. Training/Validation Loss Curves
2. Training/Validation Accuracy Curves
3. Optimizer Comparison
4. Confusion Matrices
5. Training Summary Dashboard
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import seaborn as sns
from datetime import datetime

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams.update({
    'figure.figsize': (12, 8),
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18
})

def load_training_data():
    """Load training history and evaluation data from JSON files."""
    results_dir = Path('training_results')
    
    # Load training histories
    with open(results_dir / 'CNN_adam_history.json', 'r') as f:
        adam_history = json.load(f)
    
    with open(results_dir / 'CNN_sgd_history.json', 'r') as f:
        sgd_history = json.load(f)
    
    # Load evaluation results
    with open(results_dir / 'CNN_adam_evaluation.json', 'r') as f:
        adam_eval = json.load(f)
    
    with open(results_dir / 'CNN_sgd_evaluation.json', 'r') as f:
        sgd_eval = json.load(f)
    
    return adam_history, sgd_history, adam_eval, sgd_eval

def create_loss_curves(adam_history, sgd_history):
    """Create training and validation loss curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    epochs = adam_history['epochs']
    
    # Adam optimizer
    ax1.plot(epochs, adam_history['train_loss'], 'b-', linewidth=2, label='Training Loss', marker='o')
    ax1.plot(epochs, adam_history['val_loss'], 'r-', linewidth=2, label='Validation Loss', marker='s')
    ax1.set_title('CNN with Adam Optimizer - Loss Curves', fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)
    
    # SGD optimizer
    ax2.plot(epochs, sgd_history['train_loss'], 'g-', linewidth=2, label='Training Loss', marker='o')
    ax2.plot(epochs, sgd_history['val_loss'], 'orange', linewidth=2, label='Validation Loss', marker='s')
    ax2.set_title('CNN with SGD Optimizer - Loss Curves', fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig('training_visualizations/loss_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_accuracy_curves(adam_history, sgd_history):
    """Create training and validation accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    epochs = adam_history['epochs']
    
    # Adam optimizer
    ax1.plot(epochs, [acc * 100 for acc in adam_history['train_acc']], 'b-', 
             linewidth=2, label='Training Accuracy', marker='o')
    ax1.plot(epochs, [acc * 100 for acc in adam_history['val_acc']], 'r-', 
             linewidth=2, label='Validation Accuracy', marker='s')
    ax1.set_title('CNN with Adam Optimizer - Accuracy Curves', fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy (%)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 100)
    
    # SGD optimizer
    ax2.plot(epochs, [acc * 100 for acc in sgd_history['train_acc']], 'g-', 
             linewidth=2, label='Training Accuracy', marker='o')
    ax2.plot(epochs, [acc * 100 for acc in sgd_history['val_acc']], 'orange', 
             linewidth=2, label='Validation Accuracy', marker='s')
    ax2.set_title('CNN with SGD Optimizer - Accuracy Curves', fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig('training_visualizations/accuracy_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_optimizer_comparison(adam_history, sgd_history, adam_eval, sgd_eval):
    """Create comparison charts between optimizers."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    epochs = adam_history['epochs']
    
    # Combined Loss Comparison
    ax1.plot(epochs, adam_history['train_loss'], 'b-', linewidth=2, label='Adam Training', marker='o')
    ax1.plot(epochs, adam_history['val_loss'], 'b--', linewidth=2, label='Adam Validation', marker='s')
    ax1.plot(epochs, sgd_history['train_loss'], 'r-', linewidth=2, label='SGD Training', marker='o')
    ax1.plot(epochs, sgd_history['val_loss'], 'r--', linewidth=2, label='SGD Validation', marker='s')
    ax1.set_title('Loss Comparison: Adam vs SGD', fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Combined Accuracy Comparison
    ax2.plot(epochs, [acc * 100 for acc in adam_history['train_acc']], 'b-', 
             linewidth=2, label='Adam Training', marker='o')
    ax2.plot(epochs, [acc * 100 for acc in adam_history['val_acc']], 'b--', 
             linewidth=2, label='Adam Validation', marker='s')
    ax2.plot(epochs, [acc * 100 for acc in sgd_history['train_acc']], 'r-', 
             linewidth=2, label='SGD Training', marker='o')
    ax2.plot(epochs, [acc * 100 for acc in sgd_history['val_acc']], 'r--', 
             linewidth=2, label='SGD Validation', marker='s')
    ax2.set_title('Accuracy Comparison: Adam vs SGD', fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)
    
    # Final Performance Bar Chart
    optimizers = ['Adam', 'SGD']
    test_accuracies = [adam_eval['accuracy'] * 100, sgd_eval['accuracy'] * 100]
    val_accuracies = [max(adam_history['val_acc']) * 100, max(sgd_history['val_acc']) * 100]
    
    x = np.arange(len(optimizers))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, test_accuracies, width, label='Test Accuracy', alpha=0.8)
    bars2 = ax3.bar(x + width/2, val_accuracies, width, label='Best Validation Accuracy', alpha=0.8)
    
    ax3.set_title('Final Performance Comparison', fontweight='bold')
    ax3.set_ylabel('Accuracy (%)')
    ax3.set_xlabel('Optimizer')
    ax3.set_xticks(x)
    ax3.set_xticklabels(optimizers)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax3.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    for bar in bars2:
        height = bar.get_height()
        ax3.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    # Training Summary Table
    ax4.axis('tight')
    ax4.axis('off')
    
    summary_data = [
        ['Metric', 'Adam', 'SGD'],
        ['Test Accuracy', f'{adam_eval["accuracy"]*100:.1f}%', f'{sgd_eval["accuracy"]*100:.1f}%'],
        ['Best Val Accuracy', f'{max(adam_history["val_acc"])*100:.1f}%', f'{max(sgd_history["val_acc"])*100:.1f}%'],
        ['Final Train Accuracy', f'{adam_history["train_acc"][-1]*100:.1f}%', f'{sgd_history["train_acc"][-1]*100:.1f}%'],
        ['Final Loss', f'{adam_history["train_loss"][-1]:.3f}', f'{sgd_history["train_loss"][-1]:.3f}'],
        ['Training Epochs', '12', '12']
    ]
    
    table = ax4.table(cellText=summary_data, cellLoc='center', loc='center',
                     colWidths=[0.4, 0.3, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    
    # Style the header row
    for i in range(3):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax4.set_title('Training Summary', fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('training_visualizations/optimizer_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_confusion_matrices(adam_eval, sgd_eval):
    """Create confusion matrices for both models."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Class names (people in the dataset)
    class_names = ['Abdullah_Gul', 'Alejandro_Toledo', 'Alvaro_Uribe', 'Amelie_Mauresmo',
                   'Andre_Agassi', 'Andy_Roddick', 'Angelina_Jolie', 'Ariel_Sharon']
    
    # Adam confusion matrix
    cm_adam = np.array(adam_eval['confusion_matrix'])
    im1 = ax1.imshow(cm_adam, interpolation='nearest', cmap=plt.cm.Blues)
    ax1.set_title('CNN Adam - Confusion Matrix', fontweight='bold')
    ax1.set_xlabel('Predicted Label')
    ax1.set_ylabel('True Label')
    
    # Add colorbar
    plt.colorbar(im1, ax=ax1)
    
    # Add text annotations
    thresh = cm_adam.max() / 2.
    for i in range(cm_adam.shape[0]):
        for j in range(cm_adam.shape[1]):
            ax1.text(j, i, format(cm_adam[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm_adam[i, j] > thresh else "black",
                    fontsize=10)
    
    # Set tick labels
    tick_marks = np.arange(len(class_names))
    ax1.set_xticks(tick_marks)
    ax1.set_yticks(tick_marks)
    ax1.set_xticklabels([name.split('_')[0] for name in class_names], rotation=45, ha='right')
    ax1.set_yticklabels([name.split('_')[0] for name in class_names])
    
    # SGD confusion matrix
    cm_sgd = np.array(sgd_eval['confusion_matrix'])
    im2 = ax2.imshow(cm_sgd, interpolation='nearest', cmap=plt.cm.Reds)
    ax2.set_title('CNN SGD - Confusion Matrix', fontweight='bold')
    ax2.set_xlabel('Predicted Label')
    ax2.set_ylabel('True Label')
    
    # Add colorbar
    plt.colorbar(im2, ax=ax2)
    
    # Add text annotations
    thresh = cm_sgd.max() / 2.
    for i in range(cm_sgd.shape[0]):
        for j in range(cm_sgd.shape[1]):
            ax2.text(j, i, format(cm_sgd[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm_sgd[i, j] > thresh else "black",
                    fontsize=10)
    
    # Set tick labels
    ax2.set_xticks(tick_marks)
    ax2.set_yticks(tick_marks)
    ax2.set_xticklabels([name.split('_')[0] for name in class_names], rotation=45, ha='right')
    ax2.set_yticklabels([name.split('_')[0] for name in class_names])
    
    plt.tight_layout()
    plt.savefig('training_visualizations/confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_training_dashboard(adam_history, sgd_history, adam_eval, sgd_eval):
    """Create a comprehensive training dashboard."""
    fig = plt.figure(figsize=(20, 12))
    
    # Create a grid layout
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    epochs = adam_history['epochs']
    
    # Title
    fig.suptitle('CNN Face Recognition Training Results - Complete Dashboard', 
                fontsize=24, fontweight='bold', y=0.95)
    
    # 1. Training Loss
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, adam_history['train_loss'], 'b-', linewidth=2, label='Adam', marker='o')
    ax1.plot(epochs, sgd_history['train_loss'], 'r-', linewidth=2, label='SGD', marker='s')
    ax1.set_title('Training Loss', fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Validation Loss
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs, adam_history['val_loss'], 'b-', linewidth=2, label='Adam', marker='o')
    ax2.plot(epochs, sgd_history['val_loss'], 'r-', linewidth=2, label='SGD', marker='s')
    ax2.set_title('Validation Loss', fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Training Accuracy
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(epochs, [acc * 100 for acc in adam_history['train_acc']], 'b-', 
             linewidth=2, label='Adam', marker='o')
    ax3.plot(epochs, [acc * 100 for acc in sgd_history['train_acc']], 'r-', 
             linewidth=2, label='SGD', marker='s')
    ax3.set_title('Training Accuracy', fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Accuracy (%)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 100)
    
    # 4. Validation Accuracy
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.plot(epochs, [acc * 100 for acc in adam_history['val_acc']], 'b-', 
             linewidth=2, label='Adam', marker='o')
    ax4.plot(epochs, [acc * 100 for acc in sgd_history['val_acc']], 'r-', 
             linewidth=2, label='SGD', marker='s')
    ax4.set_title('Validation Accuracy', fontweight='bold')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Accuracy (%)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 100)
    
    # 5. Performance Comparison
    ax5 = fig.add_subplot(gs[1, :2])
    optimizers = ['Adam', 'SGD']
    test_acc = [adam_eval['accuracy'] * 100, sgd_eval['accuracy'] * 100]
    val_acc = [max(adam_history['val_acc']) * 100, max(sgd_history['val_acc']) * 100]
    train_acc = [adam_history['train_acc'][-1] * 100, sgd_history['train_acc'][-1] * 100]
    
    x = np.arange(len(optimizers))
    width = 0.25
    
    bars1 = ax5.bar(x - width, test_acc, width, label='Test Accuracy', alpha=0.8)
    bars2 = ax5.bar(x, val_acc, width, label='Best Val Accuracy', alpha=0.8)
    bars3 = ax5.bar(x + width, train_acc, width, label='Final Train Accuracy', alpha=0.8)
    
    ax5.set_title('Performance Comparison', fontweight='bold')
    ax5.set_ylabel('Accuracy (%)')
    ax5.set_xlabel('Optimizer')
    ax5.set_xticks(x)
    ax5.set_xticklabels(optimizers)
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax5.annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
    
    # 6. Dataset Information
    ax6 = fig.add_subplot(gs[1, 2:])
    ax6.axis('off')
    
    dataset_info = """
    📊 DATASET INFORMATION
    • Dataset: LFW (Labeled Faces in the Wild)
    • Total Images: 154
    • Number of People: 8
    • Train/Val/Test Split: 107/23/24
    • Image Resolution: Preprocessed
    • Task: Face Recognition (8-class classification)
    
    🏗️ CNN ARCHITECTURE
    • Convolutional layers with ReLU activation
    • Max pooling for dimensionality reduction
    • Fully connected layers for classification
    • Dropout for regularization
    • Softmax output for multi-class classification
    """
    
    ax6.text(0.05, 0.95, dataset_info, transform=ax6.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    # 7. Key Metrics Summary
    ax7 = fig.add_subplot(gs[2, :])
    ax7.axis('tight')
    ax7.axis('off')
    
    summary_data = [
        ['Metric', 'Adam Optimizer', 'SGD Optimizer', 'Improvement'],
        ['Test Accuracy', f'{adam_eval["accuracy"]*100:.1f}%', f'{sgd_eval["accuracy"]*100:.1f}%', 
         f'+{(adam_eval["accuracy"] - sgd_eval["accuracy"])*100:.1f}%'],
        ['Best Validation Accuracy', f'{max(adam_history["val_acc"])*100:.1f}%', 
         f'{max(sgd_history["val_acc"])*100:.1f}%',
         f'+{(max(adam_history["val_acc"]) - max(sgd_history["val_acc"]))*100:.1f}%'],
        ['Final Training Loss', f'{adam_history["train_loss"][-1]:.3f}', 
         f'{sgd_history["train_loss"][-1]:.3f}',
         f'{sgd_history["train_loss"][-1] - adam_history["train_loss"][-1]:.3f}'],
        ['Training Time', '~1.7 min', '~1.7 min', 'Equal'],
        ['Convergence', 'Fast & Stable', 'Slow & Unstable', 'Adam Superior']
    ]
    
    table = ax7.table(cellText=summary_data, cellLoc='center', loc='center',
                     colWidths=[0.3, 0.2, 0.2, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2)
    
    # Style the header row
    for i in range(4):
        table[(0, i)].set_facecolor('#2E86AB')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color code the improvement column
    for i in range(1, 6):
        if 'Adam Superior' in summary_data[i][3] or '+' in summary_data[i][3]:
            table[(i, 3)].set_facecolor('#A8E6CF')  # Light green for positive
        elif 'Equal' in summary_data[i][3]:
            table[(i, 3)].set_facecolor('#FFE5B4')  # Light yellow for neutral
    
    plt.savefig('training_visualizations/training_dashboard.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function to generate all visualizations."""
    print("🎨 Creating CNN Training Visualizations")
    print("=" * 50)
    
    # Create output directory
    Path('training_visualizations').mkdir(exist_ok=True)
    
    # Load data
    print("📊 Loading training data...")
    adam_history, sgd_history, adam_eval, sgd_eval = load_training_data()
    
    # Generate visualizations
    print("📈 Creating loss curves...")
    create_loss_curves(adam_history, sgd_history)
    
    print("📈 Creating accuracy curves...")
    create_accuracy_curves(adam_history, sgd_history)
    
    print("📊 Creating optimizer comparison...")
    create_optimizer_comparison(adam_history, sgd_history, adam_eval, sgd_eval)
    
    print("🔍 Creating confusion matrices...")
    create_confusion_matrices(adam_eval, sgd_eval)
    
    print("📊 Creating comprehensive dashboard...")
    create_training_dashboard(adam_history, sgd_history, adam_eval, sgd_eval)
    
    print("\n✅ All visualizations created successfully!")
    print("📁 Check the 'training_visualizations/' directory for all graphs")
    print("\n📊 Generated files:")
    print("  • loss_curves.png")
    print("  • accuracy_curves.png") 
    print("  • optimizer_comparison.png")
    print("  • confusion_matrices.png")
    print("  • training_dashboard.png")
    print("\n🎉 Ready for your report!")

if __name__ == "__main__":
    main() 