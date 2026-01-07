"""Real-time training analysis for currently running experiments."""
import re
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def parse_current_log(log_path):
    """Parse validation checkpoints from ModelCheckpoint callbacks."""
    epochs = []
    val_dices = []
    best_dices = []
    
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Pattern: Epoch X, global step Y: 'val/dice' reached Z (best W)
            match = re.search(r"Epoch (\d+), global step \d+: 'val/dice' reached ([\d.]+) \(best ([\d.]+)\)", line)
            if match:
                epoch = int(match.group(1))
                val_dice = float(match.group(2))
                best_dice = float(match.group(3))
                
                epochs.append(epoch)
                val_dices.append(val_dice)
                best_dices.append(best_dice)
    
    return np.array(epochs), np.array(val_dices), np.array(best_dices)

def analyze_and_plot(log_path, output_dir):
    """Analyze current training progress."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("📊 REAL-TIME TRAINING ANALYSIS")
    print("="*80)
    print(f"Log: {log_path}")
    
    epochs, val_dices, best_dices = parse_current_log(log_path)
    
    if len(epochs) == 0:
        print("\n❌ No validation data found yet. Training may not have reached first validation checkpoint.")
        return
    
    print(f"\n✅ Found {len(epochs)} validation checkpoints")
    print(f"   Epoch range: {epochs[0]} - {epochs[-1]}")
    print(f"   Current best Dice: {best_dices[-1]:.4f} @ Epoch {epochs[np.argmax(val_dices)]}")
    
    # Statistics
    print("\n📊 CURRENT STATISTICS")
    print("="*80)
    print(f"Initial Dice (Epoch {epochs[0]}): {val_dices[0]:.4f}")
    print(f"Current Dice (Epoch {epochs[-1]}): {val_dices[-1]:.4f}")
    print(f"Best Dice:    {best_dices[-1]:.4f}")
    print(f"Improvement:  +{val_dices[-1] - val_dices[0]:.4f} ({((val_dices[-1] - val_dices[0]) / val_dices[0] * 100):.1f}%)")
    
    # Stability (last 5 checkpoints if available)
    if len(val_dices) >= 5:
        recent_std = np.std(val_dices[-5:])
        print(f"Recent stability (last 5 checkpoints): std = {recent_std:.4f}")
        
        # Trend
        recent_trend = val_dices[-1] - val_dices[-5]
        if recent_trend > 0.01:
            print(f"  🟢 Improving: +{recent_trend:.4f} improvement")
        elif recent_trend < -0.01:
            print(f"  🔴 Declining: {recent_trend:.4f} decrease")
        else:
            print(f"  🟡 Plateau: {abs(recent_trend):.4f} change (stable)")
    
    # Visualization
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    ax.plot(epochs, val_dices, 'o-', label='Validation Dice', linewidth=2, markersize=6)
    ax.axhline(y=best_dices[-1], color='r', linestyle='--', alpha=0.7, 
               label=f'Best: {best_dices[-1]:.4f}')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Dice Score', fontsize=12)
    ax.set_title('Real-Time Training Progress', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Annotate best point
    best_idx = np.argmax(val_dices)
    ax.annotate(f'Best: {val_dices[best_idx]:.4f}\n@ Epoch {epochs[best_idx]}',
                xy=(epochs[best_idx], val_dices[best_idx]),
                xytext=(10, 10), textcoords='offset points',
                fontsize=9, color='red',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='red'))
    
    plt.tight_layout()
    output_file = output_dir / 'current_training_progress.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✅ Plot saved: {output_file}")
    
    # Issue detection
    print("\n🔍 ISSUE DETECTION")
    print("="*80)
    
    issues = []
    
    # Check for plateau
    if len(val_dices) >= 10:
        last_10_improvement = val_dices[-1] - val_dices[-10]
        if abs(last_10_improvement) < 0.005:
            issues.append("🟡 PLATEAU: Less than 0.5% improvement in last 10 checkpoints")
    
    # Check for decline
    if len(val_dices) >= 5:
        if val_dices[-1] < val_dices[-5] - 0.01:
            issues.append("🔴 DECLINE: Performance dropped >1% in recent checkpoints")
    
    # Check if best was early
    best_epoch_idx = np.argmax(val_dices)
    if best_epoch_idx < len(val_dices) * 0.3:
        issues.append(f"🟡 EARLY_PEAK: Best performance at epoch {epochs[best_epoch_idx]} (early in training)")
    
    if issues:
        for issue in issues:
            print(f"  {issue}")
    else:
        print("  🟢 No issues detected - training looks healthy!")
    
    print("\n✅ Analysis complete!")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-file', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='results/evaluation/current')
    args = parser.parse_args()
    
    analyze_and_plot(args.log_file, args.output_dir)
