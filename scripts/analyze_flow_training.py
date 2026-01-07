"""
Flow Model Training Analysis Script
분석 항목:
1. Loss 및 Validation 지표 추출 및 시각화
2. 학습 진행 패턴 분석
3. Overfitting 여부 확인
4. Loss component별 기여도 분석
"""
import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

def parse_training_log(log_path):
    """로그 파일에서 학습 및 검증 지표 추출"""
    epochs = []
    train_losses = []
    val_dices = []
    val_ious = []
    val_precisions = []
    val_recalls = []
    val_specificities = []
    val_recon_losses = []
    
    # Best performance tracking
    best_scores = []
    
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Epoch 완료 라인 파싱 (validation metrics 포함)
            if 'Epoch' in line and '100%' in line and 'val/dice' in line:
                # Extract epoch number
                epoch_match = re.search(r'Epoch (\d+):', line)
                if not epoch_match:
                    continue
                epoch = int(epoch_match.group(1))
                
                # Extract train/loss
                train_loss_match = re.search(r'train/loss=([\d.]+)', line)
                if train_loss_match:
                    train_loss = float(train_loss_match.group(1))
                else:
                    continue
                
                # Extract validation metrics
                val_dice_match = re.search(r'val/dice=([\d.]+)', line)
                val_iou_match = re.search(r'val/iou=([\d.]+)', line)
                val_precision_match = re.search(r'val/precision=([\d.]+)', line)
                val_recall_match = re.search(r'val/recall=([\d.]+)', line)
                val_spec_match = re.search(r'val/specificity=([\d.]+)', line)
                val_recon_match = re.search(r'val/reconstruction_loss=([\d.]+)', line)
                
                if all([val_dice_match, val_iou_match, val_precision_match, 
                       val_recall_match, val_spec_match, val_recon_match]):
                    epochs.append(epoch)
                    train_losses.append(train_loss)
                    val_dices.append(float(val_dice_match.group(1)))
                    val_ious.append(float(val_iou_match.group(1)))
                    val_precisions.append(float(val_precision_match.group(1)))
                    val_recalls.append(float(val_recall_match.group(1)))
                    val_specificities.append(float(val_spec_match.group(1)))
                    val_recon_losses.append(float(val_recon_match.group(1)))
            
            # Best score tracking
            if 'val/dice\' reached' in line and 'best' in line:
                score_match = re.search(r"'val/dice' reached ([\d.]+) \(best ([\d.]+)\)", line)
                if score_match:
                    current = float(score_match.group(1))
                    best = float(score_match.group(2))
                    best_scores.append((current, best))
    
    return {
        'epochs': np.array(epochs),
        'train_losses': np.array(train_losses),
        'val_dices': np.array(val_dices),
        'val_ious': np.array(val_ious),
        'val_precisions': np.array(val_precisions),
        'val_recalls': np.array(val_recalls),
        'val_specificities': np.array(val_specificities),
        'val_recon_losses': np.array(val_recon_losses),
        'best_scores': best_scores
    }


def analyze_training_progress(data, save_path):
    """학습 진행 패턴 분석 및 시각화"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Flow Model Training Analysis - XCA Dataset', fontsize=16, fontweight='bold')
    
    epochs = data['epochs']
    
    # 1. Training Loss
    ax = axes[0, 0]
    ax.plot(epochs, data['train_losses'], 'b-', linewidth=2, alpha=0.7, label='Train Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 2. Validation Dice Score
    ax = axes[0, 1]
    ax.plot(epochs, data['val_dices'], 'g-', linewidth=2, alpha=0.7, label='Val Dice')
    ax.axhline(y=data['val_dices'].max(), color='r', linestyle='--', 
               label=f'Best: {data["val_dices"].max():.4f}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Dice Score')
    ax.set_title('Validation Dice Score')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 3. Validation IoU
    ax = axes[0, 2]
    ax.plot(epochs, data['val_ious'], 'orange', linewidth=2, alpha=0.7, label='Val IoU')
    ax.axhline(y=data['val_ious'].max(), color='r', linestyle='--', 
               label=f'Best: {data["val_ious"].max():.4f}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('IoU')
    ax.set_title('Validation IoU')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 4. Precision vs Recall
    ax = axes[1, 0]
    ax.plot(epochs, data['val_precisions'], 'purple', linewidth=2, alpha=0.7, label='Precision')
    ax.plot(epochs, data['val_recalls'], 'brown', linewidth=2, alpha=0.7, label='Recall')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Score')
    ax.set_title('Validation Precision vs Recall')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 5. Reconstruction Loss
    ax = axes[1, 1]
    ax.plot(epochs, data['val_recon_losses'], 'red', linewidth=2, alpha=0.7, label='Recon Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Validation Reconstruction Loss')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 6. Training Loss vs Val Dice (Overfitting 확인)
    ax = axes[1, 2]
    ax2 = ax.twinx()
    l1 = ax.plot(epochs, data['train_losses'], 'b-', linewidth=2, alpha=0.7, label='Train Loss')
    l2 = ax2.plot(epochs, data['val_dices'], 'g-', linewidth=2, alpha=0.7, label='Val Dice')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Train Loss', color='b')
    ax2.set_ylabel('Val Dice', color='g')
    ax.set_title('Overfitting Check: Train Loss vs Val Dice')
    ax.tick_params(axis='y', labelcolor='b')
    ax2.tick_params(axis='y', labelcolor='g')
    
    # Combine legends
    lns = l1 + l2
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Training analysis plot saved to: {save_path}")
    plt.close()


def analyze_performance_statistics(data):
    """성능 통계 분석"""
    print("\n" + "="*80)
    print("📊 PERFORMANCE STATISTICS")
    print("="*80)
    
    epochs = data['epochs']
    
    # 초기 성능 (첫 10 epoch)
    print("\n🔵 Initial Performance (First 10 epochs):")
    if len(epochs) >= 10:
        print(f"  Train Loss: {data['train_losses'][:10].mean():.4f} ± {data['train_losses'][:10].std():.4f}")
        print(f"  Val Dice:   {data['val_dices'][:10].mean():.4f} ± {data['val_dices'][:10].std():.4f}")
        print(f"  Val IoU:    {data['val_ious'][:10].mean():.4f} ± {data['val_ious'][:10].std():.4f}")
    
    # 최종 성능 (마지막 10 epoch)
    print("\n🟢 Final Performance (Last 10 epochs):")
    print(f"  Train Loss: {data['train_losses'][-10:].mean():.4f} ± {data['train_losses'][-10:].std():.4f}")
    print(f"  Val Dice:   {data['val_dices'][-10:].mean():.4f} ± {data['val_dices'][-10:].std():.4f}")
    print(f"  Val IoU:    {data['val_ious'][-10:].mean():.4f} ± {data['val_ious'][-10:].std():.4f}")
    
    # Best 성능
    best_epoch = epochs[np.argmax(data['val_dices'])]
    best_idx = np.argmax(data['val_dices'])
    
    print(f"\n🏆 Best Performance (Epoch {best_epoch}):")
    print(f"  Train Loss:   {data['train_losses'][best_idx]:.4f}")
    print(f"  Val Dice:     {data['val_dices'][best_idx]:.4f}")
    print(f"  Val IoU:      {data['val_ious'][best_idx]:.4f}")
    print(f"  Val Precision: {data['val_precisions'][best_idx]:.4f}")
    print(f"  Val Recall:    {data['val_recalls'][best_idx]:.4f}")
    print(f"  Val Specificity: {data['val_specificities'][best_idx]:.4f}")
    
    # 개선 분석
    initial_dice = data['val_dices'][:10].mean() if len(epochs) >= 10 else data['val_dices'][0]
    final_dice = data['val_dices'][-10:].mean()
    improvement = final_dice - initial_dice
    
    print(f"\n📈 Overall Improvement:")
    print(f"  Dice Score: {initial_dice:.4f} → {final_dice:.4f} (+{improvement:.4f}, +{improvement/initial_dice*100:.2f}%)")
    
    # Training stability
    print(f"\n📊 Training Stability (Last 100 epochs):")
    recent_dice = data['val_dices'][-100:]
    recent_loss = data['train_losses'][-100:]
    print(f"  Val Dice Std: {recent_dice.std():.4f} (lower is better)")
    print(f"  Train Loss Std: {recent_loss.std():.4f}")
    
    # Overfitting 체크
    print(f"\n⚠️  Overfitting Analysis:")
    train_loss_trend = np.polyfit(epochs[-100:], data['train_losses'][-100:], 1)[0]
    val_dice_trend = np.polyfit(epochs[-100:], data['val_dices'][-100:], 1)[0]
    
    print(f"  Train Loss Trend (last 100 epochs): {train_loss_trend:.6f} (negative = decreasing)")
    print(f"  Val Dice Trend (last 100 epochs): {val_dice_trend:.6f} (positive = improving)")
    
    if train_loss_trend < 0 and val_dice_trend < 0:
        print("  🔴 WARNING: Train loss decreasing but val dice decreasing → Overfitting!")
    elif train_loss_trend < 0 and val_dice_trend > 0:
        print("  🟢 GOOD: Both train loss and val dice improving → Healthy training")
    elif abs(train_loss_trend) < 0.0001 and abs(val_dice_trend) < 0.0001:
        print("  🟡 PLATEAU: Training has plateaued → May need LR adjustment or early stopping")
    
    return {
        'best_epoch': best_epoch,
        'best_dice': data['val_dices'][best_idx],
        'improvement': improvement
    }


def detect_issues(data):
    """학습 문제점 진단"""
    print("\n" + "="*80)
    print("🔍 ISSUE DETECTION")
    print("="*80)
    
    issues = []
    
    # 1. Early peak detection (조기에 성능이 피크했다가 떨어지는 경우)
    max_dice_epoch = data['epochs'][np.argmax(data['val_dices'])]
    if max_dice_epoch < len(data['epochs']) * 0.5:
        issues.append({
            'type': 'EARLY_PEAK',
            'severity': 'HIGH',
            'message': f"Best performance at epoch {max_dice_epoch} (early in training)",
            'recommendation': "Consider: 1) Reduce learning rate, 2) Add LR scheduler, 3) Check for overfitting"
        })
    
    # 2. High variance (불안정한 학습)
    recent_dice_std = data['val_dices'][-100:].std()
    if recent_dice_std > 0.02:
        issues.append({
            'type': 'HIGH_VARIANCE',
            'severity': 'MEDIUM',
            'message': f"High variance in validation dice: {recent_dice_std:.4f}",
            'recommendation': "Consider: 1) Reduce learning rate, 2) Increase batch size, 3) Add gradient clipping"
        })
    
    # 3. Precision-Recall imbalance
    final_precision = data['val_precisions'][-10:].mean()
    final_recall = data['val_recalls'][-10:].mean()
    pr_ratio = final_precision / final_recall if final_recall > 0 else 1.0
    
    if pr_ratio > 1.2:
        issues.append({
            'type': 'HIGH_PRECISION_LOW_RECALL',
            'severity': 'MEDIUM',
            'message': f"Precision ({final_precision:.4f}) >> Recall ({final_recall:.4f})",
            'recommendation': "Model is conservative (under-segmenting). Consider: 1) Adjust loss weights, 2) Lower threshold"
        })
    elif pr_ratio < 0.8:
        issues.append({
            'type': 'LOW_PRECISION_HIGH_RECALL',
            'severity': 'MEDIUM',
            'message': f"Precision ({final_precision:.4f}) << Recall ({final_recall:.4f})",
            'recommendation': "Model is aggressive (over-segmenting). Consider: 1) Adjust loss weights, 2) Higher threshold"
        })
    
    # 4. Low absolute performance
    best_dice = data['val_dices'].max()
    if best_dice < 0.7:
        issues.append({
            'type': 'LOW_PERFORMANCE',
            'severity': 'HIGH',
            'message': f"Best Dice score is low: {best_dice:.4f}",
            'recommendation': "Consider: 1) Check data quality, 2) Increase model capacity, 3) Adjust loss function"
        })
    
    # 5. Plateau detection (성능 향상 정체)
    last_50_improvement = data['val_dices'][-1] - data['val_dices'][-50]
    if abs(last_50_improvement) < 0.005:
        issues.append({
            'type': 'PLATEAU',
            'severity': 'LOW',
            'message': f"Training has plateaued (last 50 epochs improvement: {last_50_improvement:.4f})",
            'recommendation': "Consider: 1) Learning rate reduction, 2) Early stopping, 3) Architecture changes"
        })
    
    # Print issues
    if not issues:
        print("\n✅ No major issues detected! Training looks healthy.")
    else:
        for i, issue in enumerate(issues, 1):
            severity_emoji = {'HIGH': '🔴', 'MEDIUM': '🟡', 'LOW': '🔵'}[issue['severity']]
            print(f"\n{i}. {severity_emoji} [{issue['severity']}] {issue['type']}")
            print(f"   Message: {issue['message']}")
            print(f"   💡 {issue['recommendation']}")
    
    return issues


def analyze_flow_sauna_specifics(log_path):
    """Flow-SAUNA 특화 분석"""
    print("\n" + "="*80)
    print("🌊 FLOW-SAUNA SPECIFIC ANALYSIS")
    print("="*80)
    
    # SAUNA soft label 정보 추출
    sauna_stats = defaultdict(list)
    
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            if 'use_sauna_transform=True' in line and 'unique:' in line:
                # Extract unique value count
                unique_match = re.search(r'unique: (\d+)', line)
                if unique_match:
                    sauna_stats['unique_values'].append(int(unique_match.group(1)))
    
    if sauna_stats['unique_values']:
        unique_values = np.array(sauna_stats['unique_values'])
        print(f"\n📊 SAUNA Soft Label Statistics:")
        print(f"  Unique values per sample: {unique_values.mean():.1f} ± {unique_values.std():.1f}")
        print(f"  Min unique values: {unique_values.min()}")
        print(f"  Max unique values: {unique_values.max()}")
        print(f"  Total samples analyzed: {len(unique_values)}")
        
        # Soft label richness 분석
        if unique_values.mean() < 500:
            print(f"\n  ⚠️  Low soft label richness (avg {unique_values.mean():.0f} unique values)")
            print(f"     → SAUNA soft labels may be too simple")
            print(f"     → Consider: 1) Check SAUNA transform parameters, 2) Verify soft label generation")
        elif unique_values.mean() > 2000:
            print(f"\n  ✅ High soft label richness (avg {unique_values.mean():.0f} unique values)")
            print(f"     → SAUNA soft labels are well-distributed")
    
    # Loss component 분석
    print(f"\n🎯 Loss Configuration:")
    print(f"  - Loss Function: FlowSaunaFMLoss")
    print(f"  - Components: Flow Matching + SAUNA-weighted BCE + SAUNA-weighted Dice")
    print(f"  - Alpha (soft label weight): 2.0")
    print(f"  - Lambda_geo (geometry loss weight): 0.1")
    print(f"  - Time-gating: Enabled (geometry loss increases with t)")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Analyze Flow Model Training')
    parser.add_argument('--log-file', type=str, 
                        default='logs/flow_sauna_medsegdiff_medsegdiff_flow_xca_20260105_134204_train.log',
                        help='Path to training log file')
    parser.add_argument('--output-dir', type=str, default='results/evaluation',
                        help='Directory to save analysis results')
    args = parser.parse_args()
    
    log_path = Path(args.log_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("🚀 FLOW MODEL TRAINING ANALYSIS")
    print("="*80)
    print(f"Log file: {log_path}")
    print(f"Output directory: {output_dir}")
    
    # Parse log
    print("\n📖 Parsing training log...")
    data = parse_training_log(log_path)
    
    print(f"   Found {len(data['epochs'])} validation epochs")
    print(f"   Epoch range: {data['epochs'][0]} - {data['epochs'][-1]}")
    
    # Visualize training progress
    print("\n📊 Generating training analysis plots...")
    plot_path = output_dir / 'flow_sauna_training_analysis.png'
    analyze_training_progress(data, plot_path)
    
    # Performance statistics
    stats = analyze_performance_statistics(data)
    
    # Issue detection
    issues = detect_issues(data)
    
    # Flow-SAUNA specific analysis
    analyze_flow_sauna_specifics(log_path)
    
    # Summary
    print("\n" + "="*80)
    print("📝 SUMMARY")
    print("="*80)
    print(f"Best Epoch: {stats['best_epoch']}")
    print(f"Best Dice: {stats['best_dice']:.4f}")
    print(f"Total Improvement: +{stats['improvement']:.4f}")
    print(f"Issues Detected: {len(issues)}")
    
    if issues:
        high_severity = [i for i in issues if i['severity'] == 'HIGH']
        if high_severity:
            print(f"\n⚠️  {len(high_severity)} HIGH severity issue(s) found!")
            print("   → Review recommendations above")
    
    print("\n✅ Analysis complete!")
    print(f"📁 Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
