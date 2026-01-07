"""
Extract gradient conflict data from TensorBoard logs and create a detailed report.
"""
import numpy as np
from pathlib import Path
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt

def load_tensorboard_data(log_dir):
    """Load gradient conflict data from TensorBoard logs"""
    log_path = Path(log_dir)
    event_file = list(log_path.glob('events.out.tfevents.*'))[0]
    
    ea = event_accumulator.EventAccumulator(str(event_file))
    ea.Reload()
    
    # Get all scalar tags
    tags = ea.Tags()['scalars']
    
    data = {}
    for tag in tags:
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        data[tag] = {'steps': np.array(steps), 'values': np.array(values)}
    
    return data


def analyze_gradient_conflicts(data):
    """Analyze gradient conflicts from extracted data"""
    print("\n" + "="*80)
    print("🔍 GRADIENT CONFLICT ANALYSIS")
    print("="*80)
    
    # Extract metrics
    metrics = {
        'cosine_similarities': {},
        'gradient_magnitudes': {},
        'total_loss': None
    }
    
    for tag, values in data.items():
        if 'grad_cos' in tag.lower():
            # Extract component pair name (e.g., "flow_vs_bce")
            pair_name = tag.split('/')[-1]
            metrics['cosine_similarities'][pair_name] = values
        elif 'grad_norm' in tag.lower():
            # Extract component name
            component = tag.split('/')[-1]
            metrics['gradient_magnitudes'][component] = values
        elif 'loss/total' in tag.lower():
            metrics['total_loss'] = values
    
    # Analysis 1: Cosine Similarities between different loss pairs
    if metrics['cosine_similarities']:
        print(f"\n📊 Pairwise Cosine Similarities:")
        
        for pair, values_dict in metrics['cosine_similarities'].items():
            cos_sim = values_dict['values']
            print(f"\n  {pair}:")
            print(f"    Mean: {cos_sim.mean():+.4f}")
            print(f"    Std:  {cos_sim.std():.4f}")
            print(f"    Min:  {cos_sim.min():+.4f}")
            print(f"    Max:  {cos_sim.max():+.4f}")
            
            # Count conflicts (negative cosine similarity)
            negative_count = (cos_sim < 0).sum()
            negative_ratio = negative_count / len(cos_sim)
            print(f"    Conflicts: {negative_count}/{len(cos_sim)} ({negative_ratio*100:.1f}%)")
            
            if cos_sim.mean() < -0.3:
                print(f"    🔴 SEVERE CONFLICT: Gradients strongly oppose each other!")
            elif cos_sim.mean() < 0:
                print(f"    🟡 CONFLICT: Gradients work against each other")
            elif cos_sim.mean() < 0.3:
                print(f"    🟡 WEAK ALIGNMENT: Gradients are somewhat independent")
            else:
                print(f"    🟢 ALIGNED: Gradients work together")
    
    # Analysis 2: Gradient Magnitudes & Balance
    if metrics['gradient_magnitudes']:
        print(f"\n📏 Gradient Magnitudes:")
        
        grad_means = {}
        for component, values_dict in metrics['gradient_magnitudes'].items():
            values = values_dict['values']
            grad_means[component] = values.mean()
            print(f"\n  {component}:")
            print(f"    Mean: {values.mean():.6f}")
            print(f"    Std:  {values.std():.6f}")
            print(f"    Max:  {values.max():.6f}")
        
        # Check for magnitude imbalance
        if len(grad_means) > 1:
            print(f"\n  📊 Gradient Magnitude Ratios:")
            components = list(grad_means.keys())
            for i, comp1 in enumerate(components):
                for comp2 in components[i+1:]:
                    ratio = grad_means[comp1] / grad_means[comp2]
                    print(f"    {comp1}/{comp2}: {ratio:.2f}x")
                    
                    if ratio > 10 or ratio < 0.1:
                        print(f"      🔴 SEVERE IMBALANCE: Consider adjusting loss weights!")
                    elif ratio > 5 or ratio < 0.2:
                        print(f"      🟡 IMBALANCE: May affect training dynamics")
    
    return metrics


def plot_gradient_analysis(data, save_path):
    """Create visualization of gradient conflicts"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Gradient Conflict Analysis', fontsize=16, fontweight='bold')
    
    # Extract metrics
    cosine_sim = None
    conflict_ratio = None
    grad_norms = {}
    
    for tag, values in data.items():
        if 'cosine_similarity' in tag.lower():
            cosine_sim = values
        elif 'conflict' in tag.lower() and 'ratio' in tag.lower():
            conflict_ratio = values
        elif 'grad_norm' in tag.lower():
            component = tag.split('/')[-1]
            grad_norms[component] = values
    
    # Plot 1: Cosine Similarity
    if cosine_sim:
        ax = axes[0, 0]
        ax.plot(cosine_sim['steps'], cosine_sim['values'], 'b-', linewidth=2, alpha=0.7)
        ax.axhline(y=0, color='r', linestyle='--', linewidth=1.5, label='Zero Line')
        ax.set_xlabel('Batch')
        ax.set_ylabel('Cosine Similarity')
        ax.set_title('Cosine Similarity between Gradient Components')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # Plot 2: Conflict Ratio
    if conflict_ratio:
        ax = axes[0, 1]
        ax.plot(conflict_ratio['steps'], conflict_ratio['values']*100, 'r-', 
                linewidth=2, alpha=0.7)
        ax.axhline(y=50, color='orange', linestyle='--', linewidth=1.5, label='50% Line')
        ax.set_xlabel('Batch')
        ax.set_ylabel('Conflict Ratio (%)')
        ax.set_title('Gradient Conflict Ratio Over Time')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # Plot 3: Gradient Norms
    if grad_norms:
        ax = axes[1, 0]
        colors = ['blue', 'green', 'orange', 'purple']
        for i, (component, values_dict) in enumerate(grad_norms.items()):
            color = colors[i % len(colors)]
            ax.plot(values_dict['steps'], values_dict['values'], 
                   color=color, linewidth=2, alpha=0.7, label=component)
        ax.set_xlabel('Batch')
        ax.set_ylabel('Gradient Norm')
        ax.set_title('Gradient Magnitudes by Component')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # Plot 4: Distribution
    if cosine_sim:
        ax = axes[1, 1]
        values = cosine_sim['values']
        ax.hist(values, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
        ax.axvline(x=values.mean(), color='r', linestyle='--', 
                  linewidth=2, label=f'Mean: {values.mean():.3f}')
        ax.axvline(x=0, color='orange', linestyle='--', 
                  linewidth=2, label='Zero')
        ax.set_xlabel('Cosine Similarity')
        ax.set_ylabel('Frequency')
        ax.set_title('Cosine Similarity Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Gradient analysis plot saved to: {save_path}")
    plt.close()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--grad-dir', type=str, required=True, help='Gradient analysis directory')
    parser.add_argument('--output-dir', type=str, default='results/evaluation', help='Output directory')
    args = parser.parse_args()
    
    log_dir = Path(args.grad_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("🧮 GRADIENT CONFLICT ANALYSIS FROM TENSORBOARD")
    print("="*80)
    print(f"Log directory: {log_dir}")
    
    # Load data
    print("\n📥 Loading TensorBoard data...")
    data = load_tensorboard_data(log_dir)
    print(f"   Found {len(data)} scalar metrics")
    
    # Analyze
    metrics = analyze_gradient_conflicts(data)
    
    # Plot
    print("\n📊 Creating visualizations...")
    plot_path = output_dir / 'gradient_conflict_analysis.png'
    plot_gradient_analysis(data, plot_path)
    
    # Recommendations
    print("\n" + "="*80)
    print("💡 RECOMMENDATIONS")
    print("="*80)
    
    if metrics.get('cosine_similarity'):
        cos_sim_mean = metrics['cosine_similarity']['values'].mean()
        
        if cos_sim_mean < -0.1:
            print("\n🔴 CRITICAL GRADIENT CONFLICT DETECTED!")
            print("\nImmediate Actions:")
            print("  1. Reduce lambda_geo from 0.1 to 0.01 or 0.001")
            print("  2. Consider using gradient projection (PCGrad)")
            print("  3. Try alternative loss balancing (e.g., uncertainty weighting)")
            print("  4. Verify loss implementation - check sign conventions")
            
        elif cos_sim_mean < 0.1:
            print("\n🟡 MODERATE GRADIENT CONFLICT")
            print("\nSuggested Improvements:")
            print("  1. Fine-tune loss weights (current lambda_geo: 0.1)")
            print("  2. Consider adaptive loss weighting")
            print("  3. Monitor training more closely for instability")
            
        else:
            print("\n🟢 GRADIENTS ARE REASONABLY ALIGNED")
            print("\nCurrent Configuration:")
            print("  ✓ Loss weighting appears appropriate")
            print("  ✓ Flow and geometry objectives are compatible")
            print("  → Continue with current settings or minor adjustments")
    
    print("\n✅ Analysis complete!")


if __name__ == '__main__':
    main()
