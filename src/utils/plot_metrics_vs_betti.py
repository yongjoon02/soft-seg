"""
Create scatter plots: Dice/clDice vs Betti Error
Shows trade-off between segmentation accuracy and topological accuracy
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def create_scatter_plots(
    betti_csv: str,
    output_dir: str,
    dataset_name: str = "OCTA500-3M"
):
    """
    Create scatter plots of Dice/clDice vs Betti errors.
    
    Args:
        betti_csv: Path to betti errors summary CSV
        output_dir: Directory to save figures
        dataset_name: Name of dataset for title
    """
    # Manual scores from evaluation results
    scores = {
        'cenet': {'dice': 0.8892, 'cldice': 0.9256},
        'csnet': {'dice': 0.8885, 'cldice': 0.9268},
        'aacaunet': {'dice': 0.8934, 'cldice': 0.9298},
        'vesselnet': {'dice': 0.8970, 'cldice': 0.9314},
        'transunet': {'dice': 0.8910, 'cldice': 0.9287},
        'dscnet': {'dice': 0.8968, 'cldice': 0.9309},
        'berdiff': {'dice': 0.8884, 'cldice': 0.9286},
        'medsegdiff': {'dice': 0.8861, 'cldice': 0.9286},
        'segdiff': {'dice': 0.8816, 'cldice': 0.9298},
        'colddiff': {'dice': 0.8785, 'cldice': 0.9243},
        'maskdiff': {'dice': 0.8906, 'cldice': 0.9349}
    }
    
    # Read betti errors
    betti_df = pd.read_csv(betti_csv)
    
    # Merge data
    data = []
    for _, row in betti_df.iterrows():
        network = row['network']
        if network in scores:
            data.append({
                'network': network,
                'dice': scores[network]['dice'],
                'cldice': scores[network]['cldice'],
                'betti_0_error': row['betti_0_error_mean'],
                'betti_1_error': row['betti_1_error_mean']
            })
    
    df = pd.DataFrame(data)
    
    # Categorize networks
    diffusion_models = ['berdiff', 'medsegdiff', 'segdiff', 'colddiff', 'maskdiff']
    supervised_models = ['cenet', 'csnet', 'aacaunet', 'vesselnet', 'transunet', 'dscnet']
    
    df['model_type'] = df['network'].apply(
        lambda x: 'Diffusion' if x in diffusion_models else 'Supervised'
    )
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define colors and markers
    colors = {'Diffusion': '#2E86AB', 'Supervised': '#A23B72'}
    markers = {'Diffusion': 'o', 'Supervised': 's'}
    
    # ========================
    # Figure 1: Dice vs Betti
    # ========================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Dice vs Betti-0 Error
    ax1 = axes[0]
    for model_type in ['Supervised', 'Diffusion']:
        mask = df['model_type'] == model_type
        ax1.scatter(
            df[mask]['dice'],
            df[mask]['betti_0_error'],
            c=colors[model_type],
            marker=markers[model_type],
            s=120,
            alpha=0.7,
            edgecolors='black',
            linewidths=1.5,
            label=model_type
        )
        
        # Add network labels
        for _, row in df[mask].iterrows():
            ax1.annotate(
                row['network'],
                (row['dice'], row['betti_0_error']),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8,
                alpha=0.8
            )
    
    ax1.set_xlabel('Dice Score', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Betti-0 Error', fontsize=12, fontweight='bold')
    ax1.set_title('Dice vs Betti-0 Error', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10, loc='upper right')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Plot 2: Dice vs Betti-1 Error
    ax2 = axes[1]
    for model_type in ['Supervised', 'Diffusion']:
        mask = df['model_type'] == model_type
        ax2.scatter(
            df[mask]['dice'],
            df[mask]['betti_1_error'],
            c=colors[model_type],
            marker=markers[model_type],
            s=120,
            alpha=0.7,
            edgecolors='black',
            linewidths=1.5,
            label=model_type
        )
        
        # Add network labels
        for _, row in df[mask].iterrows():
            ax2.annotate(
                row['network'],
                (row['dice'], row['betti_1_error']),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8,
                alpha=0.8
            )
    
    ax2.set_xlabel('Dice Score', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Betti-1 Error', fontsize=12, fontweight='bold')
    ax2.set_title('Dice vs Betti-1 Error', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10, loc='upper right')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.suptitle(f'Trade-off: Dice Score vs Topological Accuracy\n({dataset_name})',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save Dice figure
    output_file = output_dir / f'dice_vs_betti_{dataset_name.lower().replace("-", "_")}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved Dice figure to: {output_file}")
    
    output_file_pdf = output_dir / f'dice_vs_betti_{dataset_name.lower().replace("-", "_")}.pdf'
    plt.savefig(output_file_pdf, bbox_inches='tight')
    print(f"✅ Saved Dice figure to: {output_file_pdf}")
    
    plt.close()
    
    # ========================
    # Figure 2: clDice vs Betti
    # ========================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: clDice vs Betti-0 Error
    ax1 = axes[0]
    for model_type in ['Supervised', 'Diffusion']:
        mask = df['model_type'] == model_type
        ax1.scatter(
            df[mask]['cldice'],
            df[mask]['betti_0_error'],
            c=colors[model_type],
            marker=markers[model_type],
            s=120,
            alpha=0.7,
            edgecolors='black',
            linewidths=1.5,
            label=model_type
        )
        
        # Add network labels
        for _, row in df[mask].iterrows():
            ax1.annotate(
                row['network'],
                (row['cldice'], row['betti_0_error']),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8,
                alpha=0.8
            )
    
    ax1.set_xlabel('clDice Score', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Betti-0 Error', fontsize=12, fontweight='bold')
    ax1.set_title('clDice vs Betti-0 Error', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10, loc='upper right')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Plot 2: clDice vs Betti-1 Error
    ax2 = axes[1]
    for model_type in ['Supervised', 'Diffusion']:
        mask = df['model_type'] == model_type
        ax2.scatter(
            df[mask]['cldice'],
            df[mask]['betti_1_error'],
            c=colors[model_type],
            marker=markers[model_type],
            s=120,
            alpha=0.7,
            edgecolors='black',
            linewidths=1.5,
            label=model_type
        )
        
        # Add network labels
        for _, row in df[mask].iterrows():
            ax2.annotate(
                row['network'],
                (row['cldice'], row['betti_1_error']),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8,
                alpha=0.8
            )
    
    ax2.set_xlabel('clDice Score', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Betti-1 Error', fontsize=12, fontweight='bold')
    ax2.set_title('clDice vs Betti-1 Error', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10, loc='upper right')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.suptitle(f'Trade-off: clDice Score vs Topological Accuracy\n({dataset_name})',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save clDice figure
    output_file = output_dir / f'cldice_vs_betti_{dataset_name.lower().replace("-", "_")}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved clDice figure to: {output_file}")
    
    output_file_pdf = output_dir / f'cldice_vs_betti_{dataset_name.lower().replace("-", "_")}.pdf'
    plt.savefig(output_file_pdf, bbox_inches='tight')
    print(f"✅ Saved clDice figure to: {output_file_pdf}")
    
    plt.close()
    
    # Print statistics
    print("\n" + "="*90)
    print("STATISTICS")
    print("="*90)
    print(f"{'Model Type':<15} {'Dice':>10} {'clDice':>10} {'Betti-0 Error':>15} {'Betti-1 Error':>15}")
    print("-"*90)
    for model_type in ['Diffusion', 'Supervised']:
        mask = df['model_type'] == model_type
        print(f"{model_type:<15} "
              f"{df[mask]['dice'].mean():>10.4f} "
              f"{df[mask]['cldice'].mean():>10.4f} "
              f"{df[mask]['betti_0_error'].mean():>15.2f} "
              f"{df[mask]['betti_1_error'].mean():>15.2f}")
    print("="*90)


def main():
    parser = argparse.ArgumentParser(
        description="Create Dice/clDice vs Betti Error scatter plots"
    )
    parser.add_argument(
        "--betti_csv",
        type=str,
        required=True,
        help="Path to betti errors summary CSV"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save figures"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="OCTA500-3M",
        help="Name of dataset for title"
    )
    
    args = parser.parse_args()
    
    create_scatter_plots(
        betti_csv=args.betti_csv,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name
    )


if __name__ == "__main__":
    main()
