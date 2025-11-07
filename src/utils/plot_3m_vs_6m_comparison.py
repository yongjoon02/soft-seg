"""
Create comparison plots: 3M vs 6M datasets
Compare Dice/clDice vs Betti Error across different scan areas
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def create_3m_vs_6m_comparison(
    betti_3m_csv: str,
    betti_6m_csv: str,
    output_dir: str
):
    """
    Create comparison plots between 3M and 6M datasets.
    
    Args:
        betti_3m_csv: Path to 3M betti errors summary CSV
        betti_6m_csv: Path to 6M betti errors summary CSV
        output_dir: Directory to save figures
    """
    # 3M scores
    scores_3m = {
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
    
    # 6M scores
    scores_6m = {
        'cenet': {'dice': 0.8544, 'cldice': 0.8967},
        'csnet': {'dice': 0.8550, 'cldice': 0.9010},
        'aacaunet': {'dice': 0.8502, 'cldice': 0.9014},
        'vesselnet': {'dice': 0.8493, 'cldice': 0.8958},
        'transunet': {'dice': 0.8677, 'cldice': 0.9102},
        'dscnet': {'dice': 0.8411, 'cldice': 0.8868},
        'berdiff': {'dice': 0.8600, 'cldice': 0.8956},
        'medsegdiff': {'dice': 0.8616, 'cldice': 0.8955},
        'segdiff': {'dice': 0.8643, 'cldice': 0.8986},
        'colddiff': {'dice': 0.8819, 'cldice': 0.9092},
        'maskdiff': {'dice': 0.8698, 'cldice': 0.8951}
    }
    
    # Read betti errors
    betti_3m_df = pd.read_csv(betti_3m_csv)
    betti_6m_df = pd.read_csv(betti_6m_csv)
    
    # Merge data for 3M
    data_3m = []
    for _, row in betti_3m_df.iterrows():
        network = row['network']
        if network in scores_3m:
            data_3m.append({
                'network': network,
                'dataset': '3M',
                'dice': scores_3m[network]['dice'],
                'cldice': scores_3m[network]['cldice'],
                'betti_0_error': row['betti_0_error_mean'],
                'betti_1_error': row['betti_1_error_mean']
            })
    
    # Merge data for 6M
    data_6m = []
    for _, row in betti_6m_df.iterrows():
        network = row['network']
        if network in scores_6m:
            data_6m.append({
                'network': network,
                'dataset': '6M',
                'dice': scores_6m[network]['dice'],
                'cldice': scores_6m[network]['cldice'],
                'betti_0_error': row['betti_0_error_mean'],
                'betti_1_error': row['betti_1_error_mean']
            })
    
    df_3m = pd.DataFrame(data_3m)
    df_6m = pd.DataFrame(data_6m)
    df_combined = pd.concat([df_3m, df_6m], ignore_index=True)
    
    # Categorize networks
    diffusion_models = ['berdiff', 'medsegdiff', 'segdiff', 'colddiff', 'maskdiff']
    
    df_combined['model_type'] = df_combined['network'].apply(
        lambda x: 'Diffusion' if x in diffusion_models else 'Supervised'
    )
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define colors and markers
    colors_dataset = {'3M': '#E63946', '6M': '#457B9D'}
    markers_type = {'Diffusion': 'o', 'Supervised': 's'}
    
    # ========================
    # Figure 1: Dice vs Betti-0 (3M vs 6M)
    # ========================
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Dice vs Betti-0 Error
    ax1 = axes[0]
    for dataset in ['3M', '6M']:
        for model_type in ['Supervised', 'Diffusion']:
            mask = (df_combined['dataset'] == dataset) & (df_combined['model_type'] == model_type)
            ax1.scatter(
                df_combined[mask]['dice'],
                df_combined[mask]['betti_0_error'],
                c=colors_dataset[dataset],
                marker=markers_type[model_type],
                s=150,
                alpha=0.7,
                edgecolors='black',
                linewidths=1.5,
                label=f'{dataset} - {model_type}'
            )
    
    ax1.set_xlabel('Dice Score', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Betti-0 Error', fontsize=13, fontweight='bold')
    ax1.set_title('Dice vs Betti-0 Error (3M vs 6M)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10, loc='upper right', framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Plot 2: Dice vs Betti-1 Error
    ax2 = axes[1]
    for dataset in ['3M', '6M']:
        for model_type in ['Supervised', 'Diffusion']:
            mask = (df_combined['dataset'] == dataset) & (df_combined['model_type'] == model_type)
            ax2.scatter(
                df_combined[mask]['dice'],
                df_combined[mask]['betti_1_error'],
                c=colors_dataset[dataset],
                marker=markers_type[model_type],
                s=150,
                alpha=0.7,
                edgecolors='black',
                linewidths=1.5,
                label=f'{dataset} - {model_type}'
            )
    
    ax2.set_xlabel('Dice Score', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Betti-1 Error', fontsize=13, fontweight='bold')
    ax2.set_title('Dice vs Betti-1 Error (3M vs 6M)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10, loc='upper right', framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.suptitle('Dataset Comparison: OCTA500-3M vs OCTA500-6M',
                 fontsize=15, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    # Save Dice figure
    output_file = output_dir / 'dice_vs_betti_3m_vs_6m.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved Dice comparison to: {output_file}")
    
    plt.close()
    
    # ========================
    # Figure 2: clDice vs Betti (3M vs 6M)
    # ========================
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: clDice vs Betti-0 Error
    ax1 = axes[0]
    for dataset in ['3M', '6M']:
        for model_type in ['Supervised', 'Diffusion']:
            mask = (df_combined['dataset'] == dataset) & (df_combined['model_type'] == model_type)
            ax1.scatter(
                df_combined[mask]['cldice'],
                df_combined[mask]['betti_0_error'],
                c=colors_dataset[dataset],
                marker=markers_type[model_type],
                s=150,
                alpha=0.7,
                edgecolors='black',
                linewidths=1.5,
                label=f'{dataset} - {model_type}'
            )
    
    ax1.set_xlabel('clDice Score', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Betti-0 Error', fontsize=13, fontweight='bold')
    ax1.set_title('clDice vs Betti-0 Error (3M vs 6M)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10, loc='upper right', framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Plot 2: clDice vs Betti-1 Error
    ax2 = axes[1]
    for dataset in ['3M', '6M']:
        for model_type in ['Supervised', 'Diffusion']:
            mask = (df_combined['dataset'] == dataset) & (df_combined['model_type'] == model_type)
            ax2.scatter(
                df_combined[mask]['cldice'],
                df_combined[mask]['betti_1_error'],
                c=colors_dataset[dataset],
                marker=markers_type[model_type],
                s=150,
                alpha=0.7,
                edgecolors='black',
                linewidths=1.5,
                label=f'{dataset} - {model_type}'
            )
    
    ax2.set_xlabel('clDice Score', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Betti-1 Error', fontsize=13, fontweight='bold')
    ax2.set_title('clDice vs Betti-1 Error (3M vs 6M)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10, loc='upper right', framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.suptitle('Dataset Comparison: OCTA500-3M vs OCTA500-6M',
                 fontsize=15, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    # Save clDice figure
    output_file = output_dir / 'cldice_vs_betti_3m_vs_6m.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved clDice comparison to: {output_file}")
    
    plt.close()
    
    # Print statistics
    print("\n" + "="*100)
    print("STATISTICS: 3M vs 6M")
    print("="*100)
    print(f"{'Dataset':<10} {'Model Type':<15} {'Dice':>10} {'clDice':>10} {'Betti-0':>12} {'Betti-1':>12}")
    print("-"*100)
    for dataset in ['3M', '6M']:
        for model_type in ['Diffusion', 'Supervised']:
            mask = (df_combined['dataset'] == dataset) & (df_combined['model_type'] == model_type)
            if mask.sum() > 0:
                print(f"{dataset:<10} {model_type:<15} "
                      f"{df_combined[mask]['dice'].mean():>10.4f} "
                      f"{df_combined[mask]['cldice'].mean():>10.4f} "
                      f"{df_combined[mask]['betti_0_error'].mean():>12.2f} "
                      f"{df_combined[mask]['betti_1_error'].mean():>12.2f}")
    print("="*100)


def main():
    parser = argparse.ArgumentParser(
        description="Create 3M vs 6M comparison plots"
    )
    parser.add_argument(
        "--betti_3m_csv",
        type=str,
        required=True,
        help="Path to 3M betti errors summary CSV"
    )
    parser.add_argument(
        "--betti_6m_csv",
        type=str,
        required=True,
        help="Path to 6M betti errors summary CSV"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save figures"
    )
    
    args = parser.parse_args()
    
    create_3m_vs_6m_comparison(
        betti_3m_csv=args.betti_3m_csv,
        betti_6m_csv=args.betti_6m_csv,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
