"""
Summarize Betti Errors by Network

Calculate mean betti_0_error and betti_1_error for each network on test dataset.
"""

import argparse
from pathlib import Path
import pandas as pd
import json


def summarize_betti_errors(
    lightning_logs_dir: str,
    output_dir: str,
    metrics_filename: str = "sample_metrics.csv"
):
    """
    Calculate mean betti errors for each network.
    
    Args:
        lightning_logs_dir: Directory containing network folders
        output_dir: Directory to save results
        metrics_filename: Name of the metrics file
    """
    lightning_logs_dir = Path(lightning_logs_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all network folders
    network_folders = sorted([d for d in lightning_logs_dir.iterdir() if d.is_dir()])
    
    print(f"Found {len(network_folders)} network folders")
    print(f"Networks: {[f.name for f in network_folders]}\n")
    
    # Collect results
    results = []
    
    for network_folder in network_folders:
        network_name = network_folder.name
        metrics_file = network_folder / "predictions" / metrics_filename
        
        if not metrics_file.exists():
            print(f"⚠️  Skipping {network_name}: {metrics_file.name} not found")
            continue
        
        # Read metrics
        df = pd.read_csv(metrics_file)
        
        # Check if required columns exist
        if "betti_0_error" not in df.columns or "betti_1_error" not in df.columns:
            print(f"⚠️  Skipping {network_name}: Missing betti error columns")
            continue
        
        # Calculate means
        betti_0_mean = df["betti_0_error"].mean()
        betti_1_mean = df["betti_1_error"].mean()
        num_samples = len(df)
        
        results.append({
            "network": network_name,
            "betti_0_error": round(betti_0_mean, 2),
            "betti_1_error": round(betti_1_mean, 2),
            "num_samples": num_samples
        })
        
        print(f"✓ {network_name:15s} | Betti-0: {betti_0_mean:6.2f} | Betti-1: {betti_1_mean:6.2f} | Samples: {num_samples}")
    
    if not results:
        print("\n❌ No results collected!")
        return None
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("betti_0_error")
    
    # Save results
    csv_file = output_dir / "betti_errors_summary.csv"
    results_df.to_csv(csv_file, index=False)
    print(f"\n📊 Saved results to: {csv_file}")
    
    json_file = output_dir / "betti_errors_summary.json"
    results_dict = results_df.to_dict(orient="records")
    with open(json_file, "w") as f:
        json.dump(results_dict, f, indent=2)
    print(f"📊 Saved results to: {json_file}")
    
    # Print summary table
    print("\n" + "=" * 70)
    print("BETTI ERRORS SUMMARY (sorted by betti_0_error)")
    print("=" * 70)
    print(f"{'Network':<15} {'Betti-0 Error':>15} {'Betti-1 Error':>15} {'Samples':>10}")
    print("-" * 70)
    for _, row in results_df.iterrows():
        print(f"{row['network']:<15} {row['betti_0_error']:>15.2f} {row['betti_1_error']:>15.2f} {row['num_samples']:>10}")
    print("=" * 70)
    
    return results_df


def main():
    parser = argparse.ArgumentParser(
        description="Summarize betti errors for each network"
    )
    parser.add_argument(
        "--lightning_logs_dir",
        type=str,
        required=True,
        help="Directory containing network folders (e.g., lightning_logs/octa500_3m)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save results"
    )
    parser.add_argument(
        "--metrics_filename",
        type=str,
        default="sample_metrics.csv",
        help="Name of the metrics file (default: sample_metrics.csv)"
    )
    
    args = parser.parse_args()
    
    # Summarize
    summarize_betti_errors(
        lightning_logs_dir=args.lightning_logs_dir,
        output_dir=args.output_dir,
        metrics_filename=args.metrics_filename
    )


if __name__ == "__main__":
    main()
