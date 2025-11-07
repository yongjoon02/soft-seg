"""
Collect Betti Errors from Multiple Networks

This script collects betti_0_error and betti_1_error from sample_metrics.csv files
across different network architectures.
"""

import argparse
from pathlib import Path
import pandas as pd
import json


def collect_betti_errors(
    lightning_logs_dir: str,
    output_dir: str,
    metrics_filename: str = "sample_metrics.csv",
    network_order: list = None
) -> pd.DataFrame:
    """
    Collect betti errors from all network predictions.
    
    Args:
        lightning_logs_dir: Directory containing network folders
        output_dir: Directory to save results
        metrics_filename: Name of the metrics file (default: sample_metrics.csv)
        network_order: Desired order of networks in output (optional)
    
    Returns:
        DataFrame with collected metrics
    """
    lightning_logs_dir = Path(lightning_logs_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all network folders
    network_folders = sorted([d for d in lightning_logs_dir.iterdir() if d.is_dir()])
    
    print(f"Found {len(network_folders)} network folders")
    print("Networks:", [f.name for f in network_folders])
    
    # Collect metrics from each network
    all_results = []
    network_summaries = {}
    
    for network_folder in network_folders:
        network_name = network_folder.name
        metrics_file = network_folder / "predictions" / metrics_filename
        
        if not metrics_file.exists():
            print(f"⚠️  Skipping {network_name}: {metrics_filename} not found")
            continue
        
        print(f"\n✓ Processing {network_name}...")
        
        # Read metrics
        df = pd.read_csv(metrics_file)
        
        # Check if required columns exist
        if "betti_0_error" not in df.columns or "betti_1_error" not in df.columns:
            print(f"  ⚠️  Missing betti error columns in {network_name}")
            continue
        
        # Select only needed columns
        df_selected = df[["sample_name", "betti_0_error", "betti_1_error"]].copy()
        df_selected["network"] = network_name
        
        all_results.append(df_selected)
        
        # Compute summary statistics
        network_summaries[network_name] = {
            "network": network_name,
            "betti_0_error_mean": float(df["betti_0_error"].mean()),
            "betti_0_error_std": float(df["betti_0_error"].std()),
            "betti_1_error_mean": float(df["betti_1_error"].mean()),
            "betti_1_error_std": float(df["betti_1_error"].std()),
            "num_samples": len(df)
        }
        
        print(f"  Betti-0: {network_summaries[network_name]['betti_0_error_mean']:.2f}, "
              f"Betti-1: {network_summaries[network_name]['betti_1_error_mean']:.2f}, "
              f"Samples: {len(df)}")
    
    if not all_results:
        print("\n❌ No results collected!")
        return pd.DataFrame()
    
    # Combine all results
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # Save detailed results
    detailed_file = output_dir / "betti_errors_detailed.csv"
    combined_df.to_csv(detailed_file, index=False)
    print(f"\n📊 Saved detailed results to {detailed_file}")
    
    # Create pivot tables with all samples (NaN for missing data)
    pivot_b0_df = combined_df.pivot_table(
        index="sample_name", 
        columns="network", 
        values="betti_0_error",
        aggfunc='first'  # Use first value if duplicates
    )
    
    pivot_b1_df = combined_df.pivot_table(
        index="sample_name",
        columns="network",
        values="betti_1_error",
        aggfunc='first'
    )
    
    # Reorder columns if network_order is provided
    if network_order:
        available_networks = [n for n in network_order if n in pivot_b0_df.columns]
        pivot_b0_df = pivot_b0_df[available_networks]
        pivot_b1_df = pivot_b1_df[available_networks]
    
    # Save pivot tables
    pivot_b0_file = output_dir / "betti_0_error_pivot.csv"
    pivot_b0_df.to_csv(pivot_b0_file)
    print(f"📊 Saved betti_0_error pivot table to {pivot_b0_file}")
    
    pivot_b1_file = output_dir / "betti_1_error_pivot.csv"
    pivot_b1_df.to_csv(pivot_b1_file)
    print(f"📊 Saved betti_1_error pivot table to {pivot_b1_file}")
    
    # Save network summaries
    summary_df = pd.DataFrame(network_summaries).T
    summary_df = summary_df.sort_values("betti_0_error_mean")
    
    summary_file = output_dir / "betti_errors_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"📊 Saved network summary to {summary_file}")
    
    json_file = output_dir / "betti_errors_summary.json"
    with open(json_file, "w") as f:
        json.dump(network_summaries, f, indent=2)
    print(f"📊 Saved network summary (JSON) to {json_file}")
    
    # Print summary table
    print("\n" + "=" * 80)
    print("BETTI ERRORS SUMMARY BY NETWORK")
    print("=" * 80)
    print(f"{'Network':<15} {'Betti-0 Mean':>13} {'Betti-0 Std':>13} {'Betti-1 Mean':>13} {'Betti-1 Std':>13} {'Samples':>10}")
    print("-" * 80)
    for _, row in summary_df.iterrows():
        print(f"{row['network']:<15} "
              f"{row['betti_0_error_mean']:>13.2f} "
              f"{row['betti_0_error_std']:>13.2f} "
              f"{row['betti_1_error_mean']:>13.2f} "
              f"{row['betti_1_error_std']:>13.2f} "
              f"{int(row['num_samples']):>10}")
    print("=" * 80)
    
    return combined_df


def main():
    parser = argparse.ArgumentParser(
        description="Collect betti errors from multiple networks"
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
        help="Directory to save collected results"
    )
    parser.add_argument(
        "--metrics_filename",
        type=str,
        default="sample_metrics.csv",
        help="Name of the metrics file (default: sample_metrics.csv)"
    )
    parser.add_argument(
        "--network_order",
        type=str,
        nargs='+',
        help="Desired order of networks (space-separated)"
    )
    
    args = parser.parse_args()
    
    # Collect metrics
    df = collect_betti_errors(
        lightning_logs_dir=args.lightning_logs_dir,
        output_dir=args.output_dir,
        metrics_filename=args.metrics_filename,
        network_order=args.network_order
    )
    
    print(f"\n✅ Collection complete! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
