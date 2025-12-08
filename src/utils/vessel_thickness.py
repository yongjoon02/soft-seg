"""
Vessel thickness measurement using skeletonization and distance transform.

This module provides functionality to measure mean vessel thickness from binary
segmentation masks by computing the skeleton and using distance transform.
"""

import json
from pathlib import Path
from typing import Dict, Union

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage
from skimage.morphology import medial_axis, skeletonize
from tqdm import tqdm


def compute_vessel_thickness(
    binary_mask: np.ndarray,
    method: str = "medial_axis"
) -> Dict[str, float]:
    """
    Compute vessel thickness metrics from a binary segmentation mask.
    
    Args:
        binary_mask: Binary mask where 1/255 represents vessel pixels and 0 represents background
        method: Skeletonization method ('skeletonize' or 'medial_axis')
    
    Returns:
        Dictionary containing:
            - mean_thickness: Mean vessel thickness (in pixels)
            - median_thickness: Median vessel thickness (in pixels)
            - std_thickness: Standard deviation of vessel thickness
            - max_thickness: Maximum vessel thickness
            - min_thickness: Minimum vessel thickness (non-zero)
            - skeleton_length: Total length of skeleton (in pixels)
            - vessel_area: Total vessel area (in pixels)
    """
    # Ensure binary mask is boolean
    if binary_mask.max() > 1:
        binary_mask = binary_mask > 0

    # Check if mask is empty
    if not binary_mask.any():
        return {
            "mean_thickness": 0.0,
            "median_thickness": 0.0,
            "std_thickness": 0.0,
            "max_thickness": 0.0,
            "min_thickness": 0.0,
            "skeleton_length": 0,
            "vessel_area": 0
        }

    # Compute distance transform
    # This gives the distance from each vessel pixel to the nearest background pixel
    distance_map = ndimage.distance_transform_edt(binary_mask)

    # Compute skeleton
    if method == "medial_axis":
        skeleton, distance_on_skeleton = medial_axis(binary_mask, return_distance=True)
        # Multiply by 2 because distance transform gives radius, we want diameter
        thickness_values = distance_on_skeleton[skeleton] * 2
    else:
        skeleton = skeletonize(binary_mask)
        # Get thickness values at skeleton points
        thickness_values = distance_map[skeleton] * 2

    # Calculate metrics
    skeleton_length = np.sum(skeleton)
    vessel_area = np.sum(binary_mask)

    if len(thickness_values) == 0 or skeleton_length == 0:
        return {
            "mean_thickness": 0.0,
            "median_thickness": 0.0,
            "std_thickness": 0.0,
            "max_thickness": 0.0,
            "min_thickness": 0.0,
            "skeleton_length": 0,
            "vessel_area": vessel_area
        }

    results = {
        "mean_thickness": float(np.mean(thickness_values)),
        "median_thickness": float(np.median(thickness_values)),
        "std_thickness": float(np.std(thickness_values)),
        "max_thickness": float(np.max(thickness_values)),
        "min_thickness": float(np.min(thickness_values[thickness_values > 0])) if np.any(thickness_values > 0) else 0.0,
        "skeleton_length": int(skeleton_length),
        "vessel_area": int(vessel_area)
    }

    return results


def process_dataset(
    label_dir: Union[str, Path],
    output_dir: Union[str, Path],
    method: str = "medial_axis",
    file_pattern: str = "*.bmp",
    save_individual: bool = True,
    save_visualizations: bool = False
) -> pd.DataFrame:
    """
    Process all label images in a directory and compute vessel thickness metrics.
    
    Args:
        label_dir: Directory containing binary label images
        output_dir: Directory to save results
        method: Skeletonization method ('skeletonize' or 'medial_axis')
        file_pattern: File pattern to match (e.g., '*.bmp', '*.png')
        save_individual: Whether to save individual results as JSON
        save_visualizations: Whether to save visualization images
    
    Returns:
        DataFrame with results for all images
    """
    label_dir = Path(label_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all label files
    label_files = sorted(list(label_dir.glob(file_pattern)))

    if len(label_files) == 0:
        print(f"No files found in {label_dir} matching pattern {file_pattern}")
        return pd.DataFrame()

    print(f"Found {len(label_files)} label files")

    # Process each file
    results_list = []

    for label_file in tqdm(label_files, desc="Processing labels"):
        # Read image
        image = cv2.imread(str(label_file), cv2.IMREAD_GRAYSCALE)

        if image is None:
            print(f"Warning: Could not read {label_file}")
            continue

        # Compute thickness metrics
        metrics = compute_vessel_thickness(image, method=method)

        # Add sample_name and filename
        sample_name = label_file.stem
        metrics["sample_name"] = sample_name
        metrics["filename"] = label_file.name
        metrics["file_stem"] = label_file.stem

        results_list.append(metrics)

        # Save individual results
        if save_individual:
            individual_output = output_dir / "individual_results"
            individual_output.mkdir(exist_ok=True)
            with open(individual_output / f"{label_file.stem}.json", "w") as f:
                json.dump(metrics, f, indent=2)

        # Save visualizations
        if save_visualizations:
            vis_output = output_dir / "visualizations"
            vis_output.mkdir(exist_ok=True)
            save_visualization(image, label_file.stem, vis_output, method)

    # Create DataFrame
    df = pd.DataFrame(results_list)

    # Reorder columns to put sample_name first
    if "sample_name" in df.columns:
        cols = ["sample_name"] + [col for col in df.columns if col != "sample_name"]
        df = df[cols]

    # Save summary statistics
    summary_file = output_dir / "summary_statistics.csv"
    df.to_csv(summary_file, index=False)
    print(f"\nSaved summary statistics to {summary_file}")

    # Compute and save aggregate statistics
    aggregate_stats = {
        "dataset_mean_thickness": float(df["mean_thickness"].mean()),
        "dataset_median_thickness": float(df["median_thickness"].mean()),
        "dataset_std_thickness": float(df["std_thickness"].mean()),
        "overall_mean_thickness": float(df["mean_thickness"].mean()),
        "overall_std_thickness": float(df["mean_thickness"].std()),
        "total_images": len(df),
        "total_skeleton_length": int(df["skeleton_length"].sum()),
        "total_vessel_area": int(df["vessel_area"].sum()),
        "method": method
    }

    with open(output_dir / "aggregate_statistics.json", "w") as f:
        json.dump(aggregate_stats, f, indent=2)

    print(f"\n{'='*60}")
    print("Dataset-level Statistics:")
    print(f"{'='*60}")
    print(f"Total images processed: {aggregate_stats['total_images']}")
    print(f"Mean vessel thickness (average across images): {aggregate_stats['dataset_mean_thickness']:.3f} pixels")
    print(f"Overall mean thickness (std): {aggregate_stats['overall_mean_thickness']:.3f} ± {aggregate_stats['overall_std_thickness']:.3f} pixels")
    print(f"Total vessel area: {aggregate_stats['total_vessel_area']:,} pixels")
    print(f"Total skeleton length: {aggregate_stats['total_skeleton_length']:,} pixels")
    print(f"{'='*60}")

    return df


def save_visualization(
    binary_mask: np.ndarray,
    filename_stem: str,
    output_dir: Path,
    method: str = "medial_axis"
) -> None:
    """
    Save visualization of skeleton overlaid on original mask.
    
    Args:
        binary_mask: Binary segmentation mask
        filename_stem: Name stem for output file
        output_dir: Directory to save visualization
        method: Skeletonization method used
    """
    # Ensure binary
    if binary_mask.max() > 1:
        binary_mask = binary_mask > 0

    # Compute skeleton and distance map
    distance_map = ndimage.distance_transform_edt(binary_mask)

    if method == "medial_axis":
        skeleton, _ = medial_axis(binary_mask, return_distance=True)
    else:
        skeleton = skeletonize(binary_mask)

    # Create RGB visualization
    vis = np.zeros((binary_mask.shape[0], binary_mask.shape[1], 3), dtype=np.uint8)

    # Original mask in gray
    vis[binary_mask] = [128, 128, 128]

    # Skeleton in red
    vis[skeleton] = [0, 0, 255]

    # Save
    output_path = output_dir / f"{filename_stem}_skeleton.png"
    cv2.imwrite(str(output_path), vis)

    # Also save distance map as heatmap
    if binary_mask.any():
        distance_normalized = (distance_map * 255 / distance_map.max()).astype(np.uint8)
        distance_colored = cv2.applyColorMap(distance_normalized, cv2.COLORMAP_JET)
        distance_colored[~binary_mask] = 0

        heatmap_path = output_dir / f"{filename_stem}_distance.png"
        cv2.imwrite(str(heatmap_path), distance_colored)


def main():
    """Main function for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Measure vessel thickness from binary segmentation masks"
    )
    parser.add_argument(
        "--label_dir",
        type=str,
        required=True,
        help="Directory containing binary label images"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save results"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="medial_axis",
        choices=["medial_axis", "skeletonize"],
        help="Skeletonization method"
    )
    parser.add_argument(
        "--file_pattern",
        type=str,
        default="*.bmp",
        help="File pattern to match (e.g., '*.bmp', '*.png')"
    )
    parser.add_argument(
        "--save_individual",
        action="store_true",
        help="Save individual results as JSON"
    )
    parser.add_argument(
        "--save_visualizations",
        action="store_true",
        help="Save visualization images"
    )

    args = parser.parse_args()

    # Process dataset
    process_dataset(
        label_dir=args.label_dir,
        output_dir=args.output_dir,
        method=args.method,
        file_pattern=args.file_pattern,
        save_individual=args.save_individual,
        save_visualizations=args.save_visualizations
    )

    print(f"\nProcessing complete! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
