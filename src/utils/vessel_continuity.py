"""
Vessel Continuity Analysis

This module provides functions to analyze vessel continuity from binary segmentation masks.
Metrics include:
- Connected components count
- Betti numbers (topological features)
- Average branch length
- Tortuosity
- Gap analysis
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage
from skimage import morphology, measure
from skimage.morphology import skeletonize
from tqdm import tqdm


def compute_betti_numbers(binary_image: np.ndarray) -> Tuple[int, int]:
    """
    Compute Betti numbers for topological analysis.
    
    Betti-0: Number of connected components
    Betti-1: Number of loops/holes
    
    Args:
        binary_image: Binary image (0 or 255)
    
    Returns:
        Tuple of (betti_0, betti_1)
    """
    # Ensure binary
    binary = (binary_image > 127).astype(np.uint8)
    
    # Betti-0: Number of connected components
    labeled_image = measure.label(binary, connectivity=2)
    betti_0 = labeled_image.max()
    
    # Betti-1: Euler characteristic formula
    # χ = V - E + F, where χ = betti_0 - betti_1
    # For 2D: betti_1 = betti_0 - χ
    
    # Use skeletonized version for better loop detection
    skeleton = skeletonize(binary)
    
    # Count endpoints and junctions
    # Endpoints have 1 neighbor, junctions have 3+ neighbors
    kernel = np.ones((3, 3), dtype=np.uint8)
    neighbors = cv2.filter2D(skeleton.astype(np.uint8), -1, kernel) * skeleton
    
    endpoints = np.sum(neighbors == 2)  # Including center pixel
    junctions = np.sum(neighbors >= 4)
    
    # Estimate betti_1 using Euler characteristic
    # For a graph: χ = V - E, where V = vertices, E = edges
    # betti_1 = E - V + 1 (for connected graph)
    
    # Simple approximation: loops = junctions - endpoints / 2
    betti_1 = max(0, junctions - endpoints // 2)
    
    return int(betti_0), int(betti_1)


def compute_connected_components(binary_image: np.ndarray) -> Dict:
    """
    Analyze connected components in the binary image.
    
    Args:
        binary_image: Binary image (0 or 255)
    
    Returns:
        Dictionary with component statistics
    """
    binary = (binary_image > 127).astype(np.uint8)
    
    # Label connected components
    labeled_image, num_components = measure.label(binary, connectivity=2, return_num=True)
    
    if num_components == 0:
        return {
            "num_components": 0,
            "largest_component_ratio": 0.0,
            "mean_component_size": 0.0,
            "component_size_std": 0.0
        }
    
    # Get component sizes
    component_sizes = []
    for i in range(1, num_components + 1):
        size = np.sum(labeled_image == i)
        component_sizes.append(size)
    
    component_sizes = np.array(component_sizes)
    total_vessel_pixels = np.sum(binary > 0)
    
    return {
        "num_components": num_components,
        "largest_component_ratio": float(component_sizes.max() / total_vessel_pixels) if total_vessel_pixels > 0 else 0.0,
        "mean_component_size": float(component_sizes.mean()),
        "component_size_std": float(component_sizes.std())
    }


def compute_branch_statistics(binary_image: np.ndarray) -> Dict:
    """
    Compute statistics about vessel branches using skeleton analysis.
    
    Args:
        binary_image: Binary image (0 or 255)
    
    Returns:
        Dictionary with branch statistics
    """
    binary = (binary_image > 127).astype(np.uint8)
    
    # Skeletonize
    skeleton = skeletonize(binary)
    
    if np.sum(skeleton) == 0:
        return {
            "num_endpoints": 0,
            "num_junctions": 0,
            "average_branch_length": 0.0,
            "total_skeleton_length": 0
        }
    
    # Count neighbors for each skeleton pixel
    kernel = np.ones((3, 3), dtype=np.uint8)
    neighbor_count = cv2.filter2D(skeleton.astype(np.uint8), -1, kernel) * skeleton
    
    # Endpoints: 1 neighbor (+ center = 2)
    # Junctions: 3+ neighbors (+ center = 4+)
    endpoints = np.sum(neighbor_count == 2)
    junctions = np.sum(neighbor_count >= 4)
    
    total_skeleton_pixels = np.sum(skeleton)
    
    # Estimate average branch length
    # Branch = segment between endpoint and junction, or between two junctions
    num_branches = max(1, endpoints + junctions)
    average_branch_length = total_skeleton_pixels / num_branches
    
    return {
        "num_endpoints": int(endpoints),
        "num_junctions": int(junctions),
        "average_branch_length": float(average_branch_length),
        "total_skeleton_length": int(total_skeleton_pixels)
    }


def compute_tortuosity(binary_image: np.ndarray) -> float:
    """
    Compute vessel tortuosity (how curved the vessels are).
    
    Tortuosity = actual_length / straight_line_distance
    
    Args:
        binary_image: Binary image (0 or 255)
    
    Returns:
        Average tortuosity value
    """
    binary = (binary_image > 127).astype(np.uint8)
    skeleton = skeletonize(binary)
    
    if np.sum(skeleton) < 10:  # Too few pixels
        return 0.0
    
    # Label connected components in skeleton
    labeled_skeleton = measure.label(skeleton, connectivity=2)
    num_components = labeled_skeleton.max()
    
    if num_components == 0:
        return 0.0
    
    tortuosity_values = []
    
    for i in range(1, num_components + 1):
        component = (labeled_skeleton == i)
        coords = np.argwhere(component)
        
        if len(coords) < 2:
            continue
        
        # Actual length (number of pixels in skeleton)
        actual_length = len(coords)
        
        # Straight line distance (Euclidean distance between endpoints)
        # Find approximate endpoints (use furthest points)
        distances = np.linalg.norm(coords[:, None] - coords[None, :], axis=2)
        max_dist_idx = np.unravel_index(np.argmax(distances), distances.shape)
        straight_line_dist = distances[max_dist_idx]
        
        if straight_line_dist > 0:
            tortuosity = actual_length / straight_line_dist
            tortuosity_values.append(tortuosity)
    
    return float(np.mean(tortuosity_values)) if tortuosity_values else 1.0


def compute_gap_analysis(binary_image: np.ndarray, max_gap_distance: int = 20) -> Dict:
    """
    Analyze gaps in the vessel network.
    
    Args:
        binary_image: Binary image (0 or 255)
        max_gap_distance: Maximum distance to consider as a gap (pixels)
    
    Returns:
        Dictionary with gap statistics
    """
    binary = (binary_image > 127).astype(np.uint8)
    skeleton = skeletonize(binary)
    
    # Find endpoints
    kernel = np.ones((3, 3), dtype=np.uint8)
    neighbor_count = cv2.filter2D(skeleton.astype(np.uint8), -1, kernel) * skeleton
    endpoints_mask = (neighbor_count == 2)
    
    endpoint_coords = np.argwhere(endpoints_mask)
    
    if len(endpoint_coords) < 2:
        return {
            "num_potential_gaps": 0,
            "mean_gap_distance": 0.0,
            "min_gap_distance": 0.0
        }
    
    # Calculate distances between all pairs of endpoints
    from scipy.spatial.distance import pdist, squareform
    distances = squareform(pdist(endpoint_coords))
    
    # Find gaps (close endpoints that might need connection)
    # Exclude self-distances (diagonal = 0)
    np.fill_diagonal(distances, np.inf)
    
    # Get distances below threshold
    gap_distances = distances[distances < max_gap_distance]
    
    if len(gap_distances) == 0:
        return {
            "num_potential_gaps": 0,
            "mean_gap_distance": 0.0,
            "min_gap_distance": 0.0
        }
    
    # Count each gap once (divide by 2 since distance matrix is symmetric)
    num_gaps = len(gap_distances) // 2
    
    return {
        "num_potential_gaps": int(num_gaps),
        "mean_gap_distance": float(gap_distances.mean()),
        "min_gap_distance": float(gap_distances.min())
    }


def compute_vessel_continuity(binary_image: np.ndarray, max_gap_distance: int = 20) -> Dict:
    """
    Compute comprehensive vessel continuity metrics.
    
    Args:
        binary_image: Binary image (0 or 255)
        max_gap_distance: Maximum distance to consider as a gap (pixels)
    
    Returns:
        Dictionary with all continuity metrics
    """
    # Betti numbers
    betti_0, betti_1 = compute_betti_numbers(binary_image)
    
    # Connected components
    component_stats = compute_connected_components(binary_image)
    
    # Branch statistics
    branch_stats = compute_branch_statistics(binary_image)
    
    # Gap analysis
    gap_stats = compute_gap_analysis(binary_image, max_gap_distance)
    
    # Combine all metrics (excluding betti_1 and tortuosity)
    metrics = {
        "betti_0": betti_0,
        **component_stats,
        **branch_stats,
        **gap_stats
    }
    
    return metrics


def analyze_dataset(
    label_dir: str,
    output_dir: str,
    file_pattern: str = "*.bmp",
    max_gap_distance: int = 20,
    save_individual: bool = False
) -> pd.DataFrame:
    """
    Analyze vessel continuity for all images in a dataset.
    
    Args:
        label_dir: Directory containing label images
        output_dir: Directory to save results
        file_pattern: Pattern to match label files
        max_gap_distance: Maximum distance to consider as a gap
        save_individual: Whether to save individual JSON results
    
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
    
    for label_file in tqdm(label_files, desc="Analyzing continuity"):
        # Read image
        image = cv2.imread(str(label_file), cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            print(f"Warning: Could not read {label_file}")
            continue
        
        # Compute continuity metrics
        metrics = compute_vessel_continuity(image, max_gap_distance=max_gap_distance)
        
        # Add sample_name and filename
        sample_name = label_file.stem
        metrics["sample_name"] = sample_name
        metrics["filename"] = label_file.name
        
        results_list.append(metrics)
        
        # Save individual results
        if save_individual:
            individual_output = output_dir / "individual_results"
            individual_output.mkdir(exist_ok=True)
            with open(individual_output / f"{label_file.stem}.json", "w") as f:
                json.dump(metrics, f, indent=2)
    
    # Create DataFrame
    df = pd.DataFrame(results_list)
    
    # Reorder columns to put sample_name first
    if "sample_name" in df.columns:
        cols = ["sample_name"] + [col for col in df.columns if col != "sample_name"]
        df = df[cols]
    
    # Save summary statistics
    summary_file = output_dir / "continuity_summary.csv"
    df.to_csv(summary_file, index=False)
    print(f"\nSaved continuity summary to {summary_file}")
    
    # Compute and save aggregate statistics
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    aggregate_stats = {}
    
    for col in numeric_cols:
        aggregate_stats[f"{col}_mean"] = float(df[col].mean())
        aggregate_stats[f"{col}_std"] = float(df[col].std())
        aggregate_stats[f"{col}_median"] = float(df[col].median())
        aggregate_stats[f"{col}_min"] = float(df[col].min())
        aggregate_stats[f"{col}_max"] = float(df[col].max())
    
    aggregate_stats["total_images"] = len(df)
    
    aggregate_file = output_dir / "aggregate_statistics.json"
    with open(aggregate_file, "w") as f:
        json.dump(aggregate_stats, f, indent=2)
    print(f"Saved aggregate statistics to {aggregate_file}")
    
    # Print summary
    print("\n" + "="*50)
    print("VESSEL CONTINUITY ANALYSIS SUMMARY")
    print("="*50)
    print(f"Total images analyzed: {len(df)}")
    print(f"\nKey Metrics (mean ± std):")
    print(f"  Betti-0 (components): {aggregate_stats['betti_0_mean']:.2f} ± {aggregate_stats['betti_0_std']:.2f}")
    print(f"  Number of components: {aggregate_stats['num_components_mean']:.2f} ± {aggregate_stats['num_components_std']:.2f}")
    print(f"  Largest component ratio: {aggregate_stats['largest_component_ratio_mean']:.4f} ± {aggregate_stats['largest_component_ratio_std']:.4f}")
    print(f"  Average branch length: {aggregate_stats['average_branch_length_mean']:.2f} ± {aggregate_stats['average_branch_length_std']:.2f}")
    print(f"  Potential gaps: {aggregate_stats['num_potential_gaps_mean']:.2f} ± {aggregate_stats['num_potential_gaps_std']:.2f}")
    print("="*50)
    
    return df


def main():
    parser = argparse.ArgumentParser(description="Analyze vessel continuity from binary segmentation masks")
    parser.add_argument("--label_dir", type=str, required=True,
                        help="Directory containing label images")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save results")
    parser.add_argument("--file_pattern", type=str, default="*.bmp",
                        help="Pattern to match label files (default: *.bmp)")
    parser.add_argument("--max_gap_distance", type=int, default=20,
                        help="Maximum distance to consider as a gap in pixels (default: 20)")
    parser.add_argument("--save_individual", action="store_true",
                        help="Save individual JSON results for each image")
    
    args = parser.parse_args()
    
    # Run analysis
    df = analyze_dataset(
        label_dir=args.label_dir,
        output_dir=args.output_dir,
        file_pattern=args.file_pattern,
        max_gap_distance=args.max_gap_distance,
        save_individual=args.save_individual
    )
    
    print(f"\nAnalysis complete! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
