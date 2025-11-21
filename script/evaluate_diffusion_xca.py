"""
Evaluate all trained diffusion models on XCA test data.

Usage Examples:
    # Basic usage (evaluate all diffusion models on XCA)
    uv run python script/evaluate_diffusion_xca.py
    
    # Specific models only
    uv run python script/evaluate_diffusion_xca.py --models "segdiff,medsegdiff,colddiff"
    
    # Custom output directory
    uv run python script/evaluate_diffusion_xca.py --output_dir results/xca_diffusion
    
    # Help
    uv run python script/evaluate_diffusion_xca.py --help

Parameters:
    --models: Comma-separated diffusion model names (default: all diffusion models)
    --output_dir: Output directory for results (default: results/diffusion_model)

Output:
    - CSV file: {output_dir}/evaluation_results_xca.csv
    - Individual predictions: lightning_logs/xca/{model_name}/predictions/
"""

import autorootcwd
import pandas as pd
import torch
from pathlib import Path
import lightning as L
import click
from src.archs.diffusion_model import DiffusionModel
from src.data.xca import XCADataModule
from src.loggers import PredictionLogger


def find_checkpoint(model_name):
    """Find checkpoint for diffusion model on XCA dataset."""
    # Find the latest checkpoint for this model
    checkpoint = None
    ckpt_files = list(Path("lightning_logs").glob(f"**/xca/{model_name}/checkpoints/*.ckpt"))
    
    if ckpt_files:
        # Sort by modification time and get the latest
        checkpoint = max(ckpt_files, key=lambda x: x.stat().st_mtime)
        print(f"Found checkpoint: {checkpoint}")
        return str(checkpoint)
    
    return None


def evaluate_model(model_name):
    """Evaluate single diffusion model on XCA dataset."""
    checkpoint = find_checkpoint(model_name)
    if not checkpoint:
        print(f"No checkpoint found for xca/{model_name}")
        return None
    
    print(f"Evaluating {model_name} on XCA dataset...")
    
    try:
        # Load model from checkpoint
        model = DiffusionModel.load_from_checkpoint(checkpoint)
        
        # Setup XCA data module with same settings as training
        data_module = XCADataModule(
            train_dir="data/xca_dataset_split/train",
            val_dir="data/xca_dataset_split/val",
            test_dir="data/xca_dataset_split/test",
            crop_size=320,  # XCA uses 320
            train_bs=1,  # Single batch for evaluation
            num_samples_per_image=1,  # No multi-sampling for evaluation
        )
        data_module.setup("test")
        
        # Create prediction logger
        logger = PredictionLogger(
            save_dir=f"lightning_logs/xca/{model_name}",
            name="predictions",
            version=None
        )
        
        # Create trainer with prediction logger
        trainer = L.Trainer(
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            logger=logger,
            enable_checkpointing=False,
            enable_progress_bar=True,
        )
        
        # Run test
        results = trainer.test(model, data_module)
        
        if results and len(results) > 0:
            test_metrics = results[0]
            test_metrics["Model"] = model_name
            return test_metrics
        
    except Exception as e:
        print(f"Error evaluating {model_name}: {e}")
        import traceback
        traceback.print_exc()
    
    return None


@click.command()
@click.option('--models', default='segdiff,medsegdiff,berdiff,colddiff,maskdiff', 
              help='Comma-separated list of diffusion model names')
@click.option('--output_dir', default='results/diffusion_model', help='Output directory for results')
def main(models, output_dir):
    """Evaluate all diffusion models on XCA dataset and create results table."""
    model_list = [m.strip() for m in models.split(',')]
    results = []
    
    print("="*80)
    print("EVALUATING DIFFUSION MODELS ON XCA DATASET")
    print("="*80)
    print(f"Models: {', '.join(model_list)}")
    print(f"Output: {output_dir}")
    print("="*80 + "\n")
    
    for model in model_list:
        result = evaluate_model(model)
        if result:
            results.append(result)
            print(f"✓ {model} evaluation completed\n")
        else:
            print(f"✗ {model} evaluation failed\n")
    
    if results:
        df = pd.DataFrame(results)
        df = df[["Model"] + [col for col in df.columns if col != "Model"]]
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save and print results
        results_path = f"{output_dir}/evaluation_results_xca.csv"
        df.to_csv(results_path, index=False)
        
        print("\n" + "="*80)
        print("XCA DIFFUSION MODEL EVALUATION RESULTS")
        print("="*80)
        print(df.to_string(index=False, float_format='%.4f'))
        print("="*80)
        print(f"\nResults saved to: {results_path}")
    else:
        print("\n" + "="*80)
        print("⚠ No evaluation results found!")
        print("Make sure models are trained first using train_diffusion_xca.sh")
        print("="*80)


if __name__ == "__main__":
    main()
