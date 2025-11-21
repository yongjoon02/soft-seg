"""ROSSA Retinal Vessel Segmentation Dataset

ROSSA dataset with manual and SAM annotations.
No label_prob - simpler than OCTA dataset.
"""
import autorootcwd
from torch.utils.data import ConcatDataset
from src.data.base_dataset import BaseOCTDataset, BaseOCTDataModule
from src.utils.visualize_dataloader import visualize_dataset
from src.utils.registry import DATASET_REGISTRY


class ROSSADataset(BaseOCTDataset):
    """ROSSA Dataset with 2 fields: image, label (soft labels generated during training)"""
    
    def get_data_fields(self) -> list[str]:
        """Only load image and label - soft labels will be generated dynamically"""
        return ['image', 'label']


class ROSSADataModule(BaseOCTDataModule):
    """ROSSA DataModule combining train_manual and train_sam"""
    
    dataset_class = ROSSADataset
    
    def __init__(self, 
                 train_manual_dir="data/ROSSA/train_manual",
                 train_sam_dir="data/ROSSA/train_sam",
                 val_dir="data/ROSSA/val", 
                 test_dir="data/ROSSA/test", 
                 crop_size=128, 
                 train_bs=8, 
                 num_samples_per_image=1):
        """
        Args:
            train_manual_dir: Manual annotation training data
            train_sam_dir: SAM annotation training data
            val_dir: Validation data
            test_dir: Test data
            crop_size: Crop size for training
            train_bs: Training batch size
            num_samples_per_image: Samples per image for augmentation
        """
        # Store ROSSA-specific directories
        self.train_manual_dir = train_manual_dir
        self.train_sam_dir = train_sam_dir
        
        # Call parent init with None for train_dir (we use manual+sam instead)
        super().__init__(
            train_dir=None,
            val_dir=val_dir,
            test_dir=test_dir,
            crop_size=crop_size,
            train_bs=train_bs,
            num_samples_per_image=num_samples_per_image,
            name='rossa'
        )
    
    def create_train_dataset(self):
        """Create training dataset by combining manual and SAM annotations"""
        # Train: Combine manual and SAM annotations
        train_manual_dataset = self.dataset_class(
            self.train_manual_dir, 
            augmentation=True, 
            crop_size=self.crop_size,
            num_samples_per_image=self.num_samples_per_image
        )
        train_sam_dataset = self.dataset_class(
            self.train_sam_dir, 
            augmentation=True, 
            crop_size=self.crop_size,
            num_samples_per_image=self.num_samples_per_image
        )
        
        # Concatenate manual and SAM datasets
        combined_dataset = ConcatDataset([train_manual_dataset, train_sam_dataset])
        
        print(f"ROSSA Dataset loaded:")
        print(f"  Train (manual): {len(train_manual_dataset)} samples")
        print(f"  Train (SAM): {len(train_sam_dataset)} samples")
        print(f"  Train (total): {len(combined_dataset)} samples")
        
        return combined_dataset


@DATASET_REGISTRY.register(name='rossa')
class ROSSA_DataModule(ROSSADataModule):
    """ROSSA DataModule registered in dataset registry"""
    def __init__(self, 
                 train_manual_dir="data/ROSSA/train_manual",
                 train_sam_dir="data/ROSSA/train_sam",
                 val_dir="data/ROSSA/val", 
                 test_dir="data/ROSSA/test", 
                 crop_size=128, 
                 train_bs=8, 
                 num_samples_per_image=1):
        super().__init__(
            train_manual_dir=train_manual_dir,
            train_sam_dir=train_sam_dir,
            val_dir=val_dir,
            test_dir=test_dir,
            crop_size=crop_size,
            train_bs=train_bs,
            num_samples_per_image=num_samples_per_image
        )


if __name__ == "__main__":
    print("=" * 70)
    print("Testing ROSSA Dataset")
    print("=" * 70)
    
    datamodule = DATASET_REGISTRY.get('rossa')()
    datamodule.setup()
    
    # Visualize
    visualize_dataset(datamodule.train_dataloader(), "rossa_train")
    visualize_dataset(datamodule.val_dataloader(), "rossa_val")
    visualize_dataset(datamodule.test_dataloader(), "rossa_test")
    
    print("\n✓ ROSSA dataset works correctly!")
