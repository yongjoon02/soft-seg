"""XCA (X-ray Coronary Angiography) 데이터셋 모듈

XCA 데이터는 관상동맥 조영술 이미지의 혈관 분할 데이터셋입니다.
X-ray 이미지는 ScaleIntensityd로 정규화하여 [-1, 1] 범위로 변환합니다.
"""
import autorootcwd
import math
import torch
from typing import Optional

from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    ScaleIntensityd,
    RandRotated,
    RandCropByPosNegLabeld,
    RandAdjustContrastd,
    RandFlipd,
    RandRotate90d,
    RandSpatialCropd,
)
from src.data.base_dataset import BaseOCTDataset, BaseOCTDataModule
from src.utils.registry import DATASET_REGISTRY


class XCADataset(BaseOCTDataset):
    """XCA 데이터셋 (BaseOCTDataset 상속)
    
    데이터 구조:
        root/
            image/  - 원본 X-ray 이미지 (grayscale PNG, 512x512)
            label/  - 혈관 분할 마스크 (grayscale PNG, 512x512)
    
    X-ray 특화 처리:
    - RGB/RGBA → Grayscale 변환 (첫 번째 채널만 사용)
    - RandRotated: X-ray 특성에 맞는 작은 회전 (±7.5도)
    - RandAdjustContrastd: X-ray 대비 조정
    """
    
    def get_data_fields(self) -> list[str]:
        """XCA는 image와 label만 사용"""
        return ['image', 'label']
    
    def _create_transforms(self):
        """X-ray 특화 transform 생성 (Base 오버라이드)"""
        # Base의 기본 transform 먼저 생성
        super()._create_transforms()
        
        keys = self.fields
        
        # RGB→Grayscale 변환 추가 (X-ray 특화)
        rgb_to_gray = lambda d: {
            **d,
            "image": d["image"][:1] if hasattr(d["image"], "shape") and d["image"].shape[0] > 1 else d["image"],
            "label": d["label"][:1] if hasattr(d["label"], "shape") and d["label"].shape[0] > 1 else d["label"],
        }
        
        # Default transforms에 RGB→Gray 추가
        self.default_transforms = Compose([
            self.default_transforms.transforms[0],  # EnsureChannelFirstd
            rgb_to_gray,  # RGB→Grayscale
            ScaleIntensityd(keys="image", minv=-1.0, maxv=1.0),
            ScaleIntensityd(keys="label", minv=0.0, maxv=1.0),
        ])
        
        # Base augmentation에 X-ray 특화 transform 추가
        xray_augments = [
            RandRotated(keys=keys, range_x=(math.pi/24, math.pi/24), 
                       range_y=(math.pi/24, math.pi/24), prob=0.25),
            RandAdjustContrastd(keys="image", prob=0.25, gamma=(0.9, 1.1)),
        ]
        
        # 기존 augmentation 앞에 X-ray augmentation 추가
        base_transforms = list(self.augmentation_transforms.transforms)
        self.augmentation_transforms = Compose(base_transforms[:3] + xray_augments + base_transforms[3:])
    
    def __getitem__(self, index):
        """
        Get a sample with X-ray specific post-processing.
        
        Args:
            index: Sample index
            
        Returns:
            dict: Dictionary containing image, label, and metadata
        """
        # Base class의 __getitem__ 호출
        data = super().__getitem__(index)
        
        # X-ray 특화 후처리: 이미지 clamp (augmentation 후 범위 벗어날 수 있음)
        if self.augmentation:
            data["image"] = torch.clamp(data["image"], -1.0, 1.0)
        
        return data


class XCADataModule(BaseOCTDataModule):
    """XCA 데이터 모듈 (BaseOCTDataModule 상속)
    
    Usage:
        datamodule = XCADataModule(
            train_dir='data/xca_dataset_split/train',
            val_dir='data/xca_dataset_split/val',
            test_dir='data/xca_dataset_split/test',
            crop_size=320,
            train_bs=8,
            num_samples_per_image=1,
        )
    """
    
    dataset_class = XCADataset
    
    def __init__(
        self,
        train_dir: str = 'data/xca_dataset_split/train',
        val_dir: str = 'data/xca_dataset_split/val',
        test_dir: Optional[str] = 'data/xca_dataset_split/test',
        crop_size: int = 320,
        train_bs: int = 8,
        num_samples_per_image: int = 1,
    ):
        """XCA 데이터 모듈 초기화
        
        Args:
            train_dir: 학습 데이터 디렉토리
            val_dir: 검증 데이터 디렉토리
            test_dir: 테스트 데이터 디렉토리 (선택)
            crop_size: 크롭 크기 (default: 320, 원본 512×512의 62.5%)
            train_bs: 학습 배치 크기
            num_samples_per_image: 이미지당 크롭 샘플 수
        """
        super().__init__(
            train_dir=train_dir,
            val_dir=val_dir,
            test_dir=test_dir,
            crop_size=crop_size,
            train_bs=train_bs,
            num_samples_per_image=num_samples_per_image,
            name='xca'
        )
    
    def create_train_dataset(self):
        """Create training dataset from single directory"""
        return self.dataset_class(
            self.train_dir,
            augmentation=True,
            crop_size=self.crop_size,
            num_samples_per_image=self.num_samples_per_image
        )


@DATASET_REGISTRY.register(name='xca')
class XCA_DataModule(XCADataModule):
    """Registry에 등록된 XCA 데이터 모듈 (기본 파라미터)"""
    def __init__(
        self,
        train_dir: str = 'data/xca_dataset_split/train',
        val_dir: str = 'data/xca_dataset_split/val',
        test_dir: Optional[str] = 'data/xca_dataset_split/test',
        crop_size: int = 320,
        train_bs: int = 8,
        num_samples_per_image: int = 1,
    ):
        super().__init__(
            train_dir=train_dir,
            val_dir=val_dir,
            test_dir=test_dir,
            crop_size=crop_size,
            train_bs=train_bs,
            num_samples_per_image=num_samples_per_image,
        )


if __name__ == '__main__':
    from src.utils.visualize_dataloader import visualize_dataset
    
    print("=" * 60)
    print("XCA 데이터셋 테스트 (BaseOCTDataset 상속)")
    print("=" * 60)
    
    # Registry에서 가져오기
    dm = DATASET_REGISTRY.get('xca')()
    dm.setup()
    
    # 데이터 개수 확인
    print(f"\n📊 데이터 개수:")
    print(f"   Train: {len(dm.train_dataset)} 샘플")
    print(f"   Val:   {len(dm.val_dataset)} 샘플")
    if dm.test_dataset:
        print(f"   Test:  {len(dm.test_dataset)} 샘플")
    
    # 샘플 데이터 확인
    train_sample = dm.train_dataset[0]
    print(f"\n📦 샘플 데이터 shape:")
    print(f"   Image: {train_sample['image'].shape} (range: {train_sample['image'].min():.2f} ~ {train_sample['image'].max():.2f})")
    print(f"   Label: {train_sample['label'].shape} (range: {train_sample['label'].min():.2f} ~ {train_sample['label'].max():.2f})")
    
    # 시각화
    visualize_dataset(dm.train_dataloader(), "xca_train", num_samples=10)
    visualize_dataset(dm.val_dataloader(), "xca_val")
    if dm.test_dataset:
        visualize_dataset(dm.test_dataloader(), "xca_test")
    
    print("\n✅ XCA 데이터셋 테스트 완료!")
