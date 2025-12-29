import yaml
from pathlib import Path
import torch
import autorootcwd
import src.losses  # noqa: F401 - register losses
from src.archs.flow_model import FlowModel
from src.data.xca import XCADataModule

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

exp_dir = Path("experiments/dhariwal_concat_unet_multihead/xca/dhariwal_concat_unet_multihead_xca_20251224_142312")
ckpt = exp_dir / "checkpoints" / "last.ckpt"
cfg = yaml.safe_load((exp_dir / "config.yaml").read_text())

dm = XCADataModule(
    train_dir=cfg["data"]["train_dir"],
    val_dir=cfg["data"]["val_dir"],
    test_dir=cfg["data"]["test_dir"],
    crop_size=cfg["data"]["image_size"],
    train_bs=1,
    num_samples_per_image=cfg["data"]["num_samples_per_image"],
    use_sauna_transform=cfg["data"]["use_sauna_transform"],
)
dm.setup("fit")
batch = next(iter(dm.train_dataloader()))
model = FlowModel.load_from_checkpoint(str(ckpt)).to(device).train()

images = batch["image"].to(device)
geometry = batch.get("geometry", batch.get("label")).to(device)
noise = torch.randn_like(geometry, device=device)
t, xt, ut = model.flow_matcher.sample_location_and_conditional_flow(noise, geometry)
unet_out = model.unet(xt, t, images)
geom_pred = unet_out[:, 1:2]

print("geom_pred min/max:", geom_pred.min().item(), geom_pred.max().item())
frac = ((geom_pred >= 0) & (geom_pred <= 1)).float().mean().item()
print("geom_pred in [0,1]:", frac)
