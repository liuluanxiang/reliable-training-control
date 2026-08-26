# External Data and Model Provenance

Recorded: 2026-08-25 (Asia/Shanghai)

## Tiny ImageNet-200

- Source record: https://zenodo.org/records/10720917
- Downloaded archive: `data/tiny-imagenet-200.zip`
- Archive size: 248,100,043 bytes.
- MD5: `90528d7ca1a48142e341f4ef8d21d0de` (verified before extraction).
- Extracted root: `data/tiny-imagenet-200/`.
- Verified inventory: 200 classes, 100,000 official training images, 10,000 labeled official validation images, and 10,000 unlabeled official test images.
- Protocol-v2 use: seed-fixed 90,000/10,000 split of official training data for optimization/internal validation; the labeled official validation split is isolated for one final test evaluation. The unlabeled official test split is not used.

## DeiT-Tiny/16

- Architecture source: https://github.com/facebookresearch/deit/blob/main/models.py
- Official checkpoint source: https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth
- Downloaded checkpoint: `data/pretrained/deit_tiny_patch16_224-a1311bcf.pth`
- Checkpoint size: 22,917,895 bytes.
- SHA-256: `a1311bcf4f24e3c95adaa75535db67bc4412d95535b98f7c1dfd1164dda41c97` (verified).
- Implementation: `timm` model `deit_tiny_patch16_224`, patch size 16, embedding dimension 192, depth 12, 3 attention heads.
- CIFAR-100 model size after replacing the head: 5,543,716 parameters.
- Initialization rule: load the official ImageNet-1k backbone, discard only `head.weight` and `head.bias`, and retain the seed-specific randomly initialized 100-class head.
- Fine-tuning input/optimizer: 224x224, ImageNet normalization, random resized crop/flip for training, deterministic resize/center crop for validation/test, AdamW with lr=0.0005 and weight decay=0.05.

Both external recipes are fixed before full multi-seed execution and are shared unchanged by all five comparison methods.
