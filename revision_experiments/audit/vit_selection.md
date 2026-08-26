# Selected External Architecture

- Model: DeiT-Tiny/16 (`deit_tiny_patch16_224`), approximately 5M parameters.
- Initialization: official ImageNet-1k pretrained checkpoint `deit_tiny_patch16_224-a1311bcf.pth`; the 1,000-class head is discarded and a seed-specific 100-class head is initialized.
- Data: CIFAR-100 resized to 224; ImageNet normalization; random resized crop/flip for training and deterministic resize/center-crop for validation/test.
- Optimization: AdamW, lr=0.0005, weight decay=0.05, batch=128, maximum 100 epochs.
- Comparison: five methods, seeds 1..5, equal data/model/optimizer/budget.
- Configs: `revision_experiments/configs/deit_cifar100/{smoke,full}.json`.
