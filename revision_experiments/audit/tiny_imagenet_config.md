# Tiny ImageNet Configuration

- Data: Tiny ImageNet-200, 200 classes, 64x64 RGB.
- Partitioning: seed-fixed 90/10 split of the 100,000 official training images; the 10,000 labeled official validation images are isolated as the final test set.
- Model: CIFAR-style ResNet-18, trained from scratch.
- Augmentation: random crop with 8-pixel padding and horizontal flip for training; deterministic evaluation transform for validation/test.
- Normalization: ImageNet mean/std.
- Optimization: SGD, lr=0.1, momentum=0.9, weight decay=0.0005, batch=128, maximum 120 epochs.
- Comparison: five methods, seeds 1..5, equal data/model/optimizer/budget.
- Configs: `revision_experiments/configs/tiny_imagenet/{smoke,full}.json`.
