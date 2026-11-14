# Baseline vs detector

Image-level metrics use the same fire / smoke / normal labels for both models (macro average over the classes present in each test set). mAP is box-level and only exists for the detector. Inference is ms per image, batch size 1, CPU.

## dfire_test (4306 images)

| Model | Accuracy | Precision | Recall | F1 | mAP@0.5 | mAP@0.5:0.95 | Inference (ms) |
|---|---|---|---|---|---|---|---|
| Baseline (MobileNetV3-small) | 0.769 | 0.775 | 0.798 | 0.772 | n/a | n/a | 24.9 |
| Detector (YOLO nano), image-level | 0.889 | 0.901 | 0.870 | 0.882 | 0.703 | 0.392 | 91.4 |

Detector box-level: precision 0.716, recall 0.632, F1 0.671. Per class mAP@0.5: smoke 0.758, fire 0.649.

## simuletic_test (50 images)

| Model | Accuracy | Precision | Recall | F1 | mAP@0.5 | mAP@0.5:0.95 | Inference (ms) |
|---|---|---|---|---|---|---|---|
| Baseline (MobileNetV3-small) | 0.820 | 0.831 | 0.817 | 0.824 | n/a | n/a | 25.6 |
| Detector (YOLO nano), image-level | 0.940 | 1.000 | 0.942 | 0.970 | 0.816 | 0.383 | 156.7 |

Detector box-level: precision 0.762, recall 0.818, F1 0.789. Per class mAP@0.5: smoke 0.753, fire 0.880.
