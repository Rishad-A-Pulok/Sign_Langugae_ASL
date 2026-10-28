# American Sign Language Recognition using CNN and ResNet50V2

I worked on this project to compare a CNN built from scratch with a pretrained ResNet50V2 model for recognizing static American Sign Language hand signs. I used TensorFlow/Keras and ran the experiments in Google Colab.

## What I worked on

The dataset used in my notebooks contains 69,600 images across 29 folders: the 26 alphabet letters, `del`, `nothing`, and `space`. I resized the images to 224 × 224 and used an 80/20 directory split (55,680 images for training and 13,920 for validation).

I tried two approaches:

- **Custom CNN:** three convolution and max-pooling blocks, followed by dense layers and a 29-class softmax output.
- **ResNet50V2:** an ImageNet-pretrained backbone with frozen layers and a separately trained classification head.

Both models were trained for 15 epochs. The notebooks include learning curves, predictions on sample images, classification reports, and confusion matrices.

## Results from my saved notebooks

| Model | Reported evaluation accuracy | Approximate macro F1 |
| --- | ---: | ---: |
| Custom CNN | 64.02% | 0.63 |
| ResNet50V2 | 78.36% | 0.79 |

The ResNet50V2 approach performed better on the notebook's validation subset. These are **historical notebook outputs**, not results from a separate held-out test dataset or from the revised training script.

## Repository contents

```text
notebooks/              Original CNN and ResNet50V2 Colab notebooks
src/train.py            Revised, configurable training and evaluation script
requirements.txt        Python dependencies
reports/EVALUATION.md   Review of the original experiments and limitations
reports/RESULTS.md      Recorded notebook results
GITHUB_SETUP.md         Instructions for publishing and updating this repository
```

## Running the revised experiment

1. Obtain the image dataset used in the notebooks and extract it into a directory containing one subfolder per class. The original notebooks expect `/content/asl-full/train/` from a Google Drive ZIP named `asl-full.zip`. The original ZIP is not included in this repository, and its provenance should be documented before public redistribution.
2. Install Python 3.10+ and the dependencies:

```bash
python -m pip install -r requirements.txt
```

3. Run one of the revised experiments:

```bash
python src/train.py --data-dir /path/to/asl-full/train --model cnn
python src/train.py --data-dir /path/to/asl-full/train --model resnet50v2
```

The revised script uses a deterministic 70/15/15 split, augmentation only on training data, appropriate model-specific preprocessing, and a separate held-out test set. It saves evaluation metrics and a classification report to `outputs/`. Its results should not be directly compared with the historical 80/20 experiments because the data split and preprocessing differ.

## What I learned and what I would improve

Comparing these models helped me see the value of transfer learning for image classification. I also found several issues in the original evaluation setup that I want to address before making stronger claims: the validation split was called a test set, validation augmentation was enabled, and the ResNet50V2 input pipeline did not use the backbone's recommended preprocessing. A future version should also evaluate images from different signers and recording conditions, rather than relying only on a random image split.

This project recognizes **static ASL image categories**. It is not a complete real-time sign-language translation system, and it does not address continuous signing or sentence-level recognition.

## Tools

Python, TensorFlow/Keras, NumPy, scikit-learn, Matplotlib, Google Colab.

## Notes

The notebooks are retained as a record of the original experiments. The newer script is a proposed corrected pipeline and has not been trained on the original images in this package. Dataset licensing and provenance should be checked before uploading images or trained weights.
