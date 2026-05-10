# Crowd Head Detection and Counting — Custom CNN (No YOLO)

A custom Convolutional Neural Network built from scratch for crowd head detection and density-based counting. Deliberately avoids pre-built object detectors like YOLO to demonstrate fundamental understanding of CNN architecture and regression-based counting.

---

## Project Overview

This project frames crowd counting as a regression problem. Given an image of a crowd, the model predicts the total number of heads present. The CNN is trained end-to-end on crowd images and evaluated across multiple density ranges to understand performance at different crowd sizes.

Key design decision: no YOLO, no pretrained detector, no shortcut. The architecture is built and trained from the ground up.

---

## Results

The model predictions track closely with ground truth counts across density ranges from 2 to 80 heads. Sample prediction: **Predicted 20.8 vs Ground Truth 24.0** — within 4 heads on a real crowd scene.

Performance is analysed across three density bins:
- Low density (0-20 heads)
- Mid density (20-40 heads)
- High density (40-80 heads)

---

## Visualisations

**Sample Prediction on Real Crowd Image**
`epoch_017.png`

**Predictions vs True Values (Full Range)**
`predictions.png`

**Prediction Distribution by Bin**
`bin_histogram.png`

**Confusion Matrix — Full Range**
`confusion_matrix_full.png`

**Confusion Matrix — Higher Density Range**
`confusion_matrix_higher.png`

**Confusion Matrix — Lower Density Range**
`confusion_matrix_lower.png`

**Higher Range Predictions**
`higher_range_predictions.png`

**Higher Range Bin Histogram**
`higher_range_bin_histogram.png`

**Lower Range Predictions**
`lower_range_predictions.png`

**Lower Range Bin Histogram**
`lower_range_bin_histogram.png`

---

## Methodology

### Problem Formulation
Crowd counting is treated as a regression task. The model outputs a single continuous value representing the predicted head count for a given image.

### Architecture
A custom CNN built without any pretrained backbone:
- Multiple convolutional blocks with ReLU activations and max pooling
- Batch normalisation for training stability
- Fully connected regression head outputting a single count value
- Trained with MSE loss

### Density-Based Evaluation
Rather than reporting a single global metric, the model is evaluated separately across density ranges. This gives a clearer picture of where the model performs well and where it struggles — important for real-world crowd management applications.

---

## Repository Structure

```
crowd-head-detection-cnn/
│
├── model.py       # CNN architecture definition
├── dataset.py     # Data loading and preprocessing
├── train.py       # Training loop and configuration
├── evaluate.py    # Evaluation, binned analysis, and visualisation
├── utils.py       # Helper functions
```

---

## Setup & Usage

### 1. Install Dependencies

```bash
pip install torch torchvision opencv-python numpy matplotlib scikit-learn pandas
```

### 2. Download Dataset

This project uses a crowd counting dataset from Kaggle. Place images in a `data/` directory following the structure expected by `dataset.py`.

### 3. Train the Model

```bash
python train.py
```

### 4. Evaluate and Generate Visualisations

```bash
python evaluate.py
```

---

## Key Libraries

- PyTorch — CNN architecture and training
- OpenCV — Image preprocessing
- NumPy, Pandas — Data handling
- Matplotlib, Seaborn — Visualisation and evaluation plots
- scikit-learn — Binned evaluation metrics

---

## Author

**Omer Farooq**
BS Artificial Intelligence — Information Technology University (ITU), Lahore
[GitHub](https://github.com/omer-687)

---

## Note

This project was completed as part of Computer Vision coursework at ITU Lahore. The focus was on understanding CNN fundamentals by building a counting system without relying on pre-built object detection frameworks.
