# Real-Time Shoplifting Detection Using Pose-Based LSTM

A lightweight, privacy-preserving shoplifting detection system designed to run on low-spec Algerian retail hardware. The system extracts human pose keypoints using YOLOv8-Pose, engineers 10 geometric features per person, and classifies 15-frame sequences as "Normal" or "Theft" using an LSTM with Attention.

## Architecture

The pipeline consists of four stages:

1. **Pose Extraction:** YOLOv8-Pose detects 17 body keypoints per person per frame.
2. **Feature Engineering:** 10 normalized geometric features are computed (torso length, wrist-to-hip distance, wrist-to-nose distance, elbow angles, relative wrist positions).
3. **Temporal Modeling:** A single-layer LSTM (64 hidden units) with a soft attention mechanism learns which frames in a 15-frame sequence are most important.
4. **Classification:** A sigmoid output produces the probability of theft.

## Dataset

The model is trained on the [PoseLift dataset](https://github.com/TeCSAR-UNCC/PoseLift) (WACV 2025), a privacy-preserving, pose-based shoplifting benchmark containing 153 video instances from 6 camera views. It includes 43 shoplifting and 112 normal scenarios, with frame-level ground-truth labels.

The dataset must be downloaded separately and placed at:
Pickle_files/
├── train/ (104 normal videos)
├── test/ (47 mixed videos)
└── GT/ (47 .npy label files)


## Setup

```
python -m venv venv
venv\Scripts\activate
pip install torch torchvision ultralytics opencv-python numpy scikit-learn

```


## Usage

Run the scripts in this exact order:

1. **Generate labels for the normal training videos**

   ```
   python create_train_labels.py
   
   ```

   Creates `Pickle_files/GT_normal/` with 104 `.npy` files of zeros.

2. **Split the dataset (stratified by video)**

   ```
   python split_dataset.py
   
   ```

   Creates `dataset_splits.txt` with:

   * Train: 105 videos (28 theft, 77 normal)

   * Validation: 22 videos (6 theft, 16 normal)

   * Test: 24 videos (7 theft, 17 normal)

3. **Train the model**

   ```
   python train.py
   
   ```

   Trains the LSTM with early stopping (best model saved from the epoch with the lowest validation loss) and a weighted loss function (`pos_weight`) to handle class imbalance. Saves `poselift_final_model.pth`.

4. **Evaluate on the held-out test set**

   ```
   python evaluate.py
   
   ```

   Reports Accuracy, Precision, Recall, F1-Score, AUC-ROC, and the Confusion Matrix.

5. **Run inference on a video**

   ```
   python inference.py
   
   ```

   Place a video named `test_video.mp4` in the project folder. The output is saved as `final_output.mp4`, with green boxes (normal) and red boxes (theft) overlaid on each person.

## Results

| Metric | Value | 
| ----- | ----- | 
| **Accuracy** | 92.14% | 
| **Recall** | 82.91% | 
| **Precision** | 22.81% | 
| **F1-Score** | 0.3577 | 
| **AUC-ROC** | 0.8948 | 

The AUC-ROC of 0.8948 indicates strong discriminative ability. The low precision is a deliberate trade-off from the weighted loss function, which prioritizes catching thefts over avoiding false alarms.

## Notes

* The model runs entirely offline on a CPU. No GPU or internet connection is required.

* The `Pickle_files/` folder, `venv/`, and `.pth` model weights are excluded from this repository via `.gitignore`.

* To reproduce the results, download the PoseLift dataset from the official GitHub link above.

## Citation

If you use this work, please cite the PoseLift paper:

```
@article{rashvand2025exploringposebasedanomalydetection,
  title={Exploring Pose-Based Anomaly Detection for Retail Security: A Real-World Shoplifting Dataset and Benchmark},
  author={Narges Rashvand and Ghazal Alinezhad Noghre and Armin Danesh Pazho and Shanle Yao and Hamed Tabkhi},
  year={2025},
  eprint={2501.06591},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}

```