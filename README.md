## Parkinson's Disease Detection via Spiral Drawings (CNN)

This project implements a **Convolutional Neural Network (CNN)** for the automated detection of **Parkinson's Disease (PD)** using digitized images of hand-drawn spirals. The application is deployed using **Streamlit** for a user-friendly web interface, allowing individuals to upload a spiral drawing for immediate prediction.

---

### Overview

Parkinson's Disease is a progressive neurological disorder that affects movement. One common method for preliminary assessment involves analyzing the quality and consistency of a patient's hand-drawn spiral. Tremors, micrographia (small handwriting), and rigidity caused by PD often lead to noticeable distortions in these drawings.

This system is built upon a deep learning model trained on a publicly available dataset of both healthy and Parkinsonian spiral drawings.

---

### 💻 Technology Stack

* **Python:** Primary programming language.
* **TensorFlow/Keras:** Deep Learning framework for building and training the CNN model.
* **Streamlit:** For creating the interactive web application interface.
* **PIL (Pillow) & NumPy:** For image loading, manipulation, and numerical processing.
* **Scikit-learn, Matplotlib, Seaborn:** For model evaluation (Classification Report, Confusion Matrix).

---

### 📋 Dataset

The model was trained on the **Spiral Drawings for Parkinson's Classification** dataset (Version 3) sourced from **Roboflow**.

| Metric | Value |
| :--- | :--- |
| **Total Images** | **8,640** |
| **Train Set** | 7,758 Images (90%) |
| **Validation Set** | 534 Images (6%) |
| **Test Set** | 348 Images (4%) |

The images were preprocessed to ensure consistency before training. The preprocessing steps include: **Auto-Orient**, **Resize to 640x640** (stretch), **Grayscale**, and **Auto-Adjust Contrast (Adaptive Equalization)**.

The original dataset can be found here: [Roboflow Dataset Link](https://universe.roboflow.com/parkinson-classification/my-first-project-etocp/dataset/3)

---

### 🚀 Model Architecture & Training

#### Model Architecture (Defined in `train.py`)

The CNN uses a sequential architecture optimized for grayscale image classification:

1.  **Input:** (256 x 256 x 1) Grayscale image.
2.  **Conv Block 1:** `Conv2D(64)` $\rightarrow$ `MaxPooling2D`
3.  **Conv Block 2:** `Conv2D(128)` $\rightarrow$ `MaxPooling2D` $\rightarrow$ `Dropout(0.3)`
4.  **Conv Block 3:** `Conv2D(256)` $\rightarrow$ `MaxPooling2D` $\rightarrow$ `Dropout(0.4)`
5.  **Classifier Head:** `Flatten` $\rightarrow$ `Dense(256, activation='relu')` $\rightarrow$ `Dropout(0.5)` $\rightarrow$ `Dense(1, activation='sigmoid')`

#### Training Configuration

* **Optimizer:** Adam with a learning rate of $\mathbf{0.0001}$.
* **Loss Function:** $\mathbf{Binary\ Crossentropy}$.
* **Callbacks:** **ModelCheckpoint** and **EarlyStopping** (patience=10 on `val_loss`).
* **Augmentation (Training Data):** Rotation, width/height shifts, and zoom ranges.

---

### 📝 Model Evaluation

The final model (`parkinsons_detection_model.h5`) was assessed on the independent Test Set (348 images), yielding the following results:

| Metric | Result |
| :--- | :--- |
| **Test Accuracy** | **95.69%** |
| **Precision (Parkinson's Class)** | **98%** |
| **Recall (Parkinson's Class)** | **93%** |

The high precision suggests the model is very good at avoiding false alarms (minimizing **False Positives**), while the strong recall ensures that very few actual Parkinson's cases are missed (minimizing **False Negatives**).


### ⚙️ How to Run the Application

#### 1. Setup

Clone the repository and install the required Python libraries (you may need to install `tensorflow`, `streamlit`, `Pillow`, `numpy`, `scikit-learn`, `matplotlib`, and `seaborn`):

```bash
git clone <repository-link>
cd Parkinsons-Disease-Detection
pip install -r requirements.txt
python train.py
streamlit run app.py

