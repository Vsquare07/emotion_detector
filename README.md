# CODEFEST'26 IIT BHU: Facial Expression Recognition <a href="https://docs.google.com/document/d/1dxECWTUPwEAqB5bG1lmVBrcVxnf02lNuHl-S-U3YTDs/edit?usp=sharing">Vision Quest</a>

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Git](https://img.shields.io/badge/git-%23F05033.svg?style=for-the-badge&logo=git&logoColor=white)

A deep learning project focused on classifying human emotions from **48x48 grayscale images** using a custom Convolutional Neural Network (CNN) trained on the **FER-2013** dataset provided in problem statement.

---

## 📂 Project Architecture

The project is modularized for clarity and scalability:

* **`model.py`**: Contains the `myModel` class architecture.
* **`train.py`**: Handles the training pipeline, data loading, and optimization.
* **`test.py`**: Evaluation script that generates accuracy metrics and the confusion matrix.
* **`visualizer.py`**: A utility to overlay actual vs. predicted labels on sample images.

---

## 🧠 Model Specifications

The architecture consists of three sequential convolutional blocks followed by a dense classifier. _The following output is generated from `model.py`_

<p align="center">
  <img src="https://github.com/user-attachments/assets/649ebe35-67d9-425e-9ea4-dfb3791866a0" width="700" alt="Model Architecture">
</p>

| Component | Details |
| :--- | :--- |
| **Optimizer** | Adam |
| **Loss Function** | Cross Entropy Loss |
| **Batch Size** | 16 |
| **Learning Rate** | Step-down from 0.01 to 0.001 |

---

## 📈 Training Results & Analytics

The model was trained in two phases: **50 epochs** at $LR=0.01$ and **50 epochs** at $LR=0.001$.

### Loss Convergence
<p align="center">
  <img src="https://github.com/user-attachments/assets/a649f37d-6901-43e2-b1fd-3b1e67290bb7" width="400" />
  <img src="https://github.com/user-attachments/assets/b635b3f5-93ac-4d8d-8573-41f13e036534" width="400" />
</p>

### Performance Metrics
* **Training Accuracy:** 77%
* **Testing Accuracy:** 56% _(generated from `test.py`)_

The significant gap between training and testing accuracy suggests a high degree of **overfitting**, likely due to the model memorizing dataset-specific noise and the model getting confused between different emotions with same expression.

The trained models are present in `/models` with the best one named as `model4(best).pth`

### Confusion Matrix
_generated from `test.py`_
<p align="center">
  <img src="https://github.com/user-attachments/assets/8d1dd1bd-98a6-490a-bfce-de20b0a03eac" width="600" alt="Confusion Matrix" />
</p>

---

## ⚠️ Dataset Limitations & Observations

A critical takeaway from this project is the inherent flaw in the **FER-2013** labels. Visualization reveals that human emotion is far more versatile than a single rigid label allows. The following examples show the flaw in predicting human expression just on the basis of facial expressions as expression can be related to more than one type of emotion.

> **Key Format:** `True Label` -> `Predicted Label` with the red ones being incorrect prediction and green ones being correct

<br>_the following images are generated from `visualizer.py`_
<p align="center">
  <img src="https://github.com/user-attachments/assets/281c87d1-f10f-4a55-a1b8-54a1986f9c02" width="800" />
  <img src="https://github.com/user-attachments/assets/f24270e2-c7ee-42da-b025-95fab31e28e8" width="800" />
  <img src="https://github.com/user-attachments/assets/09a0af9c-cac1-432c-9f0b-dc5eadcce6d5" width="800" />
</p>

**In several instances, the model's "incorrect" prediction appears more human-accurate than the dataset's ground truth.** This highlights the difficulty of training models on subjective emotional data where multiple emotions can manifest through similar facial expressions.

---

## 🛠️ Environment
* **uv**: Python package management
* **MPS**: Metal Performance Shaders for GPU acceleration(for apple system). _One can also use CUDA(for nvidia GPU)_
* **Libraries**: Refer to requirements.txt





