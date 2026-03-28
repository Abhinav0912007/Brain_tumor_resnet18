# 🧠 Brain Tumor Detection using ResNet (PyTorch)

## 📌 Overview

This project focuses on detecting brain tumors from MRI images using Deep Learning techniques. A pre-trained **ResNet (Residual Network)** model is fine-tuned using **PyTorch** to classify MRI scans into tumor and non-tumor categories (or multiple tumor classes, depending on dataset).

The goal is to build an accurate, efficient, and scalable AI model that can assist in early diagnosis of brain tumors.

---

## 🚀 Features

* Uses **ResNet architecture** (e.g., ResNet18 / ResNet50)
* Transfer learning for faster training and better accuracy
* Image preprocessing and augmentation
* Model training, validation, and testing pipeline
* Performance metrics (Accuracy, Precision, Recall, F1-score)
* Easy-to-use and modular code structure

---

## 🧠 Dataset

* MRI brain scan images
* Common classes:

  * Tumor
  * No Tumor
    *(or multi-class: Glioma, Meningioma, Pituitary, No Tumor)*

**Dataset Structure Example:**

```
dataset/
│── train/
│   ├── tumor/
│   └── no_tumor/
│── val/
│   ├── tumor/
│   └── no_tumor/
│── test/
    ├── tumor/
    └── no_tumor/
```

---

## 🛠️ Tech Stack

* Python
* PyTorch
* Torchvision
* NumPy
* Matplotlib
* OpenCV (optional)

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/brain-tumor-detection-resnet.git
cd brain-tumor-detection-resnet
```

### 2. Create virtual environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 📊 Model Architecture

We use a pre-trained **ResNet model**:

* Replace the final fully connected layer
* Fine-tune on MRI dataset

Example:

```python
import torchvision.models as models
import torch.nn as nn

model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)
```

---

## 🏋️ Training

Run the training script:

```bash
python train.py
```

### Training includes:

* Data augmentation
* Loss function: CrossEntropyLoss
* Optimizer: Adam / SGD
* Learning rate scheduling

---

## 🧪 Evaluation

Run:

```bash
python test.py
```

Metrics:

* Accuracy
* Confusion Matrix
* Precision / Recall / F1-score

---

## 📈 Results

| Metric    | Value (Example) |
| --------- | --------------- |
| Accuracy  | 95%             |
| Precision | 94%             |
| Recall    | 96%             |
| F1 Score  | 95%             |

---

## 🖼️ Sample Output

* Correctly classified MRI scans
* Visualization of predictions

---

## 📂 Project Structure

```
├── dataset/
├── models/
│   └── resnet_model.py
├── utils/
│   └── preprocessing.py
├── train.py
├── test.py
├── requirements.txt
└── README.md
```

---

## 🔍 Future Improvements

* Use advanced architectures (EfficientNet, Vision Transformers)
* Deploy as a web app (Flask / FastAPI)
* Integrate Grad-CAM for explainability
* Use larger medical datasets for better generalization

---

## ⚠️ Disclaimer

This project is for educational and research purposes only. It is **not a substitute for professional medical diagnosis**.

---

## 🤝 Contributing

Contributions are welcome! Feel free to fork the repository and submit pull requests.

---

## 📜 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

Your Name
GitHub: [https://github.com/yourusername](https://github.com/yourusername)

---

⭐ If you like this project, give it a star!
