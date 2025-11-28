# cifar10_Image-classificatiom

This project implements a Convolutional Neural Network (CNN) trained from scratch on the CIFAR-10 dataset using TensorFlow/Keras. The model includes multiple convolution blocks, dropout regularization, and dense layers to achieve multiclass classification across 10 object categories.

## Model Architecture (Custom CNN)
The model is manually designed using standard CNN layers:

- Conv2D(32, 3x3, ReLU, same)
- Conv2D(32, 3x3, ReLU, same)
- MaxPooling + Dropout(0.25)

- Conv2D(64, 3x3, ReLU, same)
- Conv2D(64, 3x3, ReLU, same)
- MaxPooling + Dropout(0.25)

- Flatten
- Dense(512, ReLU) + Dropout(0.5)
- Dense(10, Softmax)

This architecture is simple, efficient, and well-suited for CIFAR-10 training.

---

##  Features
- Full training pipeline on CIFAR-10  
- Data augmentation (rotation, shift, flip)  
- Accuracy & loss plots  
- Confusion matrix visualization  
- Automatic saving of best model (`model_colab.h5`)  
- Prediction script for custom images  
- Google Colab friendly (GPU supported)

---

## 📦Files in this Repository
- `train.py` – Training script for custom CNN  
- `predict.py` – Predict single images  
- `requirements.txt` – Install dependencies  
- `utils.py` – Plotting & evaluation tools  
- `README.md` – Documentation  
- `models/` – Folder for saved `.h5` models  
- `plots/` – Auto-generated accuracy/loss graphs  
- `examples/` – Sample test images

---

##  How to Run (Colab)
1. Upload repository to Colab  
2. Install dependencies  
   ```bash
   pip install -r requirements.txt

