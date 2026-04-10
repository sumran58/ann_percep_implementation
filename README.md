# Artificial Neural Network Perceptron Implementation for practicing the ann and perceptron implementation

A comprehensive machine learning project implementing and comparing **Perceptron** and **Artificial Neural Network** classifiers on the Iris dataset. This project demonstrates fundamental concepts in neural networks, data preprocessing, model training, and performance evaluation.

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [Author](#author)

## 🎯 Project Overview

This project compares two classification approaches:
1. **Perceptron Model** - A simple linear classifier from scikit-learn
2. **Artificial Neural Network (ANN)** - A deeper sequential neural network using TensorFlow/Keras

Both models are trained and evaluated on the classic Iris flower dataset to classify flowers into three species based on morphological features.

## ✨ Features

- **Data Exploration**: Comprehensive exploratory data analysis with visualization
- **Data Preprocessing**: Feature scaling using StandardScaler and label encoding
- **Perceptron Classifier**: Simple linear model with 90% accuracy on test set
- **Neural Network**: Multi-layer sequential model achieving 96.67% test accuracy
- **Model Comparison**: Side-by-side evaluation of both approaches
- **Visualization**: Pair plots and performance metrics analysis
- **Detailed Documentation**: Code comments explaining each step

## 📊 Dataset

**Iris Dataset** (150 samples, 3 classes)
- Features: Sepal length, Sepal width, Petal length, Petal width
- Target: Flower species (Setosa, Versicolor, Virginica)
- Train/Test Split: 80-20 (120 training, 30 test samples)
- Source: Seaborn built-in dataset

Class Distribution:
- Setosa: 50 samples
- Versicolor: 50 samples
- Virginica: 50 samples

## 🚀 Installation

### Prerequisites
- Python 3.7+
- Jupyter Notebook or JupyterLab

### Required Libraries
```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow keras
```

Or install using requirements.txt:
```bash
pip install -r requirements.txt
```

### Recommended Installation
```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow
```

## 📁 Project Structure

```
ann_perceptron_implementation/
├── ann_perceptron_implementation.ipynb  # Main notebook
├── README.md                            # This file
└── requirements.txt                     # Package dependencies
```

## 💻 Usage

1. **Clone or download the repository**
   ```bash
   git clone https://github.com/sumran58/ann_percep_implementation.git
   cd ann_perceptron_implementation
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Open the notebook**
   ```bash
   jupyter notebook ann_perceptron_implementation.ipynb
   ```

4. **Run all cells** (Kernel → Restart & Run All) or run cells individually

## 🧠 Model Architecture

### Perceptron Model
```
Input Features (4)
    ↓
Perceptron Classifier
    ↓
Output (3 classes)
```
- **Optimizer**: Default (SGD)
- **Max Iterations**: 1000
- **Activation**: Threshold function
- **Test Accuracy**: 90.0%

### Artificial Neural Network
```
Input Layer (4 features)
    ↓
Dense Layer 1 (16 units, ReLU activation)
    ↓
Dense Layer 2 (8 units, ReLU activation)
    ↓
Output Layer (3 units, Softmax activation)
```

**Model Specifications:**
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Metrics**: Accuracy
- **Epochs**: 100
- **Batch Size**: 8
- **Validation Split**: 20%
- **Test Accuracy**: 96.67%

## 📈 Results

### Perceptron Performance
```
Precision  Recall  F1-Score  Support
    0        1.00      1.00      1.00       10
    1        1.00      0.70      0.82       10
    2        0.77      1.00      0.87       10

Accuracy: 0.90
```

### Neural Network Performance
```
Validation Accuracy: 100% (at epoch 48+)
Test Accuracy: 96.67%
Test Loss: 0.0295
```

### Key Observations
- ANN outperforms the simple Perceptron (96.67% vs 90%)
- ANN achieves perfect validation accuracy by epoch 48
- Both models handle the linearly separable Setosa class well
- ANN shows better generalization on Versicolor and Virginica classes

## 📚 Technologies Used

| Technology | Purpose |
|-----------|---------|
| **Python** | Programming language |
| **NumPy** | Numerical computations |
| **Pandas** | Data manipulation and analysis |
| **Matplotlib** | Static visualization |
| **Seaborn** | Statistical visualization |
| **Scikit-Learn** | Perceptron model and metrics |
| **TensorFlow** | Neural network framework |
| **Keras** | High-level neural network API |
| **Jupyter** | Interactive notebook environment |

## 🔍 Code Walkthrough

### Step 1: Data Loading
```python
df = sns.load_dataset('iris')
```

### Step 2: Data Exploration
```python
X = df.drop(columns=['species'], axis=1)
y = df['species']
```

### Step 3: Preprocessing
```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### Step 4: Perceptron Training
```python
per = Perceptron(max_iter=1000, random_state=42)
per.fit(X_train_scaled, y_train)
```

### Step 5: ANN Model Definition
```python
model = Sequential([
    Dense(16, input_dim=4, activation='relu'),
    Dense(8, activation='relu'),
    Dense(3, activation='softmax')
])
```

### Step 6: Training & Evaluation
```python
model.fit(X_train_scaled, y_train_cat, 
          epochs=100, batch_size=8, 
          validation_split=0.2)
```

## 🎓 Learning Outcomes

After working with this project, you'll understand:
- How to preprocess and explore datasets
- The differences between linear and non-linear classifiers
- Training neural networks with Keras
- Evaluating models with various metrics
- The importance of feature scaling
- Validation and test set evaluation
- Model comparison and selection

## 📝 Notes

- The Iris dataset is relatively small and linearly separable, making it suitable for learning
- Feature scaling is crucial for the Perceptron and improves ANN performance
- The ANN's deeper architecture allows it to learn non-linear decision boundaries
- Both models achieve high accuracy, but ANN generalizes better on this dataset
- Consider trying different architectures and hyperparameters for further improvement

## 🔄 Potential Improvements

- Add dropout layers to the ANN to prevent overfitting
- Implement cross-validation for more robust evaluation
- Try different activation functions (Tanh, Sigmoid, ELU)
- Experiment with different optimizers (SGD, RMSprop)
- Add early stopping to prevent overfitting
- Visualize decision boundaries
- Try other datasets with more complexity

## ⚖️ License

This project is open source and available under the MIT License. Feel free to use, modify, and distribute this code for educational and commercial purposes.

## 👤 Author

Created as a practice project for learning machine learning and neural network fundamentals.


---
Last Updated: April 10 2026
