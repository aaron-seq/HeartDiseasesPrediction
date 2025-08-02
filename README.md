# Heart Disease Prediction using Advanced Neural Networks 🫀

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/) [![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/) [![Keras](https://img.shields.io/badge/Keras-2.x-red.svg)](https://keras.io/) [![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-green.svg)](https://scikit-learn.org/) [![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A comprehensive machine learning project for predicting heart disease using a state-of-the-art neural network architecture. This repository contains a Jupyter Notebook that walks through the entire pipeline, from exploratory data analysis with rich visualizations to building an interactive prediction UI.

## 📊 Project Overview

Heart disease is a leading cause of death globally, making early and accurate prediction crucial for preventive healthcare. This project applies advanced machine learning techniques to the UCI Cleveland Heart Disease dataset, achieving high accuracy with a modern neural network architecture. The notebook is designed to be both educational and functional, providing clear insights into the data and the model's performance.

### Key Features:

* **In-Depth Exploratory Data Analysis (EDA)**: Rich visualizations to understand feature distributions and relationships.
* **Robust Preprocessing Pipeline**: A `ColumnTransformer` to handle scaling and imputation efficiently.
* **Modern Neural Network Architecture**: A deep neural network with Batch Normalization and Dropout for improved performance and reduced overfitting.
* **Interactive Prediction UI**: A user-friendly interface built with `ipywidgets` to perform real-time predictions directly within the notebook.
* **Explainable AI**: Integration of SHAP (SHapley Additive exPlanations) for model interpretability.
* **Comprehensive Evaluation**: Detailed model evaluation with performance graphs, including a confusion matrix and ROC curve.

## 📈 Visualizations & Insights

The notebook includes a variety of visualizations to explore the dataset and evaluate the model.

| Correlation Matrix                                                                                                             | Training History                                                                                                               |
| :-----------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------: |
| ![Correlation Matrix](https://placehold.co/600x400/2d3748/ffffff?text=Correlation+Heatmap)                                        | ![Training History](https://placehold.co/600x400/2d3748/ffffff?text=Model+Training+History)                                     |
| *A heatmap showing the correlation between different features.* | *A plot of the model's accuracy and loss over epochs.* |
| **Confusion Matrix** | **ROC Curve** |
| ![Confusion Matrix](https://placehold.co/600x400/2d3748/ffffff?text=Confusion+Matrix)                                            | ![ROC Curve](https://placehold.co/600x400/2d3748/ffffff?text=ROC+Curve)                                                          |
| *A matrix showing the model's performance on the test set.* | *The Receiver Operating Characteristic curve, illustrating the model's diagnostic ability.* |
| **SHAP Summary Plot** | **Interactive UI** |
| ![SHAP Summary Plot](https://placehold.co/600x400/2d3748/ffffff?text=SHAP+Summary+Plot)                                          | ![Interactive UI](https://placehold.co/600x400/2d3748/ffffff?text=Interactive+Prediction+UI)                                   |
| *A SHAP plot showing the impact of each feature on the model's output.* | *An interactive UI for real-time heart disease prediction.* |

## 🎯 Performance

The enhanced model demonstrates strong predictive performance on the binary classification task (Heart Disease vs. No Heart Disease).

| Metric          | Score   |
| --------------- | ------- |
| **Accuracy** | ~87%    |
| **ROC-AUC Score** | ~0.92   |

## 🚀 Getting Started

### Prerequisites

* Python 3.8+
* Jupyter Notebook or JupyterLab
* The libraries listed in `requirements.txt`.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/aaronseq12/HeartDiseasesPrediction.git](https://github.com/aaronseq12/HeartDiseasesPrediction.git)
    cd HeartDiseasesPrediction
    ```

2.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

### Usage

Open and run the `Interactive_Heart_Disease_Prediction.ipynb` notebook in Jupyter:

```bash
jupyter notebook Interactive_Heart_Disease_Prediction.ipynb
```

The notebook is self-contained and will guide you through all the steps of the project.

## 💡 Interactive Prediction UI

The notebook features an interactive UI that allows you to input patient data using sliders and dropdowns to get a real-time prediction from the trained model. This makes it easy to test different scenarios and understand the model's behavior.

## 🧠 Explainable AI

To ensure the model is not a "black box," SHAP (SHapley Additive exPlanations) is used to provide insights into the model's predictions. The SHAP summary plot helps visualize which features are most influential in predicting heart disease.

## 🤝 Contributing

Contributions are welcome! If you have suggestions for improvements or find any issues, please feel free to open an issue or submit a pull request.

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
