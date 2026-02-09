# Sonar Rock vs Mine Classification using Logistic Regression

This project implements a **Logistic Regression** machine learning model to classify underwater sonar signals as either **Rock (R)** or **Mine (M)**.  
The model is trained on the **Sonar Dataset** and uses signal frequency data to make predictions.

---

## 📌 Project Overview

Sonar systems emit sound waves underwater and analyze the reflected signals.  
This project uses those reflected signal features to determine whether the object detected is a **rock** or a **mine**.

- **Algorithm Used:** Logistic Regression
- **Type:** Binary Classification
- **Tools:** Python, NumPy, Pandas, Scikit-learn

---

## 📊 Dataset Information

- The dataset contains **208 samples**
- Each sample has **60 numerical features**
- The **61st column** is the target label:
  - `R` → Rock
  - `M` → Mine

---

## 🛠️ Technologies & Libraries Used

- Python
- NumPy
- Pandas
- Scikit-learn

---

## 🔄 Workflow

1. Load and explore the dataset
2. Separate features and labels
3. Split data into training and testing sets
4. Train Logistic Regression model
5. Evaluate model accuracy
6. Predict output for new input data

---

## 📈 Model Performance

- **Training Accuracy:** ~90%
- **Testing Accuracy:** ~85–90%  
(Accuracy may vary slightly due to random state)

---

## 🧪 Example Prediction

The model can predict whether an object is a **Rock** or **Mine** using new sonar input data:

```python
input_data = (0.0258, 0.0433, 0.0547, ..., 0.0050)
prediction = model.predict(input_data_reshaped)

git clone https://github.com/your-username/sonar-rock-mine-classification.git

pip install numpy pandas scikit-learn

python sonar_prediction.py
