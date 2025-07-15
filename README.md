# 🏡 ML_Project – Bangalore House Price Predictor

This project is a **Supervised Machine Learning** application that predicts **house prices in Bangalore** using **Linear Regression**, **Lasso**, and **Ridge Regression** models.

📈 It achieves a prediction **accuracy of 93.1%** based on features like:
- 📍 Location (in Bangalore)
- 🛏️ Number of BHKs
- 🚿 Number of Bathrooms
- 📐 Total Area (in Sq. ft)

---

## 🌐 Live Website

👉 **Try the app live here**:  
🔗 [Bangalore House Price Predictor](https://bangalore-house-price-predictor-dibd.onrender.com/)

Enter your location, area, number of bedrooms, and bathrooms to get an instant price estimate!

---

## 📊 Dataset Used

- **Source:** [Kaggle – Bengaluru House Price Data](https://www.kaggle.com/datasets/amitabhajoy/bengaluru-house-price-data)
- Contains location-wise housing data with price, area, and configuration

---

## 📌 Machine Learning Approach

### ✅ Supervised Learning
Supervised ML is a technique where the model is trained on input data (features) **with known output labels** (target values).  
The model **learns the mapping** between input and output so that it can predict future values.

### 🔁 Regression
This is a **type of supervised learning** where the predicted output is a **continuous numeric value** (e.g., price).  
We’ve used three regression models:
- **Linear Regression**
- **Lasso Regression** (L1 Regularization)
- **Ridge Regression** (L2 Regularization)

---

## 🧠 Key Features of the Project

- 🧹 Data Cleaning: Handles missing values, inconsistent location names
- 🧠 Feature Engineering: Converts location into numerical features using one-hot encoding
- 🏷️ Model Training: Trains and compares multiple regression algorithms
- 📊 Performance Evaluation: Accuracy and error metrics used to validate performance
- 🌍 Web Deployment: Integrated with Flask and deployed using **Render**

---

## 🛠️ Tech Stack

- **Language:** Python
- **Libraries:** `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `Flask`
- **Deployment:** Render (for live website)

---

## 📁 Project Structure

```
ML_Project/
│
├── app/                     # Flask app folder
│   ├── templates/           # HTML templates for the web app
│   │   └── index.html       # Frontend for user input and result display
│   └── app.py               # Flask backend logic and routing
│
├── model/                   # Trained model pickle files (.pkl)
│
├── notebooks/               # Jupyter notebooks for EDA and model training
│
├── Bangalore_House_Data.csv # Raw dataset used for model training
│
├── requirements.txt         # Python dependencies
│
└── README.md                # Project overview and instructions
```

---

## 📌 Results

- Achieved **93.1% accuracy** on the test data.
- User-friendly **web interface** for real-time predictions.

---

## 📦 Future Improvements

- Use ensemble models like **Random Forest** or **XGBoost**
- Add more features like year built, proximity to tech parks, etc.
- Enhance UI/UX and make it mobile responsive

---

## 👩‍💻 Author

Built with 💙 by **Adrika Mandal**  
📍 [LinkedIn](https://www.linkedin.com/in/adrika-mandal-753226246/)  
💬 Always open to feedback, suggestions, and collaborations!

---

⭐ *If you found this useful, don’t forget to star the repo!*
