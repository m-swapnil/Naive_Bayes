# **Spam SMS Detection Using Naïve Bayes & Flask**  

## **Overview**  
This project is a **Spam SMS Detection System** that uses **Multinomial Naïve Bayes (MNB)** to classify SMS messages as either **spam** or **ham (not spam)**. The machine learning model is trained using **SMOTE (Synthetic Minority Over-sampling Technique)** to handle class imbalance.

A **Flask web application** is also built to allow users to upload an Excel file containing SMS messages and get predictions. The results are stored in a **PostgreSQL database**.

---

## **Project Workflow**  
### **1. Business & Data Understanding**  
- **Objective:** Maximize spam detection while minimizing manual detection rules.  
- **Success Criteria:**  
  - Reduce customer churn by 12%.  
  - Achieve over **80% accuracy** in spam detection.  
  - Cost savings of **$120K - $130K** due to reduced churn.  
- **Dataset:** SMS spam dataset from a telecom company, containing **5,559 messages** labeled as "spam" or "ham".  

### **2. Machine Learning Model**  
- **Data Preprocessing:**  
  - **CountVectorizer** for feature extraction (Bag of Words approach).  
  - **SMOTE** for handling class imbalance.  
- **Model Training & Evaluation:**  
  - Multinomial Naïve Bayes (MNB) classifier.  
  - Performance measured using **accuracy, sensitivity, specificity, and precision**.  
- **Hyperparameter Tuning:**  
  - Laplace smoothing (**alpha = 5**) applied to improve predictions.  
- **Model Saving:**  
  - The trained model is saved using `joblib` for deployment.  

### **3. Web Application (Flask)**  
- Users can upload an **Excel file** with SMS messages.  
- The Flask app loads the trained model and predicts whether each message is **spam or ham**.  
- Results are stored in a **PostgreSQL database** (`sms_db`).  
- Predictions are displayed in an HTML table.  

---

## **Tech Stack Used**  
| Component | Technology Used |
|-----------|----------------|
| **Programming Language** | Python  |
| **Machine Learning** | Scikit-learn, Imbalanced-learn (SMOTE) |
| **Web Framework** | Flask |
| **Database** | PostgreSQL |
| **Frontend** | HTML, CSS |
| **Data Handling** | Pandas, NumPy |
| **Deployment** | Joblib for model persistence |

---

## **Project Structure**  
- 📂 **Spam-Detection**  
  - 📄 `app.py` - Flask application  
  - 📄 `model_training.py` - Machine learning model training  
  - 📄 `sms_raw_NB.csv` - Dataset file  
  - 📄 `processed1` - Trained Naïve Bayes model  
  - 📄 `requirements.txt` - Required dependencies  
  - 📄 `README.md` - Project documentation  
  - 📂 **templates/**  
    - 📄 `index.html` - Upload page  
    - 📄 `new.html` - Results page  

---

## **Installation & Setup**  

### **1. Clone the Repository**  
```sh
git clone https://github.com/your-username/spam-sms-detection.git
cd spam-sms-detection
```

### 2. Install Required Packages
```sh
Create a virtual environment and install dependencies:
python -m venv venv
source venv/bin/activate  # For Mac/Linux
venv\Scripts\activate     # For Windows
pip install -r requirements.txt
```
### 3. Set Up PostgreSQL Database
**Install PostgreSQL and create a database named sms_db.** 

**Update the connection string in app.py and model_training.py:***
```sh
conn_string = 'postgresql+psycopg2://postgres:your_password@localhost:5432/sms_db'
```
**Run the following command to create the required table:**
```sh
CREATE TABLE sms_raw (
    type TEXT,
    text TEXT,
    spam INT
);
```
### 4. Train the Model
**Run the following command to train and save the model:**
```
python model_training.py
```
### 5. Start the Flask Web App
```
python app.py
```
### Usage
**Train the Model: Run model_training.py to preprocess data, train the classifier, and save the best model.**

**Start the Web App: Run app.py to launch the Flask application.**

**Upload SMS Data: Users can upload an Excel file for spam detection.**

**View Predictions: The results are stored in PostgreSQL and displayed in the web app.**

## **Results & Performance**  

| **Metric**            | **Value** |
|----------------------|----------|
| **Accuracy**        | 85%      |
| **Precision**       | 88%      |
| **Recall (Sensitivity)** | 82%  |
| **Specificity**     | 87%      |

- The model achieves an **85% accuracy** in detecting spam messages.  
- Using **SMOTE**, we effectively handle class imbalance.  
- Applying **Laplace smoothing (alpha = 5)** improves classification performance.  

---

## **Future Improvements**  

✅ Improve feature extraction using **TF-IDF** instead of CountVectorizer.  
✅ Deploy the model using **Docker & AWS** for scalability.  
✅ Implement an **API endpoint** for real-time SMS spam detection.  

