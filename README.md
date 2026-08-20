# Twitter Spam Detection

A machine learning and NLP-based web application that detects whether a Twitter message is **Spam** or **Not Spam**.

## Project Overview

Spam messages on social media can contain unwanted advertisements, misleading information, malicious links, and other harmful content. This project uses **Natural Language Processing (NLP)** and **Machine Learning** techniques to automatically classify Twitter messages as spam or legitimate.

The project includes data preprocessing, exploratory data analysis, text feature extraction, machine learning classification, and a web-based prediction application.

## Features

* Text preprocessing and cleaning
* Natural Language Processing (NLP)
* TF-IDF feature extraction
* Spam and non-spam classification
* Machine learning model comparison
* Exploratory Data Analysis (EDA)
* Web application for real-time prediction
* User-friendly interface for entering a tweet and getting a prediction

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* NLTK
* Matplotlib
* Seaborn
* Flask
* HTML
* CSS
* Jupyter Notebook

## Machine Learning Models

The project explores multiple machine learning algorithms for spam classification:

* Naive Bayes
* Support Vector Machine (SVM)
* Logistic Regression

Among the evaluated models, **Support Vector Machine (SVM)** achieved the best performance with an accuracy of approximately **98.6%**.

## Project Workflow

```text
Twitter Dataset
      ↓
Data Cleaning
      ↓
Text Preprocessing
      ↓
Tokenization & Stopword Removal
      ↓
TF-IDF Feature Extraction
      ↓
Machine Learning Models
      ↓
Model Evaluation
      ↓
Spam / Not Spam Prediction
      ↓
Web Application
```

## Dataset

The project uses a Twitter spam dataset containing text messages labelled as spam or non-spam.

The dataset is preprocessed before model training by cleaning the text and converting it into numerical features using TF-IDF.

## Model Evaluation

The models were evaluated using metrics such as:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

The best-performing model achieved approximately **98.6% accuracy** on the evaluated dataset.

## Web Application

The project includes a web application where users can enter a Twitter message and receive a prediction indicating whether the message is:

**Spam** or **Not Spam**
<img width="317" height="230" alt="Screenshot 2026-08-19 192541" src="https://github.com/user-attachments/assets/5dadc695-56a5-4072-9b39-ca14adfe40f2" />
<img width="290" height="230" alt="Screenshot 2026-08-19 192650" src="https://github.com/user-attachments/assets/32e22f5f-f3bd-4c25-9bcd-dae2c81aa0c9" />


### Example

```text
Input:
Congratulations! You have won a free prize. Click the link now!

Prediction:
Spam
```

## Project Structure

```text
Twitter-Spam-Detection/
│
├── app.py
├── twitter app.py
├── converted_script.py
├── spam.csv
├── vectorizer.pkl
│
├── eda/
│   └── EDA notebooks and analysis
│
├── static/
│   └── CSS / static files
│
├── templates/
│   └── HTML templates
│
├── .gitignore
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/Pree-026/Twitter-Spam-Detection.git
```

Navigate to the project folder:

```bash
cd Twitter-Spam-Detection
```

Install the required Python packages:

```bash
pip install pandas numpy scikit-learn nltk matplotlib seaborn flask
```

## Run the Application

Run the Flask application:

```bash
python app.py
```

Then open the local URL displayed in the terminal in your web browser.

## Results

The project demonstrates that NLP-based machine learning models can effectively identify spam messages on social media.

The **SVM classifier achieved approximately 98.6% accuracy**, making it the best-performing model among the evaluated approaches.

## Future Improvements

* Deploy the application online
* Use larger and more recent Twitter datasets
* Experiment with deep learning models such as LSTM and BERT
* Add detection for malicious URLs
* Improve real-time spam detection
* Develop an interactive dashboard for model performance

