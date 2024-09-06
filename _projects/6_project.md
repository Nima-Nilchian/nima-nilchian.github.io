---
layout: page
title: Machine Learning Projects
description: Quera Machine Learning Bootcamp projects
img: assets/img/projects/ml.png
importance: 2
category: work
---

These projects are part of the projects of the Proffesional Project-Based Course in Machine Learning from quera. <br/> 
[Source Code](https://github.com/Nima-Nilchian/ML-Projects)

#### 1 - **Caceli:**

This project aimed to predict whether users would **cancel their tickets** using **machine learning models**. After **preprocessing** the data and performing **feature engineering**, two models were trained: a **Decision Tree Classifier** for basic predictions and an **ensemble approach** using **AdaBoost** and **Random Forest** to enhance accuracy. The models' performance was evaluated using **F1 scores**, **classification reports**, and **confusion matrices**, demonstrating the application of basic and ensemble machine learning models.

#### 2 - **Gender Detection:**

The project aimed to **predict** the **gender** of Twitter and Instagram users based on the data provided. Preprocessing involved **text normalization**, **tokenization**, and removing **stopwords** using tools like **Hazm** and **NLTK**. The cleaned text data was then vectorized using a **CountVectorizer**. For modeling, a **Multinomial Naive Bayes** classifier was trained on the processed data, with the dataset split into training and validation sets. The model's performance was evaluated using **F1 scores** and **classification reports**, highlighting its effectiveness in gender prediction.

#### 3 - **Text Categorization:**

This project focused on developing a **machine learning model** to predict the **categorical topic** of a document based on **text features** like titles, descriptions, and full content. Preprocessing steps included **text normalization**, **tokenization**, and handling **class imbalance** using **RandomOverSampler**. The modeling process involved building a pipeline with a **CountVectorizer**, **TfidfTransformer**, and a **Linear SVM classifier** to classify the documents. The model's performance was evaluated using the **weighted F1-score**, emphasizing its ability to accurately classify documents into the correct categories.

#### 4 - **Search Analysis:**

This project focuses on analyzing **user search data** from the MrBilit website to gain insights and answer specific analytical questions. The purpose is to equip with **data analysis** skills and a deeper understanding of the dataset and face their challenges.

1. **Service Popularity**: The first task involved calculating the **percentage popularity** of different services like buses, planes, taxis, etc., and visualizing this data using a **pie chart**.

2. **Most Searched Cities**: The next step was to filter out non-hotel searches and identify the **top 20 most-searched cities**, which were then visualized using a **histogram**.

3. **Top Provinces**: The analysis then focused on ranking **provinces** based on the number of searches for their cities, ignoring hotel services. A list of the top 15 provinces was created using the Iranian cities dataset for accurate mapping of city names.

4. **Population Analysis**: Finally, the project compared the **most searched cities** with the **census data from 2016**, identifying cities with populations over 500,000 that were not among the top 20 most-searched cities.

This project demonstrates the importance of **data analysis** in understanding **user behavior** and preparing for more complex **machine learning tasks**.

#### 5 - **Auto Suggest:**

This project focuses on building a smart **NLP** system that predicts and suggests possible completions of user input in real-time, similar to what users experience when typing in search fields on websites. Unlike traditional suggestion systems that only offer phrases beginning with the typed string, this model is designed to be **more intelligent** and handle errors such as **typos, language mismatches**, and **variations in place names** (e.g., “Zahedan” instead of “زاهدان”).

The system utilizes **rule-based** and **probability-based** methods to make suggestions. The rule-based method checks for exact matches, substrings, and entries that are within **one character distance** of the user’s input. The probability-based method predicts the most likely outcomes based on **likelihood estimates** and **vocabulary size**.

This flexible **auto-complete model** can handle **Persian and English text**, detect typos, and offer suggestions based on user input, improving the user experience in text-based search systems.

An example of the system is as below:<br/>
```auto_suggest.suggest('ت')```: <br/>
```['تهران', 'تبریز', 'تهران - پایانه جنوب', 'تهران - پایانه غرب', 'تنکابن']```