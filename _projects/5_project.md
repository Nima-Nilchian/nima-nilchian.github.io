---
layout: page
title: Username Generator
description: translating persian usernames to equivalent english names.
img: assets/img/projects/transl.jpg
importance: 2
category: work
---

#### **Project Overview**

The goal of this project is to develop a model that can handle the transliteration task effectively, converting Persian names to appropriate Latin-script equivalents for use as usernames. The project is structured into several key steps to achieve this objective.

1. **Data Analysis**:
    - Initially, a thorough analysis of the dataset will be conducted using available tools. This includes creating various visualizations like histograms or word clouds, and interpreting these to understand the dataset better. The aim is to gain a deep understanding of the data before moving forward.
   
2. **Preprocessing**:
    - This involves cleaning the dataset to ensure that the data is suitable for model training. This step includes tasks like removing any unnecessary noise from the data, handling missing values, and normalizing the text.

3. **Feature Extraction**:
    - Key features will be extracted from the data to help the model learn effectively. This include identifying important linguistic patterns or phonetic cues that are essential for accurate transliteration.

4. **Modeling**:
    - The primary task involves implementing and training a **Seq2Seq** (Sequence-to-Sequence) model (**LSTM** and **GRU**) from scratch using **PyTorch**. The model needs to be carefully tuned to prevent overfitting or underfitting.

5. **Analysis of Results**:
    - Post-training, the results will be analyzed, including evaluating the model using metrics like the Word Error Rate (WER). Visualizations of training and test losses will be provided to illustrate the model's performance.

6. **Evaluation Metrics**:
    - The performance of the model will be evaluated using WER, with recommendations to use the "jiwer" library for accurate calculations and monitoring with Tensorboard.


The project is an excellent opportunity to delve deep into the challenges of natural language processing, specifically in transliteration, and offers a practical application that can be extended to real-world scenarios.

