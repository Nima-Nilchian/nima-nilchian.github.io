---
layout: page
title: Modern Information Retrieval (2)
description: Machine Learning and Deep Learning approach to information retrieval
img: assets/img/projects/nn.jpg
importance: 1
category: work
---

This project aims to practise retrieval systems based on machine learning using classification and clustering.
[Source Code](https://github.com/Nima-Nilchian/IR-Phase2)

### 1 - EDA of dataset:

we start with first Preprocessing document, normalizing, removing stopwords, removing multi-label documents and splitting the document to train and test.

### 2 - Naive Bayes Classifier

In this project, a document classification system was developed using the **Naive Bayes** algorithm, focusing on categorizing documents into one of three predefined classes. The process began with **feature extraction**, where the word count vectors were created for each document, transforming textual data into a numerical format that the classifier could process. I implemented the Naive Bayes classifier from *scratch*, starting with the calculation of *prior probabilities* for each class based on their frequency in the dataset. I then estimated the *likelihood* of each word in the vocabulary occurring within documents from each class, using these likelihoods to inform the classification process. The classifier combined these likelihoods with the prior probabilities to compute the posterior probabilities for each class given a document, ultimately assigning the document to the class with the highest posterior probability. To ensure comprehensive coverage, a balanced subset of the dataset was selected that included documents from all three classes.

In the **evaluation** phase of the trained Naive Bayes model, I assessed its performance using several key metrics: *precision, recall, F1 score in both macro and micro settings, and overall accuracy.*

Furthermore, an **ROC curve** was generated for this non-binary classification task, also for a more visual analysis, the **confusion matrix** was constructed, which shows the count of correct and incorrect predictions per class, without using sklearn and computed manually and visualized the results using matplotlib and seaborn.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/projects/roc.png" title="example image" class="img-fluid rounded z-depth-1" %}
    <div class="caption">The ROC-Curve (Orange: class0, Green: class1,  Blue: class2)</div>
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/projects/confusion.png" title="example image" class="img-fluid rounded z-depth-1" %}
    <div class="caption">The Confusion Matrix</div>
    </div>
</div>


### 3 - Classification with Neural Networks:

In this task, a **neural network** was built to classify scientific articles based on their *abstracts* and *titles*. I used **FastText** to generate *100-dimensional embeddings* for each word in the articles. These embeddings were then averaged using **TF-IDF** weights to create a final vector representation for each article. The neural network was trained on these embeddings, with the labels converted into numerical form. the model’s performance was monitored using **training** and **validation losses**, ensuring accurate prediction of article topics. The loss is visualized through learning curves, providing insights into how the model was learning over time.

<div class="row justify-content-center">
    <div class="col-sm-8 col-md-6 mt-3">
        {% include figure.liquid loading="eager" path="assets/img/projects/loss_curv.png" title="example image" class="img-fluid rounded z-depth-1" %}
        <div class="caption">The Loss Curve</div>
    </div>
</div>

The neural network model was evaluated using several metrics to assess its accuracy and effectiveness in classifying scientific articles. Below is a summary of the evaluation results:

<table border="1" cellpadding="5" cellspacing="0">
  <thead>
    <tr>
      <th><strong>Metric</strong></th>
      <th><strong>Value</strong></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Accuracy</strong></td>
      <td>0.9522</td>
    </tr>
    <tr>
      <td><strong>Test Loss</strong></td>
      <td>0.0201</td>
    </tr>
    <tr>
      <td><strong>F1 Score (Macro)</strong></td>
      <td>0.9521</td>
    </tr>
    <tr>
      <td><strong>F1 Score (Micro)</strong></td>
      <td>0.9522</td>
    </tr>
    <tr>
      <td><strong>Precision (Macro)</strong></td>
      <td>0.9522</td>
    </tr>
    <tr>
      <td><strong>Recall (Macro)</strong></td>
      <td>0.9527</td>
    </tr>
    <tr>
      <td><strong>Precision (Micro)</strong></td>
      <td>0.9522</td>
    </tr>
    <tr>
      <td><strong>Recall (Micro)</strong></td>
      <td>0.9522</td>
    </tr>
  </tbody>
</table>

<br/>

### 4 - Classification with Language Models:

In this task, a **classification model** was built using the well-known **BERT** model from the **Transformers** library.

##### **Steps Involved:**

1. **Loading the Model and Tokenizer:**  
   We began by loading the pre-trained **BERT** model and its corresponding **tokenizer** using the **Transformers** library to preprocess the text data.

2. **Fine-Tuning the Model:**  
   Using the dataset from previous tasks, we performed **fine-tuning** on the BERT model. The *Trainer* API from the *Transformers* library facilitated this process, providing a streamlined interface for training and evaluation.

3. **Freezing Model Weights:**  
   In one experiment, we **froze the weights** of the BERT model and only trained the **classification head**. This approach allowed us to evaluate the impact of updating the pre-trained weights on model performance.

4. **Evaluating:**  
    Then both models were evaluated using the test dataset. Based on the outputs, the relevant performance metrics were computed to assess the effectiveness of each model.

Here is the evaluation statistics of the two models using the test dataset, presented in table format:

<table border="1" cellpadding="5" cellspacing="0">
  <thead>
    <tr>
      <th><strong>Metric</strong></th>
      <th><strong>Model 1</strong></th>
      <th><strong>Model 2</strong></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Accuracy</strong></td>
      <td>0.8755</td>
      <td>0.6675</td>
    </tr>
    <tr>
      <td><strong>F1-Macro</strong></td>
      <td>0.6021</td>
      <td>0.4458</td>
    </tr>
    <tr>
      <td><strong>F1-Micro</strong></td>
      <td>0.8755</td>
      <td>0.6675</td>
    </tr>
    <tr>
      <td><strong>Precision (Macro)</strong></td>
      <td>0.7464</td>
      <td>0.4399</td>
    </tr>
    <tr>
      <td><strong>Recall (Macro)</strong></td>
      <td>0.6164</td>
      <td>0.4574</td>
    </tr>
    <tr>
      <td><strong>Precision (Micro)</strong></td>
      <td>0.8755</td>
      <td>0.6675</td>
    </tr>
    <tr>
      <td><strong>Recall (Micro)</strong></td>
      <td>0.8755</td>
      <td>0.6675</td>
    </tr>
  </tbody>
</table>

<br/>

### 5 - Clustering the documents:

In this task, the focused was on **document clustering** using **embedding vectors** and various clustering algorithms. 

#### Steps Involved:

1. **Extracting Document Embeddings:**  
   Instead of using basic methods like Bag of Words, a **transformer-based language models** (BERT) were utilized to generate high-quality **embeddings** for each document. We implemented the `extract_embedding` function to take a list of documents as input and return a list of corresponding embedding vectors. Techniques such as using the CLS token embedding from BERT or averaging word embeddings (weighted or unweighted) were employed.

2. **Dimensionality Reduction for Visualization:**  
   To visualize the clustering results, the dimensionality of the embedding vectors were reduced to two dimensions using **T-SNE**. Note that dimensionality reduction is only used for visualization purposes; all clustering steps are performed on the original, high-dimensional vectors.

3. **Clustering with KMeans and Hierarchical Clustering:**  
   **KMeans** and **Hierarchical Clustering** algorithms were used to create clusters from the embedding vectors. These clustering techniques help group similar documents based on their embeddings.

4. **Plotting the Clusters:**  
   a Function was implemented to visualize the clustering results. This function takes 2D reduced vectors and cluster assignments as input and generates a 2D scatter plot where each point represents a document, colored according to its cluster. This visualization helps in understanding the distribution and separation of clusters.

<div class="row justify-content-center">
    <div class="col-sm-8 col-md-6 mt-3">
        {% include figure.liquid loading="eager" path="assets/img/projects/tsne.png" title="example image" class="img-fluid rounded z-depth-1" %}
        <div class="caption">Dimentionality Reduction using T-SNE</div>
    </div>
</div>

#### **K-Means:**

In this task, the **K-means algorithm** was implemented from scratch to group documents into clusters based on their embeddings. First, cluster centroids were computed and then assigned each document to the nearest centroid for various values of **k** (the number of clusters). Next, the clustering results were analyzed by determining the topics of each cluster using representative documents. Also a **`silhouette analysis`** was conducted to evaluate the quality of the clusters by plotting the silhouette score for different k values, helping to select the optimal k. Additionally, the **`purity`** of the clusters were calculated using labeled data to assess the effectiveness of the clustering.

Clustering Results for K Values from 2 to 7:

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/projects/means2.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/projects/means3.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/projects/means4.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/projects/means5.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/projects/means6.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/projects/means7.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
Visual representation of document clusters using K-means with varying values of K, highlighting how clusters change as K increases.
</div>

Silhouette Analysis for Cluster Quality and Purity Scores of Document Clusters:
<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/projects/silhouette.png" title="example image" class="img-fluid rounded z-depth-1" %}
    <div class="caption">A silhouette plot showing the silhouette scores for different values of K</div>
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/projects/purity.png" title="example image" class="img-fluid rounded z-depth-1" %}
        <div class="caption">Visualization of purity scores for various K values</div>
    </div>
</div>


#### **Hierarchical Clustering Method:**

Hierarchical clustering is a technique used in machine learning for grouping data into a tree-like structure of clusters. In this task, hierarchical clustering was performed on the dataset. After applying the clustering algorithm, the resulting clusters was visualized using Matplotlib, allowing you to see the relationships and structure among the data points based on their similarities.

<div class="row justify-content-center">
    <div class="col-sm col-md mt-3">
        {% include figure.liquid loading="eager" path="assets/img/projects/hierarchical.png" title="example image" class="img-fluid rounded z-depth-1" %}
        <div class="caption">the hierarchical clustering of data, showcasing the tree-like structure of clusters and their relationships</div>
    </div>
</div>

In conclusion, this project comprehensively explored various methods for analyzing and classifying scientific documents, beginning with an **Exploratory Data Analysis (EDA)** to understand the dataset's structure and characteristics. The **Naive Bayes Classifier** was implemented as a baseline model, followed by advanced **classification techniques using Neural Networks** and **language models**. While the use of language models like BERT was intended to enhance classification performance, it resulted in decreased F1 scores due to insufficient data and the complexity of the models, which negatively affected the classification performance. Subsequently, we employed **clustering techniques** such as **K-means** and **Hierarchical Clustering** to group similar documents, utilizing dimensionality reduction for effective visualization of the clusters enhancing our understanding of the document relationships and classifications.