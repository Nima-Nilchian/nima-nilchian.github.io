---
layout: page
title: Machine Learning and Deep Learning Approach To Information Retrieval (2)
description: Second project of the Modern Information Retrieval course
img: assets/img/projects/nn.jpg
importance: 1
category: work
---

This project aims to practise Retrieval systems based on machine learning using classification and clustering.

### 1 - EDA of dataset:

we start with first Preprocessing document, normalizing, removing stopwords, removing multi-label documents and splitting the document to train and test.

### 2 - Naive Bayes Classifier

In this project, I developed a document classification system using the Naive Bayes algorithm from scratch, focusing on categorizing documents into one of three predefined classes. The process began with feature extraction, where I created word count vectors for each document, transforming textual data into a numerical format that the classifier could process. I implemented the Naive Bayes classifier from scratch, starting with the calculation of prior probabilities for each class based on their frequency in the dataset. I then estimated the likelihood of each word in the vocabulary occurring within documents from each class, using these likelihoods to inform the classification process. The classifier combined these likelihoods with the prior probabilities to compute the posterior probabilities for each class given a document, ultimately assigning the document to the class with the highest posterior probability. To ensure comprehensive coverage, I selected a balanced subset of the dataset that included documents from all three classes.

In the evaluation phase of the trained Naive Bayes model, I assessed its performance using several key metrics: precision, recall, F1 score in both macro and micro settings, and overall accuracy.

Furthermore, I generated the ROC curve for this non-binary classification task, extending the typical binary ROC analysis to handle multiple classes.

For a more visual analysis, I also constructed the confusion matrix, which shows the count of correct and incorrect predictions per class, without using sklearn and computed manually and visualized the results using matplotlib and seaborn.

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

In this task, I built a **neural network** to classify scientific articles based on their **abstracts** and **titles**. I used **FastText** to generate **100-dimensional embeddings** for each word in the articles. These embeddings were then averaged using **TF-IDF** weights to create a final vector representation for each article. The neural network was trained on these embeddings, with the **labels** converted into numerical form. I monitored the model’s performance using **training** and **validation losses**, ensuring accurate prediction of article topics. The loss is visualized through learning curves, providing insights into how the model was learning over time.

<div class="row justify-content-center">
    <div class="col-sm-8 col-md-6 mt-3">
        {% include figure.liquid loading="eager" path="assets/img/projects/loss_curv.png" title="example image" class="img-fluid rounded z-depth-1" %}
        <div class="caption">The Loss Curve</div>
    </div>
</div>

The neural network model was evaluated using several metrics to assess its accuracy and effectiveness in classifying scientific articles. Below is a summary of the evaluation results:

| **Metric**               | **Value**             |
|--------------------------|-----------------------|
| **Accuracy**             | 0.9522                |
| **Test Loss**            | 0.0201                |
| **F1 Score (Macro)**     | 0.9521                |
| **F1 Score (Micro)**     | 0.9522                |
| **Precision (Macro)**    | 0.9522                |
| **Recall (Macro)**       | 0.9527                |
| **Precision (Micro)**    | 0.9522                |
| **Recall (Micro)**       | 0.9522                |

<br/>

### 4 - Classification with Language Models:

In this task, we built a **classification model** using the well-known **BERT** model from the **Transformers** library.

###### **Steps Involved:**

1. **Loading the Model and Tokenizer:**  
   We began by loading the pre-trained **BERT** model and its corresponding **tokenizer** using the **Transformers** library to preprocess the text data.

2. **Fine-Tuning the Model:**  
   Using the dataset from previous tasks, we performed **fine-tuning** on the BERT model. The **Trainer** API from the **Transformers** library facilitated this process, providing a streamlined interface for training and evaluation.

3. **Freezing Model Weights:**  
   In one experiment, we **froze the weights** of the BERT model and only trained the **classification head**. This approach allowed us to evaluate the impact of updating the pre-trained weights on model performance.

4. **Evaluating :**  
    We then evaluated both models using the test dataset. Based on the outputs, we computed relevant performance metrics to assess the effectiveness of each model.

Here is the evaluation statistics of the two models using the test dataset, presented in table format:

| **Metric**                      | **Model 1**              | **Model 2**              |
|---------------------------------|--------------------------|--------------------------|
| **Accuracy**                    | 0.8755                   | 0.6675                   |
| **F1-Macro**                    | 0.6021                   | 0.4458                   |
| **F1-Micro**                    | 0.8755                   | 0.6675                   |
| **Precision (Macro)**           | 0.7464                   | 0.4399                   |
| **Recall (Macro)**              | 0.6164                   | 0.4574                   |
| **Precision (Micro)**           | 0.8755                   | 0.6675                   |
| **Recall (Micro)**              | 0.8755                   | 0.6675                   |

<br/>

### 5 - Clustering the documents:

In this task, we focused on **document clustering** using **embedding vectors** and various clustering algorithms. 

#### Steps Involved:

1. **Extracting Document Embeddings:**  
   Instead of using basic methods like Bag of Words, we utilized **transformer-based language models** to generate high-quality embeddings for each document. We implemented the `extract_embedding` function to take a list of documents as input and return a list of corresponding embedding vectors. Techniques such as using the CLS token embedding from BERT or averaging word embeddings (weighted or unweighted) were employed.

2. **Dimensionality Reduction for Visualization:**  
   To visualize the clustering results, we reduced the dimensionality of the embedding vectors to two dimensions using **T-SNE**. We implemented the `convert_to_2d_tsne` function to convert high-dimensional embeddings into 2D vectors for plotting. Note that dimensionality reduction is only used for visualization purposes; all clustering steps are performed on the original, high-dimensional vectors.

3. **Clustering with KMeans and Hierarchical Clustering:**  
   We used **KMeans** and **Hierarchical Clustering** algorithms to create clusters from the embedding vectors. These clustering techniques help group similar documents based on their embeddings.

4. **Plotting the Clusters:**  
   We implemented the `plot_docs` function to visualize the clustering results. This function takes 2D reduced vectors and cluster assignments as input and generates a 2D scatter plot where each point represents a document, colored according to its cluster. This visualization helps in understanding the distribution and separation of clusters.

<div class="row justify-content-center">
    <div class="col-sm-8 col-md-6 mt-3">
        {% include figure.liquid loading="eager" path="assets/img/projects/tsne.png" title="example image" class="img-fluid rounded z-depth-1" %}
        <div class="caption">Dimentionality Reduction using T-SNE</div>
    </div>
</div>

#### **K-Means:**

In the **K-means clustering** project, you will implement the K-means algorithm from scratch to group documents into clusters based on their embeddings. First, you will create the algorithm to compute cluster centroids and assign each document to the nearest centroid for various values of **k** (the number of clusters). Next, you'll analyze the clustering results by determining the topics of each cluster using representative documents. You will also conduct a **`silhouette analysis`** to evaluate the quality of the clusters by plotting the silhouette score for different k values, helping you select the optimal k. Additionally, you will calculate and visualize the **`purity`** of the clusters using labeled data to assess the effectiveness of your clustering. Finally, all steps will be encapsulated in a function, **`cluster_kmeans`**, which takes embedding vectors as input and outputs the cluster centroids and assignments.

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

Hierarchical clustering is a technique used in machine learning for grouping data into a tree-like structure of clusters. In this task, you will utilize libraries such as SciPy or other Python libraries to perform hierarchical clustering on your dataset. After applying the clustering algorithm, you will visualize the resulting clusters using Matplotlib, allowing you to see the relationships and structure among the data points based on their similarities.

<div class="row justify-content-center">
    <div class="col-sm col-md mt-3">
        {% include figure.liquid loading="eager" path="assets/img/projects/hierarchical.png" title="example image" class="img-fluid rounded z-depth-1" %}
        <div class="caption">the hierarchical clustering of data, showcasing the tree-like structure of clusters and their relationships</div>
    </div>
</div>

In conclusion, this project comprehensively explored various methods for analyzing and classifying scientific documents, beginning with an **Exploratory Data Analysis (EDA)** to understand the dataset's structure and characteristics. The **Naive Bayes Classifier** was implemented as a baseline model, followed by advanced **classification techniques using Neural Networks** and **language models**. While the use of language models like BERT was intended to enhance classification performance, it resulted in decreased F1 scores due to insufficient data and the complexity of the models, which negatively affected the classification performance. Subsequently, we employed **clustering techniques** such as **K-means** and **Hierarchical Clustering** to group similar documents, utilizing dimensionality reduction for effective visualization of the clusters enhancing our understanding of the document relationships and classifications.