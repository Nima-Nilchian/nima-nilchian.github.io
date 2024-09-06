---
layout: page
title: Keyword Extraction
description: Implementing and evaluating various keyword extraction methods in Persian Language
img: assets/img/projects/ke.jpg
importance: 3
category: work
related_publications: false
---

[Source Code](https://github.com/Nima-Nilchian/Keyword_extraction)

#### Keyword Extraction Methods
Several algorithms are implemented for keyword extraction, including:
*   Baseline TF-IDF
*   TF-IDF with N-grams
*   TF-IDF with Chunking
*   TextRank
*   SingleRank
*   TopicRank
*   PositionRank
*   MultipartiteRank
*   Yake
*   EmbedRank (Sentence Embedding and BERT Embedding)
*   KeyBERT

#### Data Preprocessing
- **Data Loading**: Text data is read from a dataset and processed.
- **Text Cleaning**: Involves normalization, stopword removal, stemming, lemmatization, and part-of-speech tagging.
- **Preprocessing Function**: Cleans and tokenizes text, applying or skipping steps like chunking or tagging based on configuration.

#### Evaluation and Metrics
- **F1-Score Calculation**: The models are evaluated based on F1-score, precision, and recall, with options for handling N-grams to improve accuracy.
- **Evaluation Function**: A method for comparing model output to reference keywords using exact matches or N-grams.

#### Model Implementations

- ##### **TF-IDF**: 
    A baseline TF-IDF approach is applied, with variations such as N-grams and chunking for enhanced keyword extraction.

    Evaluation result of TF-IDF variants:

    <table border="1" cellpadding="4" cellspacing="0">
    <tr>
        <td>Variants</td>
        <td>Without Ngram</td>
        <td>Without chunking</td>
        <td>With chunking</td>
        <td>with chunking and not exact match references</td>
    </tr>
    <tr>
        <th>F1-Score</th>
        <td>0.00</td>
        <td>0.1687</td>
        <td>0.1689</td>
        <td>0.2049</td>
    </tr>
    </table>
    <br/>

<div class="row">
<div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid loading="eager" path="assets/img/projects/tfidf-score.png" title="example image" class="img-fluid rounded z-depth-1" %}
<div class="caption">Comparing the effect of different numbers of extracted keywords on evaluation score</div>
</div>
</div>

- ##### **PerKe Algorithms**:
    The provided code is implementing various unsupervised keyword extraction algorithms using the perke library, which leverages graph-based methods to extract key phrases from texts. The code compares the performance of five keyphrase extraction algorithms:
    - **TopicRank**: A graph-based algorithm that groups similar words into clusters (topics) and extracts the most representative phrases.
    - **TextRank**: Based on Google's PageRank algorithm, it constructs a graph where words are vertices and edges are weighted based on co-occurrence.
    - **SingleRank**: Similar to TextRank but uses word co-occurrence within a fixed window size to weigh candidates.
    - **PositionRank**: Extends TextRank by giving higher importance to words that appear earlier in the text and in important syntactic positions.
    - **MultipartiteRank**: Builds a multipartite graph to represent different types of candidate phrases and captures long-range dependencies between words.

    <br/>
    <table border="1" cellpadding="4" cellspacing="0">
    <tr>
        <th>Algorithm</th>
        <th>Mean F1 Score</th>
        <th>F1 Score (Not-exact Match)</th>
    </tr>
    <tr>
        <td>Topic Rank</td>
        <td>0.121</td>
        <td>0.121</td>
    </tr>
    <tr>
        <td>Text Rank</td>
        <td>0.1392</td>
        <td>0.1631</td>
    </tr>
    <tr>
        <td>Single Rank</td>
        <td>0.0887</td>
        <td>0.094</td>
    </tr>
    <tr>
        <td>Position Rank</td>
        <td>0.0508</td>
        <td>0.0706</td>
    </tr>
    <tr>
        <td>Multipartite Rank</td>
        <td>0.1429</td>
        <td>0.1429</td>
    </tr>
    </table>
    <br/>

- ##### **YAKE Algorithm**:
   - Utilizes the YAKE keyword extraction method to generate candidates based on text, comparing scores for both raw and preprocessed text. 
   - Performance is evaluated using F1, recall, and precision scores, visualized using line plots for different numbers of keywords.
   - Multiple evaluations are performed, including exact keyword matches and n-gram-based evaluation, providing a thorough analysis.
    <br/>

    Evaluation result of YAKE Algorithm variants with 10 keywords extracted:

    <table border="1" cellpadding="4" cellspacing="0">
    <tr>
        <td>Variants</td>
        <td>raw text</td>
        <td>raw text ngrams-evaluation</td>
        <td>preprocessed text</td>
    </tr>
    <tr>
        <th>F1-Score</th>
        <td>0.116</td>
        <td>0.168</td>
        <td>0.040</td>
    </tr>
    </table>
    <br/>

<div class="row">
<div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid loading="eager" path="assets/img/projects/yake.png" title="example image" class="img-fluid rounded z-depth-1" %}
<div class="caption">Comparing the performance of different numbers of extracted keywords on evaluation score</div>
</div>
</div>

- ##### **Embedding Rank**:
    The Embedding Rank approach implement keyword extraction pipeline using the EmbedRank algorithm, leverages sentence embeddings (using models like Sent2Vec and BERT(`sharif-dal/dal-bert`)) and cosine similarity to rank keywords.

   - It processes text by **tokenizing** and **POS tagging** to extract **noun phrases** as potential keyphrase candidates.  
   - The candidates are converted into **vector embeddings**, utilizing either traditional **sentence embeddings** or **BERT**.  
   - **Cosine similarity** is calculated between the candidate phrases and the overall text.  
   - The **EmbedRank** algorithm selects the top keyphrases by balancing their similarity to the text and diversity among themselves.  
   - The performance is **evaluated** across different combinations of `beta` and keyword numbers, and results are visualized using a heatmap.

<div class="row">
<div class="col-sm-8 col-md-8 mt-3">
    {% include figure.liquid loading="eager" path="assets/img/projects/hpo.png" title="example image" class="img-fluid rounded z-depth-1" %}
<div class="caption">Hyperparameter tuning the number of keywords and the beta parameter for balancing textual and candidate similarity</div>
</div>
</div>

- ##### **KeyBeRT Embedding**:
   - The BERT-based method uses `KeyBERT` for keyword extraction from texts, evaluating different pre-trained models like `sharif-dal/dal-bert` to extract embeddings and key phrases.
   - The method compares raw and preprocessed texts, using F1 score to measure the performance of keyword extraction, particularly focusing on keyword similarity.

