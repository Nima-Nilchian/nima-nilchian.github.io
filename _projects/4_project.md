---
layout: page
title: Traditional Information Retrieval (1)
description: Implementing a document search and retrieval system that ranks documents based on a combination of vector space and probabilistic models
img: assets/img/projects/ir.png
importance: 1
category: work
---

This project is the first phase of three phases of the Modern Information Retrieval Course by Prof. Mahdie Soleymani Baghshah from Sharif University of Technology.

### EDA of dataset:

We start with first **Preprocessing document**, **normalizing**, **removing stopwords** by calculating **top frequency words**, **stemming**, **Lemmatizing**, and **Case folding**.

### Positional Index:

Constructing a **positional index** in **trie format** for searching the word and retrieving the **posting list**. Also making the index **dynamic** by adding and removing **document**.

### Index Storage and Compression:

Enhancing the system by adding the capability to **save and reload the index**. The storage has been implemented using three different methods: **no-compression**, **gamma-code**, and **variable-byte**. These compression methods have been implemented **manually** and optimized.

### Spelling Correction:

If the input query contains **spelling errors** or words not found in the dictionary, the query will be corrected before proceeding with the search. Starting by extracting **bigrams** from the word, then using the **Jaccard index** to find the 20 words with the most shared bigrams. Finally, by using the **minimum edit distance**, the best replacement of the word will be selected.

### Document Search and Retrieval:

Implementing a **document search and retrieval system** that processes **user queries** to search within the **titles** and **abstracts** of indexed documents. The system calculates a **final score** for each document by combining **weighted scores** from title and abstract searches, allowing customization through a **user-defined weight parameter**. Two **search methods** were implemented: a **vector space model** using **TF-IDF** (**ltn-lnn**, **ltc-lnc** and **lnc-ltn**) and a **probabilistic model** (**Okapi BM25**). The system returns documents ranked by their final scores.

### System Performance Evaluation:

At the end the **performance** of a **document retrieval system** will be evaluated using a set of sample queries and their corresponding actual results provided in a validation file. Various **evaluation metrics** like **precision**, **recall**, **F1**, **MAP**, **NDCG** and **MRR** have been implemented from scratch and all the above search methods have been compared.

the Evaluation statistics is as below:

<table border="1" cellpadding="5" cellspacing="0">
    <tr>
        <th>Metric</th>
        <th>ltn-lnn</th>
        <th>ltc-lnc</th>
        <th>lnc-ltn</th>
        <th>Okapi BM25</th>
    </tr>
    <tr>
        <td>Precision</td>
        <td>0.58</td>
        <td>0.63</td>
        <td>0.67</td>
        <td>0.62</td>
    </tr>
    <tr>
        <td>Recall</td>
        <td>0.70</td>
        <td>0.63</td>
        <td>0.67</td>
        <td>0.62</td>
    </tr>
    <tr>
        <td>F1 Score</td>
        <td>0.64</td>
        <td>0.63</td>
        <td>0.67</td>
        <td>0.62</td>
    </tr>
    <tr>
        <td>MAP</td>
        <td>0.72</td>
        <td>0.77</td>
        <td>0.85</td>
        <td>0.76</td>
    </tr>
    <tr>
        <td>NDCG</td>
        <td>0.61</td>
        <td>0.61</td>
        <td>0.65</td>
        <td>0.53</td>
    </tr>
    <tr>
        <td>MRR</td>
        <td>0.26</td>
        <td>0.32</td>
        <td>0.36</td>
        <td>0.31</td>
    </tr>
</table>
