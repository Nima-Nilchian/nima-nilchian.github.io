---
layout: page
title: Web Information Retrieval (3)
description: third project of the Modern Information Retrieval course
img: assets/img/projects/web_ir.jpg
importance: 1
category: work
giscus_comments: true
---

This project had several tasks, 1 - Crawling webpages 2 - Personalized PageRank 3 - Link Analysis the results with HITS algorithm 4 - Recommender System 5 - Evaluating the recommender system.

1- Crawling the Web:

The first task is to scrape the semanticscholar.org webpage for crawling papers. we start with 5 paper of 5 professor and continue with top 10 cited papers of each paper and continue until reaching 2000 paper, then store it in a json.

2 - Personalized PageRank:

In this part of the project, I implemented a personalized PageRank algorithm, an enhancement of the traditional PageRank, which considers user preferences to rank nodes in a graph based on their relevance to a specific user rather than their general importance. The personalized PageRank algorithm was applied to identify key papers that are most relevant to a particular professor's area of expertise. By inputting a specific field of interest, the algorithm analyzes the citation graph to output the top-ranked papers that are most closely aligned with that field. 

3 - HITS Algorithm:

In this project, I focused on ranking authors based on the citation relationships between them. The core idea is to analyze how authors cite each other in academic papers. Specifically, when Author A cites a paper (P) co-authored by Author B, it is considered a citation from Author A to Author B. Using this citation relationship, I constructed a citation graph where nodes represent authors and directed edges represent citations between them. To rank the authors, I applied the HITS (Hyperlink-Induced Topic Search) algorithm, which utilizes the concepts of "hub" and "authority" scores. These scores reflect an author's influence as a source of information (hub) and as a cited authority in the academic community. The final output is a ranked list of authors based on their hub and authority scores within the citation network.


4 - Recommender System

In this section of the project, I focused on developing a recommender system that suggests newly published papers to users based on their past reading history or interests. The recommender system is designed to analyze a user's previously liked papers and recommend new ones that are likely to match their preferences. The dataset provided in `recommended_papers.json` contains a list of users, where each user's `positive_papers` field includes 50 papers they have shown interest in, and the `recommendedPapers` field contains 10 newly published papers that the user has liked, ranked by importance.

The objective was to train the recommender system using this data. Users were split into training and testing groups. The model was trained on the training set using the `positive_papers` field to predict which new papers the users in the testing group would likely appreciate. During prediction, the `recommendedPapers` field for the test users was not used, ensuring that the recommendations were based purely on the learned preferences from the training data. The ultimate goal was to accurately forecast the new papers that the test users would find appealing, thereby evaluating the effectiveness of the recommender system.

I implemented this with two different methods: 1 - Collaborative Filtering and 2 - Content-Based Filtering.

1 - Collaborative Filtering:
In the Collaborative Filtering approach, the system recommends articles to a user based on the preferences of other users with similar tastes

2 - Content-Based Filtering:
In the Content-Based Filtering approach, the system recommends new articles to a user based on the content of articles they have previously liked

At the end I evaluated the performance of these recommender systems using the nDCG (normalized Discounted Cumulative Gain) metric.