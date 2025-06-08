---
layout: page
title: BandBooster
description: AI personal assistant for IELTS writing preparation, utilizing LLMs for personalized feedback, scoring, and improvement suggestions.
img: assets/img/projects/bandbooster.png
importance: 1
category: personal
---

<!-- [Source Code](https://github.com/Nima-Nilchian/Web-IR) -->

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include video.liquid path="assets/video/bandbooster.mp4" class="img-fluid rounded z-depth-1" controls=true autoplay=true %}
    </div>
</div>

# BandBooster

BandBooster is an AI-powered personal assistant designed to help IELTS candidates improve their writing skills. By leveraging large language models (LLMs), BandBooster provides automatic scoring, personalized feedback, detailed error explanations, and improved sample rewrites — all aligned with official IELTS criteria.

## Project Overview

### Problem Statement
- IELTS writing assessment is traditionally time-consuming, expensive, and reliant on human raters.
- Existing feedback tools are generic and lack personalized insights.
- Learners need fast, specific, and interactive feedback to make meaningful progress.

### Objectives
- Develop an **automated essay scoring system** using LLMs.
- Provide **personalized feedback** on grammar, structure, and vocabulary.
- Suggest **revised versions** of user writings.
- evaluate models against **official IELTS rubrics**.

---

## Methodology


### Dataset
- 4 Cambridge IELTS books (19 exercises total).
- Each sample includes: Question, User Essay, Human Score, and Feedback.
- Total of 72 labeled writing samples.

<br/>

### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| QWK (Quadratic Weighted Kappa) | Measures agreement between model and human scores |
| MAE (Mean Absolute Error) | Measures average error between predicted and actual scores |

---

## Experiments & Results

### Effect of IELTS Rubric Inclusion

| Model       | QWK (No Rubric) | MAE (No Rubric) | QWK (With Rubric) | MAE (With Rubric) |
|-------------|------------------|------------------|-------------------|-------------------|
| GPT-4o      | 0.638            | 1.31             | **0.839**         | **0.613**         |
| GPT-4o-mini | 0.647            | 1.069            | **0.839**         | **0.597**         |
| DeepSeek-V3 | 0.694            | 0.958            | **0.714**         | **0.812**         |

<br/>

**Inclusion of IELTS-specific rubric improved both QWK and MAE across all models.**

---

### Effect of Output Format

| Model       | Format     | QWK   | MAE   |
|-------------|------------|-------|-------|
| GPT-4o      | YAML       | 0.853 | 0.646 |
| GPT-4o      | JSON       | 0.839 | 0.618 |
| GPT-4o-mini | JSON       | 0.831 | 0.597 |
| O3-mini     | JSON       | 0.899 | **0.396** |
| DeepSeek-V3 | YAML       | 0.741 | 0.75  |

<br/>

🏆 **O3-mini** outperformed all other models with the best MAE and QWK.

---

## Feedback System Features

1. **Error Classification:**
   - Grammatical
   - Structural
   - Vocabulary

2. **Error Explanation:**
   - Detailed educational insight for each mistake

3. **Rewritten Version:**
   - Enhanced version of user essay based on feedback




<div class="row">
    <div class="col-sm mt-6 mt-md-2">
        {% include figure.liquid loading="eager" path="assets/img/projects/formatted_essay.png" title="example image" class="img-fluid rounded z-depth-1" %}
    <div class="caption">Formatting essay based on errors in 3 classes</div>
    </div>
    <div class="col-sm mt-6 mt-md-4">
        {% include figure.liquid loading="eager" path="assets/img/projects/explanation.png" title="example image" class="img-fluid rounded z-depth-1" %}
    <div class="caption">Error Explanation</div>
    </div>
</div>
<div class="row">
    <div class="col-sm mt-6 mt-md-4">
        {% include figure.liquid loading="eager" path="assets/img/projects/corrections.png" title="example image" class="img-fluid rounded z-depth-1" %}
    <div class="caption">Correcting Errors</div>
    </div>
</div>
