---
layout: page
title: Binary Stress Detection
description: Fine-tuned, evaluated and compared LLMs for binary stress detection and stress category classification. 
img: assets/img/projects/stress.jpg
importance: 1
category: work
related_publications: false
---
The primary goal of this project was to Finetune LMs for 1- binary detection of stress 2- classify the category of stress and also 3- evaluating and comparing different models for stress detection. [Source Code](https://github.com/Nima-Nilchian/stress-detection)

the work was done with Dreaddit: A Reddit Dataset for Stress Analysis in Social Media. which is shown below:

<div class="row justify-content-center">
    <div class="col-sm-8 col-md-6 mt-3">
        {% include figure.liquid loading="eager" path="assets/img/projects/dataset.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
        <div class="caption">the overview of the dreaddit dataset</div>
    </div>
</div>

Results:
All finetuned model performed better in the test dataset of the dreaddit, as shown below:

<div class="row justify-content-center">
    <div class="col-sm-8 col-md-6 mt-3">
        {% include figure.liquid loading="eager" path="assets/img/projects/results.png" title="example image" class="img-fluid rounded z-depth-1" %}
        <div class="caption">evaluation result of finetuned model</div>
    </div>
</div>


<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/projects/stress_2.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

