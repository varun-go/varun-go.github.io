---
title: "Projects"
layout: single
permalink: /projects/
classes: wide
---

The following posts contain notes on various mini-projects that are relevant to my research. These notes are primarily a means to solidify my own understanding of various concepts. However, I hope that these posts can be useful to other chemical engineers who are getting started in the application of data science and machine learning to molecular simulations.

## Neural-Network Potentials for Molecular Simulations
A series of posts explaining neural-network potentials.

{% for project in site.projects %}
{% if project.url contains "/projects/neural-network-potential/" %}
- [{{ project.title }}]({{ project.url }})
{% endif %}
{% endfor %}

## Graph-Based Neural Networks for Predicting Molecular Properties
A series of posts explaining on the use of graph neural networks for molecular property prediction. 

<!-- {% for project in site.projects %}
{% if project.url contains "/projects/graph-based-neural-network-potential/" %}
- [{{ project.title }}]({{ project.url }})
{% endif %}
{% endfor %} -->