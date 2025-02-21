---
title: "Projects"
layout: single
permalink: /projects/
---

## Neural-Network Potentials for Molecular Simulations
A series of posts explaining neural-network potentials.

{% for project in site.projects %}
{% if project.url contains "/projects/neural-network-potential/" %}
- [{{ project.title }}]({{ project.url }})
{% endif %}
{% endfor %}

## A Graph-Based Neural Network Potential for Predicting Molecular Properties
A series of posts explaining how to build a graph-based neural network potential.
[View Series](/projects/graph-based-neural-network-potential/)