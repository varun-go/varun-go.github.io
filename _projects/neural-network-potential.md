---
layout: single
title:  Neural-Network Potentials - A Deep Dive
permalink: /projects/neural-network-potential/
---

Understanding neural-network potentials (NNPs) and their various flavors!

## Series Posts:
{% for post in site.neural_network_potential reversed %}
* [{{ post.title }}]({{ post.url }}) - {{ post.excerpt | remove: '<p>' | remove: '</p>' }}
{% endfor %}