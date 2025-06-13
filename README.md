# CS410_final
# Topological Data Analysis in Image Processing

**Authors**: Jakob Rode, Jayden West, Marko Marceta, Mateo Campos Davis  
**Date**: June 2025

This project is about the application of Topological Data Analysis in image processing. Highlighting its strengths while showcasing it in several cases. This includes building a custom experiment on cartoon characters where we aim for recognition using persistence diagrams.

---


## Overview

Topological Data Analysis provides us with tools to study the "shape" of data by plotting when shapes connect into holes or faces. In this project, we:

- Present the mathematical background of TDA (simplicial complexes, persistent homology, filtrations).
- Explore real-world applications including:
    - Satellite cloud classification
    - Brain connectivity in children with autism
- Perform our own analysis using a dataset of cartoon character images.
- Vectorize our data to allow way for a machine learning model

---

## Methods

- **Libraries used**
  - "gudhi" – for computing persistence diagrams
  - "scikit-learn" – for machine learning (logistic regression)
  - "numpy", "matplotlib" – standard Python data stack
 
---

## Key Insights

- Even with a simple classifier and small dataset, TDA reveals several useable features.
- Persistence diagrams enable structural analysis of images that go beyond pixel values.
- Combining TDA with machine learning allows for efficient and explainable image classification.

