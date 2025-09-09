# CBIR_Cat-Image-Search-Project 🐱

## Overview
This project allows users to search for a specific cat image within a dataset containing only cat images. The system compares images based on visual similarity using **Python** and **image histograms**, returning the closest match efficiently.

## Features
- Search for a specific cat image within a dataset
- Image comparison using histogram similarity
- Fast and accurate retrieval of images
- Simple and easy-to-understand implementation for learning purposes

## Technologies Used
- Python 3.x
- OpenCV
- NumPy

## How It Works
1. Load the dataset of cat images.
2. Convert each image to a histogram representation.
3. Compare the histogram of the query image with the dataset images using similarity metrics.
4. Rank the images based on similarity and return the best match.

## Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/cat-image-search.git
