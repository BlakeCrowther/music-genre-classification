# Music Genre Classification with Librosa

The notebook for this demonstration was originally created as part of a project for the M.A.S. Data Science and Engineering Program at UCSD.

## Task

The original assignment was to investigate a dataset using a python library and present the findings. I chose to use [Librosa](https://librosa.org/), an audio and music processing library, to extract audio features from music files, and then attempt to train a music genre prediction model using [PyTorch](https://pytorch.org/).

## Dataset

To train my music genre prediction model I used the [Free Music Archive (FMA)](https://github.com/mdeff/fma), a dataset intended for music analysis. The full dataset is comprised of 106,574 untrimmed tracks, from 161 unbalanced genres.

## Results

The [Librosa_Exercise.ipynb](https://github.com/BlakeCrowther/music-genre-classification/blob/main/Librosa_Exercise.ipynb) notebook introduces Librosa by performing intial music information retrieval on local music files, and then constructs a genre prediction model trained using the FMA.

The [results.ipynb](https://github.com/BlakeCrowther/music-genre-classification/blob/main/results.ipynb) notebook contains a comparison of training a neural network on different features and subsets of the original dataset.
