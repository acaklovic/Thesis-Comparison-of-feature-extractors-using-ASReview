# Comparison-of-feature-extractors-using-ASReview

This is the data and code for the master's thesis: _Out with the Old and in with the New? - A Comparison of Classical vs. State-of-the-Art Feature Extractors in the Context of Systematic Reviews_

The purpose of the study was examine if state-of-the-art feature extractors (i.e., transformers like RoBERTa, MPNET, and SPECTER) can outperform classical feature extractors (i.e., tf-idf and Doc2Vec) when classifying systematic reviews as relevant or irrelevant. Multiple simulations were run using ASReview software to see how accurately the different feature extractors (in combination with various classifiers) classified research articles as relevant or irrelevant. 

ASReview is an AI active learning system that uses the titles and abstracts of research papers to classify a set of papers as relevant or irrelevant for the researcher (van de Schoot et al., 2021). More information on ASReview can be found at: https://github.com/asreview

### Files
1. _dataset_: Contains the data used in the study, the ASReview benchmark dataset about PTSD Trajectories dataset by Van de Schoot et al. The dataset can also be found at: https://github.com/asreview/systematic-review-datasets/tree/master/datasets/van_de_Schoot_2017

2. _code_: The code containing the preparation for the simulations and the simulation scripts themselves. 

- The classifiers used in this study (along with their implementations in ASReview) are as follows: 1) SVM, 2) Logistic Regression, 3) Random Forest using the sklearn library, 4) Naive Bayes using the sklearn Multinomial Naive Bayes classifier, and 5) NN2 classifier (a fully connected neural network with 2 hidden layers, dense and of the same size) (ASReview., 2022). Each of the classifiers was run in combination with each of the six feature extractors (with the exception of Naive Bayes, which was only run with tf-idf). 

- The feature extractor implementations are: 1) The Doc2Vec implementation using the genism library and 2) Tf-idf implemented using the sklearn library (ASReview., 2022). The sentence transformers are: 3) distilRoBERTa, 4) RoBERTA-base, 5) Allenai-SPECTER, and 6) All-mpnet-base-v2. The transformers were extracted from the Hugging Face sentence-transformers library (Hugging Face, n.d.) and implemented using the ASReview code for SBERT (ASReview., 2022). All-mpnet-base-v2 is the current default sentence transformer used by ASReview.

3. _visualizations_: The code for the visualizations. The main visualizations used in the study were recall plots based on WSS (Work Saved Over Sampling) and RRF (Relevant Records Found). Further information about the ASReview libraries for the metrics and visualizations in this study can be found at: https://github.com/asreview and https://asreview.readthedocs.io/en/latest/

5. _generated_data_: This folder contains all of the state files generated from the simulations. 

## Study Details

### Methods

### Results

### Requirements
The simulations were run on Google Colab Pro, using a high-RAM GPU to speed up the run-time of the simulations (Colab Pro can reach ~25 GB of RAM) using the ASReview Python API and command line interface. This study used ASReview version 0.19.3, but please note that the latest release of the ASReview software is version 1.0. It is recommended to have Python version 3.7 or higher. The implementation of the transformer models requires the intallation of the sentence-transformers library from Hugging Face (found at: https://huggingface.co/sentence-transformers)

### References

ASReview. (2022). API reference. ASReview LAB: Active learning for Systematic Reviews. Retrieved June 21, 2022, from https://asreview.readthedocs.io/en/latest/reference.html 

Hugging Face. (n.d.). Sentence-transformers/all-mpnet-base-V2 · hugging face. sentence-transformers/all-mpnet-base-v2 · Hugging Face. Retrieved June 5, 2022, from https://huggingface.co/sentence-transformers/all-mpnet-base-v2 

Hugging Face. (n.d.). Sentence-transformers/all-distilroberta-v1 · hugging face. sentence-transformers/all-distilroberta-v1 · Hugging Face. Retrieved June 5, 2022, from https://huggingface.co/sentence-transformers/all-distilroberta-v1 

Hugging Face. (n.d.). Sentence-transformers/allenai-specter · hugging face. sentence-transformers/allenai-specter · Hugging Face. Retrieved June 5, 2022, from https://huggingface.co/sentence-transformers/allenai-specter 

Hugging Face. (n.d.). Sentence-transformers/stsb-roberta-base-v2 · hugging face. sentence-transformers/stsb-roberta-base-v2 · Hugging Face. Retrieved June 5, 2022, from https://huggingface.co/sentence-transformers/stsb-roberta-base-v2 

van den Brand, S.A.G.E., van de Schoot, R. (2021). ASReview Simulation Mode: Class 101. Blogposts of ASReview.

van de Schoot, R., de Bruin, J., Schram, R., Zahedi, P., de Boer, J., Weijdema, F., ... others (2021). An open source machine learning framework for efficient and transparent systematic reviews. Nature Machine Intelligence, 3(2), 125–133. 

- Note: ASReview documentation and code were integral in this study, and can be found at: https://github.com/asreview


