# Comparison-of-feature-extractors-using-ASReview

This is the data and code for the master's thesis: _Out with the Old and in with the New? - A Comparison of Classical vs. State-of-the-Art Feature Extractors in the Context of Systematic Reviews_

The purpose of the study was examine if state-of-the-art feature extractors (i.e., transformers like RoBERTa, MPNET, and SPECTER) can outperform classical feature extractors (i.e., tf-idf and Doc2Vec) when classifying systematic reviews as relevant or irrelevant. Multiple simulations were run using ASReview software to see how accurately the different feature extractors (in combination with various classifiers) classified research articles as relevant or irrelevant. 

ASReview is an AI active learning system that uses the titles and abstracts of research papers to classify a set of papers as relevant or irrelevant for the researcher (van de Schoot et al., 2021). More information on ASReview can be found at: https://github.com/asreview

## Files
1. _dataset_: Contains the data used in the study, the ASReview benchmark dataset about PTSD Trajectories dataset by Van de Schoot et al. The dataset can also be found at: https://github.com/asreview/systematic-review-datasets/tree/master/datasets/van_de_Schoot_2017

2. _code_: The code containing the preparation for the simulations and the simulation scripts themselves. All simulations were run using the ASReview Python API and metrics were extracted with the command line. API documentation: https://asreview.readthedocs.io/en/latest/simulation_api_example.html

3. _visualizations_: The code for the visualizations. The main visualizations used in the study were recall plots based on WSS (Work Saved Over Sampling) and RRF (Relevant Records Found). Further information about the ASReview libraries for the metrics and visualizations in this study can be found at: https://github.com/asreview and https://asreview.readthedocs.io/en/latest/

5. _generated_data_: This folder contains all of the state files generated from the simulations. 

## Study Details

### Methods

The classifiers used in this study (along with their implementations in ASReview) are as follows: 
  1) SVM
  2) Logistic Regression
  3) Random Forest using the sklearn library
  4) Naive Bayes using the sklearn Multinomial Naive Bayes classifier
  5) NN2 classifier (a fully connected neural network with 2 hidden layers, dense and of the same size) 

Each of the classifiers was run in combination with each of the six feature extractors (with the exception of Naive Bayes, which was only run with tf-idf). 

The feature extractor implementations are: 
  1) The Doc2Vec implementation using the genism library
  2) Tf-idf implemented using the sklearn library 
The sentence transformers are:
  3) Distil-RoBERTa
  4) RoBERTA-base
  5) Allenai-SPECTER
  6) All-mpnet-base-v2

The transformers were extracted from the Hugging Face sentence-transformers library (Hugging Face, n.d.) and implemented using the ASReview code for SBERT (ASReview., 2022). All-mpnet-base-v2 is the current default sentence transformer used by ASReview.

A total of 25 combinations and simulations were run; all of the implemented feature extractors and classifiers can be viewed in the figure below: 

<img width="600" alt="image" src="https://user-images.githubusercontent.com/49207961/176444270-0ce804df-1a89-427f-a407-d7c55f963bfa.png">


The simulation settings were as follows:


<img width="500" alt="Screen Shot 2022-06-29 at 3 19 53 PM" src="https://user-images.githubusercontent.com/49207961/176446292-f1822d14-080d-4660-ac67-4dd0281db011.png">

_Performance Metrics:_

To assess the performance of the different models, the recall curves were plotted. The recall plots show two metrics - Work Saved Over Sampling (WSS) at 95% recall and Relevant References Found (RRF) at 10% recall. 

- WSS@95 means that at 95% recall, the given percentage is the amount of work that can be saved (in this case, how many less studies need to be screened). A higher WSS@95 is preferable since it indicates how much work the researcher can save by using machine learning instead of manual screening (to find 95% of all relevant articles) (van den Brand & van de Schoot, 2021).
- RRF@10 is the number of relevant articles found after screening 10% of the dataset. A higher RRF@10 score is preferable because it means that more relevant records have been found after screening only 10% of the articles. If the RRF does not change from RFF@5 to RRF@10 it could indicate that some of the relevant articles are hard to find and are taking longer to be found (van den Brand & van de Schoot, 2021). 
- Average Time to Discovery (ATD), was also utilized. ATD refers to the average time it takes to find a relevant article, expressed as a percentage/proportion of all articles in the dataset being screened (van den Brand & van de Schoot, 2021). This metric is useful for examining how much of the dataset needs to be screened in order to find a relevant article, and thus a lower ATD means the model is more efficient at discovering a relevant article.    


### Results

Tf-idf is the best performing feature extractor, especially in combination with the logistic regression, SVM, and Naive Bayes models. The results indicate that the best performing combination of feature extractor and classifier is tf-idf and Naive Bayes. This means that this model saves the most time for the researcher (WSS), finds the most relelvant articles while only needing to screen a small amount of the dataset (RRF), and needs a shorter amount of time to find a relevant article (ATD). This coincides with the past ASReview simulation results, which also indicated that the tf-idf and Naive Bayes combination has the best performance (van de Scoot et al., 2021).
 
Recall plot of top five best performing models (based on WSS@95). Models are listed in order - SPECTER with RF is the worst of the top five models; Tf-idf with Naive Bayes has the best performance amongst all of the models:

<img width="500" alt="image" src="https://user-images.githubusercontent.com/49207961/176447764-91d82c82-4e60-476a-b77d-d6c43b12ae37.png">

However, it is interesting to note that the SPECTER and random forest model is the top fourth model, while the Distil-RoBERTa and NN2 model is the top fifth model. The sentence transformer models may not be performing quite as well as tf-idf, but they do show some promise. Their WSS@95 scores are not drastically lower than those of tf-idf and Doc2Vec (except for RoBERTa-base). 

The tf-idf and Naive Bayes model has the highest RRF@10 at 100.00%. This means that the model found all relevant articles after only screening 10% of the dataset. The RRF@10 scores of the rest of the top five models are all close to 100%. Below is the RRF plot of top five models (for RRF@1, RRF@2, RRF@5, and RRF@10). Models are listed in order - SPECTER with RF is the worst of the top five models; Tf-idf with Naive Bayes has the best performance amongst all of the models:

 <img width="410" alt="Screen Shot 2022-06-29 at 3 34 43 PM" src="https://user-images.githubusercontent.com/49207961/176449563-be985283-11d3-47a1-b13f-11d043e2f9d5.png">

The results of the bottom five models based on WSS@95 show that RoBERTa-base is consistently the worst performing model. The next worst model is the MPNet model in combination with the logistic regression classifier. 

The recall plot of bottom five (the worst performance) models (based on WSS@95) can be seen below. Models are listed in order - MPNet with LR is the best of the bottom five models; Roberta-base with SVM has the worst performance of all models:

<img width="500" alt="image" src="https://user-images.githubusercontent.com/49207961/176447719-e91f32de-df20-41ba-a22b-e4d683b3c0df.png">

The RRF@10 scores of the bottom five models show more variation than the top five models. For all RRF values (RRF@1, RRF@5, etc.) the MPNET model performs better than RoBERTa-base, and RoBERTa-base combined with the NN2 classifier tends to have the lowest RRF values. The RRF plot of bottom five models (for RRF@1, RRF@2, RRF@5, RRF@10, RRF@20, and RRF@50) is:

<img width="481" alt="Screen Shot 2022-06-29 at 3 34 15 PM" src="https://user-images.githubusercontent.com/49207961/176449456-6026a502-e2ad-4ea8-9fed-7d3d7dc29dbe.png">

Further results and visualizations can be found in the files. 


## Requirements
The simulations were run on Google Colab Pro, using a high-RAM GPU to speed up the run-time of the simulations (Colab Pro can reach ~25 GB of RAM) using the ASReview Python API and command line interface. This study used ASReview version 0.19.3, but please note that the latest release of the ASReview software is version 1.0. It is recommended to have Python version 3.7 or higher. The implementation of the transformer models requires the intallation of the sentence-transformers library from Hugging Face (found at: https://huggingface.co/sentence-transformers)
## References

ASReview. (2022). API reference. ASReview LAB: Active learning for Systematic Reviews. Retrievedune 21, 2022, from https://asreview.readthedocs.io/en/latest/reference.html 

Hugging Face. (n.d.). Sentence-transformers/all-mpnet-base-V2 · hugging face. sentence-transformers/all-mpnet-base-v2 · Hugging Face. Retrieved June 5, 2022, from https://huggingface.co/sentence-transformers/all-mpnet-base-v2 

Hugging Face. (n.d.). Sentence-transformers/all-distilroberta-v1 · hugging face. sentence-transformers/all-distilroberta-v1 · Hugging Face. Retrieved June 5, 2022, from https://huggingface.co/sentence-transformers/all-distilroberta-v1 

Hugging Face. (n.d.). Sentence-transformers/allenai-specter · hugging face. sentence-transformers/allenai-specter · Hugging Face. Retrieved June 5, 2022, from https://huggingface.co/sentence-transformers/allenai-specter 

Hugging Face. (n.d.). Sentence-transformers/stsb-roberta-base-v2 · hugging face. sentence-transformers/stsb-roberta-base-v2 · Hugging Face. Retrieved June 5, 2022, from https://huggingface.co/sentence-transformers/stsb-roberta-base-v2 

van den Brand, S.A.G.E., van de Schoot, R. (2021). ASReview Simulation Mode: Class 101. Blogposts of ASReview.

van de Schoot, R., de Bruin, J., Schram, R., Zahedi, P., de Boer, J., Weijdema, F., ... others (2021). An open source machine learning framework for efficient and transparent systematic reviews. Nature Machine Intelligence, 3(2), 125–133. 

- Note: ASReview documentation and code were integral in this study, and can be found at: https://github.com/asreview


