# EMG Signal Processing for Prosthetic Finger Control
***

## Introduction.
The common used sensor for measuring the activity of muscle is electromyography (EMG). The raw signal from EMG sensor is hard to understand and interpreted by human. Thus, the implementation of pattern recognition method that can interpret the muscle activity or finger movement has important role. This project is focused on the feature extraction and pattern recognition of 
eight finger movement classification using EMG raw signals.

[1] is the initial study of this problem statement. Data is being collected via 2 surface electrode of 10 finger movements. After the collection of data, it is windowed and feature extraction is done using TD+AR+Hjorth and then the features are reduced via LDA. It is trained via KNN+SVM+Fusion and overall accuracy of 90% is obtained. 

[2,3] does further studies on this dataset. [2] does a 5 class single finger movement classification and attains a accuracy of 96.7% with same TD+AR+Hjorth feature extraction and ANN. [3] took a different path and used frequency time(wavelet) transformation and then took 17 features out of it and achieved a accuracy of 95.5% in 10 class classification. 

Other than this different researches have been done on this dataset with different feature extraction and classification algorithm. 

We have used 2 different feature extraction and 2 different classification algorithm for a 8 class classification and achieved a accuracy of 94% which is at par with the industry standard. 