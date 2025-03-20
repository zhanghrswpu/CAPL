# CAPL
The implementation of CAPL.
## Datasets 
| Dataset  | #sample | link |  
|------------|---------|----------|  
| donors | 619,326 | [Kaggle](https://www.kaggle.com/c/kdd-cup-2014-predictingexcitement-at-donors-choose) |  
| celeba | 202,599 | [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) |  
| dos | 109,353 | [UNSW](https://www.unsw.adfa.edu.au/unsw-canberracyber/cybersecurity/ADFA-NB15-Datasets/) |  
| rec | 106,987 | [UNSW](https://www.unsw.adfa.edu.au/unsw-canberracyber/cybersecurity/ADFA-NB15-Datasets/) |  
| fuz | 96,000 | [UNSW](https://www.unsw.adfa.edu.au/unsw-canberracyber/cybersecurity/ADFA-NB15-Datasets/) |  
| bac | 95,329 | [UNSW](https://www.unsw.adfa.edu.au/unsw-canberracyber/cybersecurity/ADFA-NB15-Datasets/) |  
| thyroid | 7,200 | [OpenML](https://www.openml.org/d/40497) |  

## Experiments

The below function is used to perform seen anomaly detection,
```python
cd seen_anomaly_detection
run_seen_anomaly_detection(args)
```
while *run_unseen_anomaly_detection(args)* is used to perform unseen anomaly detection.
```python
cd unseen_anomaly_detection
run_unseen_anomaly_detection(args)
```
CAPL is implemented using Tensorflow/Keras. The main packages and their versions used in this work are provided as follows:
- keras==2.3.1
- numpy==1.16.2
- pandas==0.23.4
- scikit-learn==0.20.0
- scipy==1.1.0
- tensorboard==1.14.0
- tensorflow==1.14.0

