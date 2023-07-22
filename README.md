# Baggin_SVM
Code for handling imbalanced data（SVM/Bagging SVM/Deep Learning）
## Requirements
* Python 3  

* PyTorch 1.3+ (feel free if it's a CUDA or CPU version) (Test with 2.0)  

* sklearn  

* pandas  
* matplotlib  
## Usage
This experiment was written based on JupyterNotebook.   
Of course, if you are having trouble using jupyter, you can run three deep learning models in py format files.  
<u>Change your dataset path</u>  
## Parameters in DeepLearning models  
* ***pos_weight***  
  Which determines the magnification of the gradient when calculating the loss of subcategory samples  
```criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(5))```
* ***threshold***  
  Classification confidence, as the final output of the model is a probability value, a normal 0.5 will directly
   lead to overfitting, which has a significant impact on the recall rate of subcategories  
```threshold = 0.2```  

## Issue
* Why did the output of the model not use sigmoid mapping to 0-1? Because the gradient calculation of sigmoid on small class samples was not significant during the experimental process, and the model was more overfitting  
* If the deep learning model does not display an image after running it in jupyter notebook, you may need to add ```%matplotlib inline``` in the first line

