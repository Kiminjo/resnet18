# resnet18

Resnet18 is applied to industrial data to classify data mixed with outliers.
<br></br>
<br></br>

## Anomaly classifiaction using resnet18
The most important thing in the industrial field is to detect defective products early and deliver high-quality products. To this end, many technologies have been developed to detect defective products (outlier products) in advance, and recently, methods using deep learning have also been actively studied.
<br></br>

This project can be approached as a traditional classification problem because there is an appropriate amount of labeling data along with a sufficient amount of data. Accordingly, outlier data in the image was classified using the resnet18 model.
<br></br>

In the future, when similar tasks are given, they also serve as performance records to present a guide for model selection. Therefore, efforts were made to measure the performance of the model itself except for the preprocessing process or data enhancement process that helps the model's performance.
<br></br>
<br></br>

## Performance evaluation 
Wandb, an MLOps tool for deep learning, was used to measure performance. In addition, we tried to find the best combination of performance in multiple hyperparameter spaces using sweep, a hyperparameter tuning tool.