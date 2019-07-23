## Week 03 Record

### Monday 7.15

1. Finish modifying the pull request according to Yicong's comments. 

2. Clean git commit history with Yichi and Tingxuan. Add two more attributes *starttime* and *endtime* into databse.



### Tuesday 7.16

1. (1) Finish event2mind classifier based on ClassifierBase class. 

   (2) Add event2mind model into backend/models. 

   (3) Modify configurations.py adding EVENT2MIND_MODEL_PATH.

2. Make a pull request.



### Wednesday 7.17

1. Finish event2mind dumper and classifier. 
2. And the pull request is merged to master branch.



### Thursday 7.18

1. Build training and testing dataset for images from tweet, wildfire or not wildfire. 

2. Set up CNN model to train images. 

3. Test accuracy is 80%, able to detect obvious smoke or fire as wildfire in the image.

   

### Friday 7.19

1. Write image classifier and image classification dumper, made a pull request. 
2. The process is done, which is of getting urls from dataset, download images from urls, use CNN model to make predictions, dump prediction result into database. 
3. Refactored event2mind classifier and add files in tasks to follow Runnable class.