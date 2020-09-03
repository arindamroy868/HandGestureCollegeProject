# HandGestureRecognition
It is a very basic CNN based Machine Learning Model to detect Hand Gestures.

This is a college project in which we have built a machine learning model to recognize a hand gesture which is available in the database.
1.We used CNN algorith to train the model 
2.Training and Testing dataset consists of over 3000 monochrome images of resolution 400x400 . Each class of hand gesture has average 600 images in it.
3.Model is trained using monochrome images so it can only take input of monochrome images of 400x400 so it can be used to identify the the gesture using the model.
4.Model has accuracy of over 92% on test dataset and 90% on validation dataset.

#Limitations of this model
1.Since model is trained on monochrome images and can only detect patterns in a monochrome image, we have lot less detail to work on.
2.Monochrome images leads to one more challenege and that is: we have to seprate the background from foreground to remove noise in the image.
3.To seprate the foreground from backgrou we used pixel based background subtraction,in which we take still image of background without the foreground and the we introduce the 00000
  foreground object in it.This makes it possible to fill the foreground object with white pixles and background with black pixels.
