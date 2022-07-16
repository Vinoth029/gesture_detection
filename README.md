
## Deep Learning - Gesture Recognition Project

*As a data scientist at a home electronics company that manufactures state-of-the-art smart televisions. We want to develop a cool feature in the smart TV that can recognize five different gestures performed by the user which will help users control the TV without using a remote.*

| **Actions** |	**Controls**        |       
|-------------|---------------------|
| Thumbs up   |	Increase the volume |
| Thumbs down |	Decrease the volume |
| Left swipe  |	Backward 10 seconds |
| Right swipe |	Forward 10 seconds  |
| Stop        | Pause               |  



### Dataset Reading
The training data consists of a few hundred videos categorized into one of the five classes. Each video (typically 2-3 seconds long) is divided into a sequence of 30 frames (images). These videos have been recorded by various people performing one of the five gestures in front of a webcam - similar to what the smart TV will use.

##### Preview
![](/images/preview.png)

### Objective
- To train different models on the 'train' folder to predict the action performed in each sequence or video and which performs well on the 'Val' folder as well.
- The final model should have a smaller number of parameters (lightweight) possible so that it can be implemented on the webcam mounted on top of the TV.

## Generator Function
*This is one of the most important parts of the code. In the generator, we are going to pre-process the images as we have images of 2 different dimensions (360 x 360 & 120 x 160) as well as create a batch of video frames. The generator should be able to make a batch of videos as input without any error. Steps like cropping, resizing and normalization should be performed successfully*

### Data Pre-processing:
- `Resizing:` As Input images are of two different dimensions, we have to resize the images to a common shape (image height x image width). The output image dimension is decided by the argument passed to the generator function. The output image dimension can be adjusted by the hardware available to train the model further there is a trade-off between the output image dimension and the batch size.
 
- `Normalization:` Normalization is a technique by which we can scale the values without losing out on the information. It plays a vital role in speeding up the training process. We read the images via OpenCV we convert the image from BGR to RGB, we resize it then finally we will normalize each channel separately.

- `Batch creation:` This is the curtail part of the generator function. It creates a batch of images and labels and returns them to the caller function unlike other functions generator function works differently by the technique called yielding it does not process every batch in a single go instead it yields only one batch of data when the generator function is called once. 

