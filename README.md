# TF-siamesenet

The main work of the article is to use the SiameseNet model to achieve the function of face recognition.
The SiameseNet convolutional neural network model is as follows (detailed structure moves to GitHub, a simplified version of VGG):
![SiameseNet model](https://upload-images.jianshu.io/upload_images/4019913-5b6b5dcdcb14cba6.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
![Input and output examples](https://upload-images.jianshu.io/upload_images/4019913-0e9464aa9f260866.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

The data set uses Microsoft's MS-Celeb-1M public face data set. We directly download the aligned data set. First, there is a cleaned list on the network. The file name is: MS-Celeb-1M_clean_list.txt, about 160M. Then use the combination algorithm to generate two files, positive_pairs_path.txt and negative_pairs_path.txt, each about 150W pairs.
![Generate file example](https://upload-images.jianshu.io/upload_images/4019913-c892e8c84203fb23.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

The key question is, how to define the loss function, the loss function is Logistic Regression loss function actually, it is worth noting that the last layer of the activation function should use Sigmoid Function. The Loss function is as follows:
![Loss function](https://upload-images.jianshu.io/upload_images/4019913-9061f2a26a078c24.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

The next step is training, training on TITAN X for about 3 days, we can see this Loss curve:
![Loss curve](https://upload-images.jianshu.io/upload_images/4019913-de14db779ade283d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

We can see that Loss converges very well, but after several trials, when Loss drops to around 0.1, it doesn't fall. I haven't found a reason yet. If you have ideas to solve this problem, you can give me a message or Email me (hfrommane@qq.com).

What about the correct rate of our model?
Haha, not very good, the correct rate on the LFW is 90%+, and the model needs to be optimized.

Project source code:
https://github.com/hfrommane/TF-siamesenet

Chinese readme version is:
https://www.jianshu.com/p/1df484d9eba9

If you like, give me a star, thanks.