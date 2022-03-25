
# Implementation of CANet Architecture

CANet is an attention based model built for "Semantic Segmentation"<br>

> Blog : https://www.arxiv-vanity.com/papers/2002.12041/
> Reasearch Paper: https://arxiv.org/pdf/2002.12041.pdf

The Chained Context Aggregation Network (CANet) is another fabulous approach for performing image segmentation tasks. It employs an asymmetric decoder to recover precise spatial details of prediction maps. This article is inspired by the research paper Attention-guided Chained Context Aggregation for Semantic Segmentation. This research paper covers most of the essential elements of this new technique of approach and how it can impact the performance of numerous complex tasks.

![image](https://user-images.githubusercontent.com/87253265/148214622-1b75f41f-8fd1-4892-b780-f9faa26110b2.png)

## How to use?
You can use the "model.py" file to get the complete Architecture

    from model import CANet

    #Now pass in the image shape and number of classes 
    canet_model = CANet(image_shape, classes)

    #check the summary for verification
    canet_model.summary()

#### Thank you


        

