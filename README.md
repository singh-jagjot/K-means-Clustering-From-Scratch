# Image Compression with K-means from Scratch
My implementation of compressing the image using K-means. To get a deeper understanding I've decided **not to use** any famous libraries like Tensorflow, Pytorch, etc. But, used the NumPy library for Vectorization and Pillow to read images.  
I have also implemented the cost/distortion function and logic to run K-means upto provided 'iters' and return the best values out of 'iters' iterations.  

## Instructions:
1) Install python3 on your systems.
2) Use below commands to install the required packages:
    >python3 -m pip install --upgrade pip  
    >python3 -m pip install --upgrade Pillow  
    >python3 -m pip install --upgrade matplotlib  
3) Run image_compression_kmeans.ipynb file? ??

## References:
1) Machine Learning Specialization by Andew Ng on [**Coursera**](https://www.coursera.org/specializations/machine-learning-introduction).  
2) Main notes from Stanford CS229 course.  

# Output:
![figure that compares outputs with the original](https://github.com/singh-jagjot/image-compression-with-K-means/blob/main/output/results.png)

**Note**: All images are downloaded from [**Pixabay**](https://pixabay.com) and are used according to the license.