# Toy Problem

Using the CIFAR-10 dataset, we wish to remove AWGN with the R2R model.
To do this, we will add zero-mean AWGN at two noise levels ($\sigma$ =
25, 50) like the original paper did to create our training and testing
sets. The training set consists of 50,000 images, and the testing set
has 10,000 images of size 32 x 32. After putting the data through the
R2R model, we will look at the PSNR and SSIM values to determine the
performance of our reconstruction.

# Methods of Re-Implementation

Our re-implementation begins by defining a class to add AWGN to our
images with our selected noise levels. For the creation of our datasets,
we also had to redefine the getitem function of the CIFAR-10 dataset to
output pairs of noisy and clean images. The reconstruction also required
the set up of a DnCNN. Similar to the paper, we set up our DnCNN with 17
convolution layers.

In our training function, we create our pair of noisy images at the
selected noise levels of $\sigma$ = 25, 50. Like the original paper,
training was done for 50 epochs with a batch size of 128. The testing
function outputs the results of our PSNR and SSIM for the problem. A
sample testing function was included to observe these values for an
individual test image. It also displayed the clean and noisy version of
the selected individual image and was run for both noise levels.

# Bibliography

Kai Zhang, Wangmeng Zuo, Yunjin Chen, Deyu Meng, and Lei Zhang. Beyond a
gaussian denoiser: Residual learning of deep cnn for image denoising.
IEEE Trans. Image Process., 26(7):3142--3155, 2017

Tongyao Pang, Huan Zheng, Yuhui Quan, and Hui Ji.
Recorrupted-to-Recorrupted: Unsupervised Deep Learning for Image
Denoising. IEEE/CVF Conference on Computer Vision and Pattern
Recognition, 2021. DOI: 10.1109/CVPR46437.2021.00208
https://github.com/PangTongyao/Recorrupted-to-Recorrupted-Unsupervised-Deep-Learning-for-Image-Denoising
