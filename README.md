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

# Results of Re-Implementation

For the most part, noise does appear to be much reduced by our DNN, with
a very readable output image. Individual outputs at both noise levels
can be seen in Figures 1 and 2. PSNR (Peak Signal to Noise Ratio) and
SSIM (Structural Similarity Index Measure) are metrics to help define
the similarity between clean and unclean images. Higher is better.

![PSNR = 3.49, SSIM = 0.39 when $\sigma$ =
50](checkpt3_3.1.png){#fig:plane width="6cm"}

![PSNR = 3.49, SSIM = 0.39 when $\sigma$ =
50](checkpt3_4.1.png){#fig:plane width="6cm"}

When comparing to the paper's results, the PSNR and SSIM values are
smaller. The original paper used images of size 180 x 180, which is much
larger than our 32 x 32 images. This would contribute to our values
being so much smaller.

# Areas for Improvement

Not every image came out with optimal values. In some instances, the
PSNR or SSIM came out as negative values. Example images can be seen in
Figures 3 and 4.

![PSNR = 2.49, SSIM = -0.07 when $\sigma$ =
50](checkpt3_badPSNR1.png){#fig:carrrr width="6cm"}

![PSNR = 2.49, SSIM = -0.07 when $\sigma$ =
50](checkpt3_badSSIM1.png){#fig:carrrr width="6cm"}

Our noise also seems to be extreme in our bright whites and deep blacks.
So, our noise function may need to be adjusted. The denoised images also
appear to have a haze over them that we could try to remedy in our
application to a real world problem.

# Bibliography

Kai Zhang, Wangmeng Zuo, Yunjin Chen, Deyu Meng, and Lei Zhang. Beyond a
gaussian denoiser: Residual learning of deep cnn for image denoising.
IEEE Trans. Image Process., 26(7):3142--3155, 2017

Tongyao Pang, Huan Zheng, Yuhui Quan, and Hui Ji.
Recorrupted-to-Recorrupted: Unsupervised Deep Learning for Image
Denoising. IEEE/CVF Conference on Computer Vision and Pattern
Recognition, 2021. DOI: 10.1109/CVPR46437.2021.00208
https://github.com/PangTongyao/Recorrupted-to-Recorrupted-Unsupervised-Deep-Learning-for-Image-Denoising
