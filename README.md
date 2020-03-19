# Learning a low dimensional manifold of real cancer tissue with Pathology-GAN

**Abstract:**

*Application of deep learning in digital pathology shows promise on improving disease diagnosis and understanding. We present a deep generative model that learns to simulate high-fidelity cancer tissue images while mapping the real images onto an interpretable low dimensional latent space. The key to the model is an encoder trained by a previously developed generative adversarial network, PathologyGAN. We study the latent space using 249K images from two breast cancer cohorts. We find that the latent space encodes morphological characteristics of tissues (e.g. patterns of cancer, lymphocytes, and stromal cells). In addition, the latent space reveals distinctly enriched clusters of tissue architectures in the high-risk patient group.*

## Datasets:
H&E breast cancer databases from the Netherlands Cancer Institute (NKI) cohort and the Vancouver General Hospital (VGH) cohort with 248 and 328 patients respectevely. Each of them include tissue micro-array (TMA) images, along with clinical patient data such as survival time, and estrogen-receptor (ER) status. The original TMA images all have a resolution of 1128x720 pixels, and we split each of the images into smaller patches of 224x224, and allow them to overlap by 50%. We also perform data augmentation on these images, a rotation of 90 degrees, and 180 degrees, and vertical and horizontal inversion. We filter out images in which the tissue covers less than 70% of the area. In total this yields a training set of 249K images, and a test set of 62K.

We use these Netherlands Cancer Institute (NKI) cohort and the Vancouver General Hospital (VGH) previously used in Beck et al. \[1]. These TMA images are from the [Stanford Tissue Microarray Database](https://tma.im/cgi-bin/home.pl)[2]

\[1] Beck, A.H. and Sangoi, A.R. and Leung, S. Systematic analysis of breast cancer morphology uncovers stromal features associated with survival. Science translational medicine (2018).

\[2] Robert J. Marinelli, Kelli Montgomery, Chih Long Liu, Nigam H. Shah, Wijan Prapong, Michael Nitzberg, Zachariah K. Zachariah, Gavin J. Sherlock, Yasodha Natkunam, Robert B. West, Matt van de Rijn, Patrick O. Brown, and Catherine A. Ball. The Stanford Tissue Microarray Database. Nucleic Acids Res 2008 36(Database issue): D871-7. Epub 2007 Nov 7 doi:10.1093/nar/gkm861.

You can find a pre-processed HDF5 file with patches of 224x224x3 resolution [here](https://drive.google.com/open?id=1LpgW85CVA48C8LnpmsDMdHqeCGHKsAxw), each of the patches also contains labeling information of the estrogen receptor status and survival time.

This is a sample of an original TMA image:
<p align="center">
  <img src="https://github.com/AdalbertoCq/Pathology-GAN/blob/master/demos/original_tma.jpg" width="400">
</p>
