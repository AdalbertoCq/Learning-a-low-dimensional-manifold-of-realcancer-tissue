# Learning a low dimensional manifold of real cancer tissue with Pathology-GAN

**Abstract:**

*Application of deep learning in digital pathology shows promise on improving disease diagnosis and understanding. We present a deep generative model that learns to simulate high-fidelity cancer tissue images while mapping the real images onto an interpretable low dimensional latent space. The key to the model is an encoder trained by a previously developed generative adversarial network, PathologyGAN. We study the latent space using 249K images from two breast cancer cohorts. We find that the latent space encodes morphological characteristics of tissues (e.g. patterns of cancer, lymphocytes, and stromal cells). In addition, the latent space reveals distinctly enriched clusters of tissue architectures in the high-risk patient group.*

## Demo Materials:

* [**Real Tissue Reconstructions**](https://github.com/AdalbertoCq/Pathology-GAN/tree/master/demos/real_recon):
  **(a)** correspond to the real tissue images and **(b)** to the reconstructions, the images are  paired  in  columns. The reconstructions follow the real imageattributes.
<p align="center">
    <img src="https://github.com/AdalbertoCq/Learning-a-low-dimensional-manifold-of-realcancer-tissue/blob/master/demos/real_recon/Real_reconstructed_PathologyGAN_Enc_Incr_0.png" width="700">
    <img src="https://github.com/AdalbertoCq/Learning-a-low-dimensional-manifold-of-realcancer-tissue/blob/master/demos/real_recon/Real_reconstructed_PathologyGAN_Enc_Incr_1.png" width="700">
    <img src="https://github.com/AdalbertoCq/Learning-a-low-dimensional-manifold-of-realcancer-tissue/blob/master/demos/real_recon/Real_reconstructed_PathologyGAN_Enc_Incr_2.png" width="700">
    <img src="https://github.com/AdalbertoCq/Learning-a-low-dimensional-manifold-of-realcancer-tissue/blob/master/demos/real_recon/Real_reconstructed_PathologyGAN_Enc_Incr_3.png" width="700">
    <img src="https://github.com/AdalbertoCq/Learning-a-low-dimensional-manifold-of-realcancer-tissue/blob/master/demos/real_recon/Real_reconstructed_PathologyGAN_Enc_Incr_4.png" width="700">
</p>

* [**Latent Space Images**](https://github.com/AdalbertoCq/Pathology-GAN/tree/master/demos/latent_space):
   Uniform Manifold Approximation and Projection (UMAP) representation of real tissue samples in the latent space using samples from Netherlands CancerInstitute (NKI) and Vancouver General Hospital (VGH) patient cohorts. 
<p align="center">
    <img src="https://github.com/AdalbertoCq/Learning-a-low-dimensional-manifold-of-realcancer-tissue/blob/master/demos/latent_space/NKI%2BVGH_real_latent_full.png" width="400">
 <img src="https://github.com/AdalbertoCq/Learning-a-low-dimensional-manifold-of-realcancer-tissue/blob/master/demos/latent_space/NKI%2BVGH_real_latent_linear_inter.png" width="400">
</p>

* [**Survival Tissue Predominance**](https://github.com/AdalbertoCq/Pathology-GAN/tree/master/demos/survival):
  - **NKI**: 
    1. Densities  of  tissue  architectures  in  patients  with  greater  **(a)**  and  lesser **(b)**  than  5  year  survival  of  NKI  cohort. Tissue predominance on high-risk patients.
       <p align="center">
           <img src="https://github.com/AdalbertoCq/Learning-a-low-dimensional-manifold-of-realcancer-tissue/blob/master/demos/survival/NKI_real_More_presence_in_lesser_5_years_Patches_greater__5_years_n_comp_200.png" width="380">
        <img src="https://github.com/AdalbertoCq/Learning-a-low-dimensional-manifold-of-realcancer-tissue/blob/master/demos/survival/NKI_real_More_presence_in_lesser_5_years_Patches_greater__5_years_n_comp_200.png" width="380">
       </p>
    2. Densities  of  tissue  architectures  in  patients  with  greater  **(a)**  and  lesser **(b)**  than  5  year  survival  of  NKI  cohort. Tissue predominance on low-risk patients.
       <p align="center">
           <img src="https://github.com/AdalbertoCq/Learning-a-low-dimensional-manifold-of-realcancer-tissue/blob/master/demos/survival/NKI_real_More_presence_in_greater__5_years_Patches_greater__5_years_n_comp_200.png" width="380">
        <img src="https://github.com/AdalbertoCq/Learning-a-low-dimensional-manifold-of-realcancer-tissue/blob/master/demos/survival/NKI_real_More_presence_in_greater__5_years_Patches_greater__5_years_n_comp_200.png" width="380">
       </p>
  - **VGH**: 
    1. Densities  of  tissue  architectures  in  patients  with  greater  **(a)**  and  lesser **(b)**  than  5  year  survival  of  VGH  cohort. Tissue predominance on high-risk patients.
       <p align="center">
           <img src="https://github.com/AdalbertoCq/Learning-a-low-dimensional-manifold-of-realcancer-tissue/blob/master/demos/survival/VGH_real_More_presence_in_lesser_5_years_Patches_greater__5_years_n_comp_200.png" width="380">
        <img src="https://github.com/AdalbertoCq/Learning-a-low-dimensional-manifold-of-realcancer-tissue/blob/master/demos/survival/VGH_real_More_presence_in_lesser_5_years_Patches_greater__5_years_n_comp_200.png" width="380">
       </p>
    2. Densities  of  tissue  architectures  in  patients  with  greater  **(a)**  and  lesser **(b)**  than  5  year  survival  of  VGH  cohort. Tissue predominance on low-risk patients.
       <p align="center">
           <img src="https://github.com/AdalbertoCq/Learning-a-low-dimensional-manifold-of-realcancer-tissue/blob/master/demos/survival/VGH_real_More_presence_in_greater__5_years_Patches_greater__5_years_n_comp_200.png" width="380">
        <img src="https://github.com/AdalbertoCq/Learning-a-low-dimensional-manifold-of-realcancer-tissue/blob/master/demos/survival/VGH_real_More_presence_in_greater__5_years_Patches_greater__5_years_n_comp_200.png" width="380">
       </p>
       

## Datasets:
H&E breast cancer databases from the Netherlands Cancer Institute (NKI) cohort and the Vancouver General Hospital (VGH) cohort with 248 and 328 patients respectevely. Each of them include tissue micro-array (TMA) images, along with clinical patient data such as survival time, and estrogen-receptor (ER) status. The original TMA images all have a resolution of 1128x720 pixels, and we split each of the images into smaller patches of 224x224, and allow them to overlap by 50%. We also perform data augmentation on these images, a rotation of 90 degrees, and 180 degrees, and vertical and horizontal inversion. We filter out images in which the tissue covers less than 70% of the area. In total this yields a training set of 249K images, and a test set of 62K.

We use these Netherlands Cancer Institute (NKI) cohort and the Vancouver General Hospital (VGH) previously used in Beck et al. \[1]. These TMA images are from the [Stanford Tissue Microarray Database](https://tma.im/cgi-bin/home.pl)[2]

\[1] Beck, A.H. and Sangoi, A.R. and Leung, S. Systematic analysis of breast cancer morphology uncovers stromal features associated with survival. Science translational medicine (2018).

\[2] Robert J. Marinelli, Kelli Montgomery, Chih Long Liu, Nigam H. Shah, Wijan Prapong, Michael Nitzberg, Zachariah K. Zachariah, Gavin J. Sherlock, Yasodha Natkunam, Robert B. West, Matt van de Rijn, Patrick O. Brown, and Catherine A. Ball. The Stanford Tissue Microarray Database. Nucleic Acids Res 2008 36(Database issue): D871-7. Epub 2007 Nov 7 doi:10.1093/nar/gkm861.

You can find a pre-processed HDF5 file with patches of 224x224x3 resolution [here](https://drive.google.com/open?id=1LpgW85CVA48C8LnpmsDMdHqeCGHKsAxw), each of the patches also contains labeling information of the estrogen receptor status and survival time.

This is a sample of an original TMA image:
<p align="center">
  <img src="https://github.com/AdalbertoCq/Learning-a-low-dimensional-manifold-of-realcancer-tissue/blob/master/demos/original_tma.jpg" width="400">
</p>   

## Pre-trained Models:

You can find pre-trained weights for the model [here](https://figshare.com/s/0a311b5418f21ab2ebd4)

## Python Enviroment:
```
h5py                    2.9.0
numpy                   1.16.1
pandas                  0.24.1
scikit-image            0.14.2
scikit-learn            0.20.2
scipy                   1.2.0
seaborn                 0.9.0
sklearn                 0.0
tensorboard             1.12.2
tensorflow              1.12.0
tensorflow-probability  0.5.0
python                  3.6.7
```

## Training PathologyGAN:
You can find a pre-processed HDF5 file with patches of 224x224x3 resolution [here](https://drive.google.com/open?id=1LpgW85CVA48C8LnpmsDMdHqeCGHKsAxw), each of the patches also contains labeling information of the estrogen receptor status and survival time. Place the 'vgh_nki' under the 'dataset' folder in the main PathologyGAN path.

Each model was trained on an NVIDIA Titan Xp 12 GB for 45 epochs, approximately 80 hours.

```
usage: run_pathgan_encoder.py [-h] [--epochs EPOCHS] [--batch_size BATCH_SIZE]
                      [--model MODEL]

PathologyGAN Encoder trainer.

optional arguments:
  -h, --help            show this help message and exit
  --epochs EPOCHS       Number epochs to run: default is 45 epochs.
  --batch_size BATCH_SIZE
                        Batch size, default size is 64.
  --model MODEL         Model name.
```

* Pathology GAN Encoder training example:
```
python3 run_pathgan_encoder.py 
```
