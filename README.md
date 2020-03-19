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
    <img src="https://github.com/AdalbertoCq/Learning-a-low-dimensional-manifold-of-realcancer-tissue/blob/master/demos/latent_space/NKI%2BVGH_real_latent_full.png" width="500">
 <img src="https://github.com/AdalbertoCq/Learning-a-low-dimensional-manifold-of-realcancer-tissue/blob/master/demos/latent_space/NKI%2BVGH_real_latent_linear_inter.png" width="500">
</p>

