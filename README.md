# CS766Spring2021

### Procedures
this project aims to propose a CartoGAN model for generating electronic maps that can be in multiple types and more realistic and aesthetic. In order to accomplish this goal, the steps below are required: 
1.	Collect multiple types of electronic maps, e.g., Google Maps, OpenStreetMap, Baidu Maps, etc. 
2.	Search for usable HSR remote sensing images corresponded to those electronic maps, and use multiple bands (e.g., infrared) in order to improve the model’s ability in feature recognition. 
3.	Construct a classifier similar with (Li et al., 2020) to help the model learn the differences among multiple types of electronic maps so that the model can determine whether the generated electronic map belongs to the correct type during the training process.
4.	Incorporate the concept of “render matrix” and think about its structure with less memory used. 
5.	Build effective architectures of the generator and the discriminator. 
6.	Consider suitable loss functions for the model, e.g., reconstruction loss (for pixel-wise accuracy), a style loss (to reduce high frequency artifacts), and the GAN loss (a feature-wise learnt similarity metric or content loss). 
7.	Determine the evaluation metrics of the model, e.g., Kernel Maximum Mean Discrepancy (Kernel MMD), Fréchet Inception Distance (FID), Mode Score, Inception Score, Pixel-Level Translation Accuracy, etc.
8.	Compare the results from our model with the ones from other state-of-the-art GANs.  

### Tentative Schedule 
Tasks	Expected due date
Complete research references	Feb 27
Search for suitable datasets 	March 6
Implement our CartoGAN model by referencing the existing methods  	March 20
Draft the mid-term report	March 24
Try to improve the accuracy and aesthetics of our results	April 10
Compare our model with other state-of-the-art models 	April 17
Prepare for the final presentation	April 23
Finish refining the webpage	May 5

### Major References:
1. MapGAN: An Intelligent Generation Model for Network Tile Maps
2. GeoGAN - A Conditional GAN with Reconstruction and Style Loss to Generate Standard Layer of Maps from Satellite Images
3. Image-to-Image Translation with Conditional Adversarial Nets: https://phillipi.github.io/pix2pix/
