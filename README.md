# watermelen project

## use learned-method to generate POH(Phase-only Hologram) in the SLM plain with the RGB-D information at the diffraction plain as the input

## including the reproduction of the paper 'Band-limited Angular Spectrum Method for Numerical Simulation of Free-Space Propogation in Far and Near Fields'

- Instead of using the numerical integeration result as the reference signal to measure the SNR, the extended ASM result is used.
- the circular convolution was linearized, which provides a satisfying work.

## about the models

- all with square filters, with double residual blocks in each conv block:

  - [output\models\watermelon_v1.pth](output\models\watermelon_v1.pth) is the model trained with 100 samples(not from the train dataset) with the lr = 1e-3, 10 epoches, and the batch size = 4.

  - [output\models\watermelon_v2_trainedwith3800samples.pth](output\models\watermelon_v2_trainedwith3800samples.pth) is the model trained with 3800 samples with the lr = 1e-3, 10 epoches, and the batch size = 4.

  - [output\models\watermelon_v3_lr_1e-4_.pth](output\models\watermelon_v3_lr_1e-4_.pth) is the model trained with 3800 samples with the lr = 1e-4, 10 epoches, and the batch size = 4.

## 7/27 update

- fix the bug in watermelon_v3:
  - the method of the propagater is wrong, which is now corrected from propagate_P2IP to propagate_P2AP, however the trained model is wasted...

- modify the the dataloader:
  - modify the data loader for the perceptual loss. Now it loads the **amp**, phs and depth to the trainer of the erceptual model, regardless of the previous version which load the **square root of the image**, phs and depth to the trainer.

- create model watermelon_v4:
  - use UNet_imgDepth2AP_v1 instead of UNet_imgDepth2AP_v2, which abondons the redundant process of taking the square root of the rgb channels.

  - use the propagate_P2AAP instead of propagate_P2IP, which now keeps the amp and phs information without filtering. Now it returns 9 channels, which are the amp with diffraction limitation(4f)(3 channels), the amp without diffraction limitation(3 channels), and the phs(3 channels).

  - now the loss is composed of:
    - the l2 loss of the square of the diffraction limited(4f) amp and the ground truth rgb channels.

    - the l2 loss of the depth_hat inferred by the pretrained model and the ground truth depth.  
