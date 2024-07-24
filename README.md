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

## 7/23 update

