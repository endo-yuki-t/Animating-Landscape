# Animating Landscape

+-----------------------------------------------------------  
|Animating Landscape:   
|Self-Supervised Learning of Decoupled Motion and Appearance for Single-Image Video Synthesis  
|Project page: http://www.cgg.cs.tsukuba.ac.jp/~endo/projects/AnimatingLandscape/  
<img src="https://user-images.githubusercontent.com/56748497/67206735-34fa7880-f44d-11e9-85d3-790c25fd199a.gif" width="320px">  
+-----------------------------------------------------------  

This repository contains source codes of the following paper:  

>Yuki Endo, Yoshihiro Kanamori, Shigeru Kuriyama:   
>"Animating Landscape: Self-Supervised Learning of Decoupled Motion and Appearance for Single-Image Video Synthesis,"   
>ACM Transactions on Graphics (Proc. of SIGGRAPH Asia 2019), 38, 6, pp.175:1-175:19, November 2019.   

## Dependencies  
1. Python (we used version 2.7.12)  
2. PyTorch (we used version 0.4.0)  
3. OpenCV (we used version 2.4.13)  
4. scikit-learn (we used version 0.19.0)

The other dependencies for the above libraries are also needed. It might work with other versions as well.   

## Animating landscape image
  
Download [the pretrained models](http://www.cgg.cs.tsukuba.ac.jp/~endo/projects/AnimatingLandscape/)([mirror](https://drive.google.com/file/d/1SZCygpJOqLFdJR4fejqIU3GRyVCBYGue/view?usp=sharing)), put them into the models directory, and run test.py by specifying an input image and an output directory, for example, 
```
python test.py --gpu 0 -i ./inputs/1.png -o ./outputs  
```
Three videos (looped motion, flow field, and final result) are generated in the output directory. Output videos might differ according to latent codes randomly sampled every time you run the code.   

You can also specify latent codes manually from the pre-trained codebook using simple scalar values for motion (-mz) and appearance (-az) in [0,1], for example,   
```
python test.py --gpu 0 -i ./inputs/1.png -o ./outputs -mz 0.9 -az 0.1  
```

## Training new models  
Run train.py by specifying a dataset directory and a training mode.   
```
python train.py --gpu 0 --indir ./training_data/motion --mode motion  
```
Trained models are saved in the models directory.   

Fore more optional arguments, run each code with --help option.   

## Citation
Please cite our paper if you found the code useful:
```
@article{endo2019animatinglandscape,
  title = {Animating Landscape: Self-Supervised Learning of Decoupled Motion and Appearance for Single-Image Video Synthesis},
  author = {Yuki Endo and Yoshihiro Kanamori and Shigeru Kuriyama},
  journal = {ACM Transactions on Graphics (Proceedings of ACM SIGGRAPH Asia 2019)},
  year = {2019},
  volume = 38,
  number = 6,
  pages = {175:1--175:19}
}
```

## Acknowledgements
This code borrows the encoder code from [BicycleGAN](https://github.com/junyanz/BicycleGAN) and the Instance Normalization code from [fast-neural-style](https://github.com/abhiskk/fast-neural-style). 