# LDCformer
### LDCformer: Incorporating Learnable Descriptive Convolution to Vision Transformer for Face Anti-Spoofing (ICIP '23)

## Decoupled Learnable Descriptive Convolution (Decoupled-LDC)
![plot](figures/Dual_cross_ldc.png)

## Architecture of LDCformer
![plot](figures/framework3.png)

## Architecture of Decoupled-LDC Block
![plot](figures/D_LDC_Encoder.png)

## Requirements
```
numpy==1.23.3
pytz==2022.4
requests==2.28.1
scikit_learn==1.2.0
timm==0.6.7
torch==1.10.1
torchvision==0.11.2
```

## Training & Testing
Run `train.py` to train LDCformer

Run `test.py` to test LDCformer

## Citation

If you use the LDCformer/Decoupled-LDC, please cite the paper:
 
@inproceedings{huang2023ldcformer,
  title={LDCformer: Incorporating Learnable Descriptive Convolution to Vision Transformer for Face Anti-Spoofing},
  author={Huang, Pei-Kai and Chiang, Cheng-Hsuan and Chong, Jun-Xiong and Chen, Tzu-Hsien and Ni, Hui-Yu and Hsu, Chiou-Ting},
  booktitle={2023 IEEE International Conference on Image Processing (ICIP)},
  pages={121--125},
  year={2023},
  organization={IEEE}
}

 @inproceedings{huang2022learnable,
  title={Learnable Descriptive Convolutional Network for Face Anti-Spoofing},
  author={Huang, Pei-Kai and H.Y. Ni and Y.Q. Ni and C.T. Hsu},
  booktitle={BMVC},
  year={2022}
} 
