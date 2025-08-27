 # CMTNet

This repository is an official PyTorch implementation of our paper"Enhancing Real-Time Semantic Segmentation: A Dual-Branch Architecture with Mamba-Transformer Synergy".

[Code](https://github.com/StarTwinkled/CMTNet)



## Installation

```
cuda == 11.8
Python == 3.8
Pytorch == 1.13.0+cu118

# clone this repository
git clone https://github.com/StarTwinkled/CMTNet.git
```



## Datasets

We used Cityscapes dataset and CamVid dataset to train our model.  

- You can download cityscapes dataset from [here](https://www.cityscapes-dataset.com/). 

Note: please download leftImg8bit_trainvaltest.zip(11GB) and gtFine_trainvaltest(241MB). 

The Cityscapes dataset scripts for inspection, preparation, and evaluation can download from [here](https://github.com/mcordts/cityscapesScripts).

- You can download camvid dataset from [here](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/).

The folds of your datasets need satisfy the following structures:

```
├── dataset  					# contains all datasets for the project
|  └── cityscapes 				#  cityscapes dataset
|  |  └── gtCoarse  		
|  |  └── gtFine 			
|  |  └── leftImg8bit 		
|  |  └── cityscapes_test_list.txt
|  |  └── cityscapes_train_list.txt
|  |  └── cityscapes_trainval_list.txt
|  |  └── cityscapes_val_list.txt
|  |  └── cityscapesscripts 	#  cityscapes dataset label convert scripts！
|  └── camvid 					#  camvid dataset 
|  |  └── test
|  |  └── testannot
|  |  └── train
|  |  └── trainannot
|  |  └── val
|  |  └── valannot
|  |  └── camvid_test_list.txt
|  |  └── camvid_train_list.txt
|  |  └── camvid_trainval_list.txt
|  |  └── camvid_val_list.txt
|  └── inform 	
|  |  └── camvid_inform.pkl
|  |  └── cityscapes_inform.pkl
|  └── camvid.py
|  └── cityscapes.py 

```

## Test

```
# cityscapes
python test.py --dataset cityscapes --checkpoint ./checkpoint/cityscapes/CMTNetbs6gpu1_train/model_1000.pth

# camvid
python test.py --dataset camvid --checkpoint ./checkpoint/camvid/CMTNetbs6gpu1_train/model_995.pth
```

## Predict
only for cityscapes dataset
```
python predict.py --dataset cityscapes 
```

## Results

- Please refer to our article for more details.

| Methods |  Dataset   | Input Size | mIoU(%) |
| :-----: | :--------: | :--------: | :-----: |
| CMTNet  | Cityscapes |  512x1024  |  70.7   |
| CMTNet  |   CamVid   |  360x480   |  71.4  |



