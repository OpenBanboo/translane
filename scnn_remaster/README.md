# SCNN lane detection in Pytorch

SCNN is a lane detection algorithm, proposed in ['Spatial As Deep: Spatial CNN for Traffic Scene Understanding'](https://arxiv.org/abs/1712.06080). The [official implementation](<https://github.com/XingangPan/SCNN>) is in lua torch.

This repository contains a re-master version in Pytorch.



### Updates

- 03-14-2021: Remastered SCNN in PyTorch.

<br/>

## Data preparation

### Tusimple
The dataset is available in [here](https://github.com/TuSimple/tusimple-benchmark/issues/3). Please download and unzip the files in one folder, which later is represented as `Tusimple_path`. Then modify the path of `Tusimple_path` in `config.py`.
```
Tusimple_path
├── clips
├── label_data_0313.json
├── label_data_0531.json
├── label_data_0601.json
└── test_label.json
```

**Note:  seg\_label images and gt.txt, as in CULane dataset format,  will be generated the first time `Tusimple` object is instantiated. It may take some extra time.**



<br/>

## Pre-trained Model

* Trained model on Tusimple can be downloaded [here](https://drive.google.com/open?id=1IwEenTekMt-t6Yr5WJU9_kv4d_Pegd_Q). Its configure file is in `exp0`.

| Accuracy | FP   | FN   |
| -------- | ---- | ---- |
| 94.16%   |0.0735|0.0825|

**Note**:`torch.utils.serialization` is obsolete in Pytorch 1.0+. You can directly download **the converted model [here](https://drive.google.com/open?id=1bBdN3yhoOQBC9pRtBUxzeRrKJdF7uVTJ)**.


<br/>


## Demo Test

For single image demo test:

```shell
python demo_test.py   -i demo/demo.jpg 
                      -w experiments/vgg_SCNN_DULR_w9/vgg_SCNN_DULR_w9.pth 
                      [--visualize / -v]
```

![](demo/demo_result.jpg "demo_result")



<br/>

## Train 

1. Specify an experiment directory, e.g. `experiments/exp0`. 

2. Modify the hyperparameters in `experiments/exp0/cfg.json`.

3. Start training:

   ```shell
   python train.py --exp_dir ./experiments/exp0 [--resume/-r]
   ```

4. Monitor on tensorboard:

   ```bash
   tensorboard --logdir='experiments/exp0'
   ```

**Note**


- My model is trained with `torch.nn.DataParallel`. Modify it according to your hardware configuration.
- Using the backbone is vgg16 from torchvision. Several modifications are done to the torchvision model according to paper, i.e., i). dilation of last three conv layer is changed to 2, ii). last two maxpooling layer is removed.



<br/>

## Evaluation

* Tusimple Evaluation code is ported from [tusimple repo](https://github.com/TuSimple/tusimple-benchmark/blob/master/evaluate/lane.py).

  ```Shell
  python test_tusimple.py --exp_dir ./experiments/exp0
  ```




## Acknowledgement

This repos is build based on [official implementation](<https://github.com/XingangPan/SCNN>) and [SCNN_Pytorch](<https://github.com/harryhan618/SCNN_Pytorch/blob/master>).
