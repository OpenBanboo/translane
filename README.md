# TransLane

TransLane is a light-weighted end-to-end model designed for mobile device real-time lane detection tasks.

The metrics evaluation is done on [TuSimple Lane Detection Challenge Dataset](https://github.com/TuSimple/tusimple-benchmark/tree/master/doc/lane_detection).
The skeleton Vision Transformer model is modified based on [DETR](https://github.com/facebookresearch/detr).
The lane shape prediction algorithm and loss function computation methods are modified based on [LSTR](https://github.com/liuruijin17/LSTR)
```
TuSimple/
    translane/
        README.md
        main.py
        evaluate.py
        scnn_sad_remaster <- This is the sub-repo for our PyTorch version of remastered SCNN and ENet-SAD models.
        ...
    LaneDetection/
        README.md
        label_data_0313.json
        label_data_0531.json
        label_data_0601.json
        test_label.json
        test_tasks_0627.json
        clips/
            0313-1
            0313-2
            0530
            0531
            0601
        
```

## Architecture
![TransLane](https://user-images.githubusercontent.com/14226287/109428185-c0254000-79aa-11eb-810e-632ebf8852a8.png)




### Setup virtual environment for TransLane demo:
```
conda env create --name translane --file environment.txt
conda activate translane
pip install -r requirements.txt
```

### Training on the network
```
# Start training using pretrained parameters saved at iteration 500000
python main.py translane -b 500000
```

### Testing on the network
```
# Testing TuSimple split using pretrained parameters saved at iteration 500000
python evaluate.py translane -b 500000 -m eval -s testing
```

### Run demo on customized images:
```
python evaluate.py translane -b 500000 -m customized -c ./ --debug
```
Note that, we've put 9 selected customized images captured from San Jose, California. 
The output of the model with predited lane marks can be found under ./detections folder.
(ex.)
![sample9](https://user-images.githubusercontent.com/14226287/109436557-e9f35c80-79d4-11eb-9889-4ed241cd779e.jpg)
