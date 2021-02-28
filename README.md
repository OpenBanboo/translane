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
        translane_demo.ipynb
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

Setup virtual environment for TransLane demo:
```
conda env create --name translane --file environment.txt
conda activate translane
pip install -r requirements.txt
```

Run DETR demo:
```
jupyter notebook
```
