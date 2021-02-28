# TransLane

[Tutorials](https://cs230-stanford.github.io)


TransLane is a light-weighted end-to-end model designed for mobile device real-time lane detection tasks.

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
