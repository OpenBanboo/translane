# TransLane

[Tutorials](https://cs230-stanford.github.io)


We are happy to introduce some code examples that you can use for your CS230 projects. The code contains examples for TensorFlow and PyTorch, in vision and NLP. The structure of the repository is the following:

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
