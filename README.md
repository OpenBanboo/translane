# TransLane

[Tutorials](https://cs230-stanford.github.io)


We are happy to introduce some code examples that you can use for your CS230 projects. The code contains examples for TensorFlow and PyTorch, in vision and NLP. The structure of the repository is the following:

```
README.md
detr_demo.ipynb
pytorch/
    vision/
        README.md
    nlp/
        README.md
```

Setup virtual environment for DETR demo:
```
conda create --name translane
conda activate translane
conda install -c anaconda pillow
conda install pytorch==1.6.0 torchvision==0.7.0 -c pytorch
conda install matplotlib requests
```

Run DETR demo:
```
jupyter notebook
```
