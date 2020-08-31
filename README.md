A lot of code is adapted from CS294-158 Fall 2020, UC Berkeley, with some justifications in comments 
and updates according to with PyTorch 1.14 best practice. 

Installation

```
conda create --name simple-gan python=3.7.3
conda install pytorch torchvision -c pytorch
pip install easydict matplotlib
```

To run a minimal GAN example
```
conda activate simple-gan
cd PATH_TO_REPO
python train_minimal.py
```

You will see loss curves and data visualization saved in `PATH_TO_REPO/data` folder. 