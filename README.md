A lot of code is adapted from CS294-158 Fall 2020, UC Berkeley, with some justifications in comments 
and updates according to with PyTorch 1.14 best practice. 

Installation

```
conda create --name simple-gan python=3.7.3
conda install pytorch torchvision -c pytorch
pip install easydict matplotlib tqdm
```

To run a minimal GAN example
```
conda activate simple-gan
cd PATH_TO_REPO
python train_minimal.py
```

You will see loss curves and data visualization saved under `PATH_TO_REPO/data`. 

To run Spatial Norm GAN for Cifar10
```
conda activate simple-gan
cd PATH_TO_REPO
python traincifar10.py --use_gpu
```
You can remove the `--use_gpu` flag if no GPU is available. 
Loss curves and datavisualization will be saved under `PATH_TO_REPO/data`. 
