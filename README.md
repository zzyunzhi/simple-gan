Some code and a lot of hyperparameters are stolen from [CS294-158](https://github.com/rll/deepul) Fall 2020, UC Berkeley, with some justifications in comments 
and updates according to PyTorch 1.16 best practice. 

Installation

```
conda create --name simple-gan python=3.7.3
conda install pytorch torchvision -c pytorch
pip install easydict matplotlib tqdm
```

Code structure
```
def train():
    data_batches = get_data_batches()
    gen = build_generator_network()
    disc = build_discriminator_network()

    optimizer_g = build_generator_optimizer()
    optimizer_d = build_discriminator_optimizer()

    for epoch in n_epochs:
        # training (see train_one_epoch in utils.py)
        for batch in data_batches:
            real_data = batch
            fake_data = gen.generate_fake_data()

            gen_loss = compute_gen_loss(real_data, fake_data)
            disc_loss = compute_disc_loss(real_data, fake_data)
            
            optimizer_g.gradient_step(gen_loss)
            optimizer_d.gradient_step(disc_loss)

        # evaluation if any
        pass

    # training complete
    plot_loss_curve()
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
