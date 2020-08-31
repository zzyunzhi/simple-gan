import numpy as np
import matplotlib.pyplot as plt
from easydict import EasyDict
import torch.utils.data as data
import torch
from sn_gan import Generator, Discriminator
from utils import train_one_epoch, plot_loss_curve
import os
import torchvision
from tqdm import tqdm


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)


def get_cifar10_dataset():
    data = torchvision.datasets.CIFAR10(DATA_DIR, transform=torchvision.transforms.ToTensor(),
                                              download=True, train=True)
    data = data.data.transpose((0, 3, 1, 2)) / 255.0
    return data


def process_batch_fn(batch):
    # pixel in range [0, 1] -> shift to [-1, 1]
    return batch * 2 - 1


def reverse_process_batch_fn(batch):
    # [-1, 1] -> [0, 1]
    return (batch + 1) * 0.5


def eval_one_epoch(*, gen, disc, device):
    gen.eval()
    disc.eval()
    with torch.no_grad():
        noise = gen.sample_noise(batch_size=100).to(device)
        fake = gen(noise)
        fake = reverse_process_batch_fn(fake).to('cpu').numpy()

    return fake


def plot_eval(*, fake, epoch):
    samples = torch.FloatTensor(fake)
    grid_img = torchvision.utils.make_grid(samples, nrow=10)
    plt.figure()
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.axis('off')

    plt.title(f"Epoch {epoch}")
    plt.savefig(os.path.join(DATA_DIR, f"train_cifar10_epoch_{epoch}"))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_gpu', action='store_true')
    args = parser.parse_args()

    data = get_cifar10_dataset()
    train(train_data=data, use_gpu=args.use_gpu)


def train(train_data, use_gpu=False):
    """

    :param train_data: np.array of shape (None, 3, 32, 32) with values in [0, 1]
    :param use_gpu:
    :return:
    """

    """ Build training configurations """

    hp = dict(
        n_iterations=25000,
        batch_size=256,
        n_disc_updates=5,
        lmbda=10,
    )
    hp = EasyDict(hp)

    constant = dict(device=torch.device("cpu" if not use_gpu else "cuda:0"))
    constant = EasyDict(constant)
    if use_gpu:
        torch.cuda.set_device(constant.device)

    """ Build data loader and data processor function """

    train_loader = data.DataLoader(dataset=train_data, batch_size=hp.batch_size, shuffle=True)
    n_batches = len(train_loader)
    hp.n_epochs = hp.n_iterations // n_batches
    hp.n_iterations = hp.n_epochs * n_batches
    print('n_epochs', hp.n_epochs, 'n_iterations', hp.n_iterations)

    """ Build networks """

    gen = Generator().to(constant.device)
    disc = Discriminator().to(constant.device)

    """ Build optimizers """

    optimizer_g = torch.optim.Adam(gen.parameters(), lr=2e-4, betas=(0, 0.9))
    optimizer_d = torch.optim.Adam(disc.parameters(), lr=2e-4, betas=(0, 0.9))

    """ Build loss functions """

    def disc_loss_fn(real, fake):
        current_batch_size = real.shape[0]
        real, fake = real.detach(), fake.detach()
        eps = torch.randn(current_batch_size, 1, 1, 1).to(constant.device)
        x_hat = (eps * real + (1 - eps) * fake).requires_grad_()

        disc_out = disc(x_hat)
        original_disc_loss = disc_out.mean() - disc(real).mean()

        grad, = torch.autograd.grad(
            outputs=[disc_out.mean(), ],
            inputs=x_hat,
            create_graph=True,
            retain_graph=True,
        )
        grad_penalty = (grad.norm() - 1).square()
        return original_disc_loss + hp.lmbda * grad_penalty

    def gen_loss_fn(real, fake):
        return -disc(fake).log().mean()

    """ Build learning rate schedulers """

    max_n_iterations = max(hp.n_iterations, 25000)
    scheduler_g = torch.optim.lr_scheduler.LambdaLR(
        optimizer=optimizer_g,
        lr_lambda=lambda itr: (max_n_iterations - itr) / max_n_iterations,
        last_epoch=-1
    )
    scheduler_d = torch.optim.lr_scheduler.LambdaLR(
        optimizer=optimizer_d,
        lr_lambda=lambda itr: (max_n_iterations - itr) / max_n_iterations,
        last_epoch=-1,
    )

    """ Training loop """

    history = dict(
        losses=[],
    )

    for epoch in tqdm(range(hp.n_epochs)):
        losses_one_epoch = train_one_epoch(
            n_disc_updates=hp.n_disc_updates,
            batch_iterator=train_loader,
            process_batch_fn=process_batch_fn,
            gen=gen, disc=disc, optimizer_g=optimizer_g, optimizer_d=optimizer_d,
            gen_loss_fn=gen_loss_fn,
            disc_loss_fn=disc_loss_fn,
            device=constant.device,
            scheduler_g=scheduler_g, scheduler_d=scheduler_d,
            # max_n_iterations=1,  # debug
        )
        history['losses'].extend(losses_one_epoch)

        print(f"Epoch {epoch}: loss = {np.mean(losses_one_epoch)}")

        if epoch == hp.n_epochs - 1:
            fake = eval_one_epoch(gen=gen, disc=disc, device=constant.device)
            plot_eval(fake=fake, epoch=epoch)

    history['losses'] = torch.stack(history['losses']).to('cpu').numpy()

    plot_loss_curve(
        history["losses"],
        title="train_cifar10",
        save_to=os.path.join(DATA_DIR, "train_cifar10_loss"),
    )


if __name__ == "__main__":
    main()