import numpy as np
import matplotlib.pyplot as plt
from easydict import EasyDict
import torch.utils.data as data
import torch
from mlp_gan import Generator, Discriminator
from utils import train_one_epoch, plot_loss_curve
import os


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)


def get_toy_dataset(n=20000):
    assert n % 2 == 0
    gaussian1 = np.random.normal(loc=-1, scale=0.25, size=(n // 2,))
    gaussian2 = np.random.normal(loc=0.5, scale=0.5, size=(n // 2,))
    data = (np.concatenate([gaussian1, gaussian2]) + 1).reshape([-1, 1])
    scaled_data = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)
    return 2 * scaled_data - 1


def process_batch_fn(batch):
    # no data processing
    return batch


def eval_one_epoch(*, gen, disc, device):
    """

    :param gen: Generator
    :param disc: Discriminator
    :param device: torch.device
    :return:
    """
    gen.eval()
    disc.eval()
    with torch.no_grad():
        noise = gen.sample_noise(batch_size=5000).to(device)
        fake = gen(noise)
        disc_in = (
            torch.from_numpy(np.linspace(-1, 1, num=1000))
            .float()
            .view(1000, 1)
            .to(device)
        )
        disc_out = disc(disc_in).flatten()

    fake = fake.flatten().to("cpu").numpy()
    disc_in = disc_in.flatten().to("cpu").numpy()
    disc_out = disc_out.flatten().to("cpu").numpy()

    return fake, disc_in, disc_out


def plot_eval(*, real, fake, disc_in, disc_out, epoch):
    plt.figure()
    plt.hist(real, bins=50, density=True, alpha=0.7, label="real")
    plt.hist(fake, bins=50, density=True, alpha=0.7, label="fake")

    plt.plot(disc_in, disc_out, label="discrim")
    plt.legend()
    plt.title(f"Epoch {epoch}")
    plt.savefig(os.path.join(DATA_DIR, f"train_minimal_epoch_{epoch}"))


def main():
    data = get_toy_dataset()
    train(train_data=data)


def train(train_data, use_asymm_gen_loss=True, use_gpu=False):
    """

    :param train_data: np.ndarray of shape (20000, 1)
    :param use_asymm_gen_loss: bool
    :param use_gpu: bool
    :return:
    """

    """ Build training configurations """

    hp = dict(n_epochs=20, batch_size=64, n_disc_updates=2)
    hp = EasyDict(hp)

    constant = dict(device=torch.device("cpu" if not use_gpu else "cuda:0"))
    constant = EasyDict(constant)
    if use_gpu:
        torch.cuda.set_device(constant.device)

    """ Build data loader and data processor function """

    train_loader = data.DataLoader(
        dataset=train_data, batch_size=hp.batch_size, shuffle=True
    )

    """ Build networks """

    gen = Generator().to(constant.device)
    disc = Discriminator().to(constant.device)

    """ Build optimizers """

    optimizer_g = torch.optim.Adam(gen.parameters(), lr=1e-4, betas=(0, 0.9))
    optimizer_d = torch.optim.Adam(disc.parameters(), lr=1e-4, betas=(0, 0.9))

    """ Build loss functions """

    def disc_loss_fn(real, fake):
        return (
            -disc(real.detach()).log().mean() - (1 - disc(fake.detach())).log().mean()
        )

    if use_asymm_gen_loss:

        def gen_loss_fn(real, fake):
            return -disc(fake).log().mean()

    else:

        def gen_loss_fn(real, fake):
            return (1 - disc(fake)).log().mean()

    """ Traning loop """

    history = dict(losses=[])
    for epoch in range(hp.n_epochs):

        losses_one_epoch = train_one_epoch(
            n_disc_updates=hp.n_disc_updates,
            batch_iterator=train_loader,
            process_batch_fn=process_batch_fn,
            gen=gen,
            disc=disc,
            optimizer_g=optimizer_g,
            optimizer_d=optimizer_d,
            gen_loss_fn=gen_loss_fn,
            disc_loss_fn=disc_loss_fn,
            device=constant.device,
            # max_n_iterations=1,  # uncomment this line if trying to debug
        )
        history["losses"].extend(losses_one_epoch)

        print(f"Epoch {epoch}: loss = {np.mean(losses_one_epoch)}")

        if epoch == 0 or epoch == hp.n_epochs - 1:
            fake, disc_in, disc_out = eval_one_epoch(
                gen=gen, disc=disc, device=constant.device
            )
            plot_eval(
                real=train_data,
                fake=fake,
                disc_in=disc_in,
                disc_out=disc_out,
                epoch=epoch,
            )

    history["losses"] = torch.stack(history["losses"]).to("cpu").numpy()

    plot_loss_curve(
        history["losses"],
        title="train_minimal",
        save_to=os.path.join(DATA_DIR, "train_minimal_loss"),
    )


if __name__ == "__main__":
    main()
