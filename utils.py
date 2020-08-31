import matplotlib.pyplot as plt


def plot_loss_curve(losses, title, save_to):
    plt.figure()
    plt.plot(losses)
    plt.xlabel("Training Iteration")
    plt.ylabel("Loss")
    plt.title(title)
    plt.savefig(save_to)


def train_one_epoch(
    *,
    n_disc_updates,
    batch_iterator,
    process_batch_fn,
    gen,
    disc,
    optimizer_g,
    optimizer_d,
    gen_loss_fn,
    disc_loss_fn,
    device,
    scheduler_g=None,
    scheduler_d=None,
    max_n_iterations=None,
):
    gen.train()
    disc.train()

    losses = []

    for itr, batch in enumerate(batch_iterator):
        batch = process_batch_fn(batch.float().to(device))
        current_batch_size = batch.shape[0]

        """ Train Discriminator """

        # fake examples
        noise = gen.sample_noise(batch_size=current_batch_size).to(device)
        gen_out = gen(noise)

        disc_loss = disc_loss_fn(real=batch, fake=gen_out)

        optimizer_d.zero_grad()
        disc_loss.backward()
        optimizer_d.step()

        losses.append(disc_loss.detach())

        if itr % n_disc_updates == 0:

            """ Train Generator """

            noise = gen.sample_noise(batch_size=current_batch_size).to(device)
            gen_out = gen(noise)
            gen_loss = gen_loss_fn(real=None, fake=gen_out)

            optimizer_g.zero_grad()
            gen_loss.backward()
            optimizer_g.step()

        if scheduler_d is not None:
            scheduler_d.step()
        if scheduler_g is not None:
            scheduler_g.step()

        if itr == max_n_iterations:
            break

    return losses
