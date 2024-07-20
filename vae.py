from typing import Tuple, Optional
import utils
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class VAE(nn.Module):
    """
    Base class of a variational auto-encoder (VAE).
    Consists of an encoder and decoder nn.Module.
    The encoder must return a tuple representing the latent mean and std. dev vector.
    Some code is taken from:  https://github.com/PyTorchLightning/lightning-bolts/blob/5bfb846ba86f8cd651165ab3a3b77bcac655d21b/pl_bolts/models/autoencoders/basic_vae/basic_vae_module.py#L18
    """

    def __init__(
            self,
            encoder: nn.Module,
            decoder: nn.Module,
            latent_dim: int,
            # We set this to a very small value in this exercise since we use only two dims for educational purposes
            # Because were in such low dimensions, the model does not have much capacity to output deviations from a standard gaussin
            # If we set this to higher values, the encoder keeps outputting mean vectors of 0, s.t. no differences in the images are learned
            kl_weight: float = 0.0005
    ):
        """
        Initializes the VAE model. Mainly passes the encoder and decoder architectures.
        :param encoder: The NN mapping data to a latent mean and log. variance vector.
        :param decoder: The NN mapping a sample drawn from a Gaussian distribution using the mean and log. variance
        :param latent_dim: The dimensionality of the latent vectors outputted by the encoder (mean and log var).
        :param kl_weight: The weigh of the KL divergence for the ELBO loss.
        """
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight

    def forward(self, x_input: Tensor):
        """
        Encodes samples in the x_input Tensor to latent vectors depicting the mean and log variance of the gaussian
        distribution.
        Subsequently creates samples from the parameterized distribution and uses the decoder to reconstruct the input
        :param x_input: (Tensor) A torch tensor representing the data batch. The shape must be [B x ...], where B is
        the batch size.
        :return: Returns a Tuple of the latent vectors, the reconstructed inputs and the encoder and prior distribution
        """
        mu, log_var = self.encoder(x_input)
        p, q, z = self.sample_forward(mu, log_var)
        x_reconstructed = self.decoder(z)
        return z, x_reconstructed, p, q

    def sample_forward(self, mu, log_var):
        """
        Performs sampling from the encoder distribution in the forward step.
        :param mu: A batch of mean vectors of the encoder distribution
        :param log_var: A batch of log variance vectors of the encoder distribution.
        """
        # Compute the standard deviation from the log variance.
        std = torch.exp(log_var / 2)
        # Create the prior distribution N(0, 1)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        # Create the distribution approximated by the encoder
        q = torch.distributions.Normal(mu, std)
        # Sample latent vectors from the distribution of the encoder
        z = q.rsample()
        return p, q, z

    def calc_loss(self, x_input: Tensor):
        z, x_reconstruct, p, q = self.forward(x_input)
        reconstruction_loss = F.mse_loss(x_input, x_reconstruct, reduction="mean")
        kld = self.kl_divergence(p, q, z)
        return reconstruction_loss + kld

    def train_step(self, x_input: Tensor, optimizer: torch.optim.Optimizer) -> Tuple[float, float, float]:
        optimizer.zero_grad()
        # Get latent vectors, the reconstructed data sample, also distributions for the prior and the encoder dist.
        z, x_reconstruct, p, q = self.forward(x_input)

        # Compute the loss reconstruction loss
        reconstruction_loss = F.mse_loss(x_input, x_reconstruct, reduction="mean")
        kld = self.kl_divergence(p, q, z)
        total_loss = reconstruction_loss + kld

        # Backwards pass; Calculate gradients and update the model with the optimizer
        total_loss.backward()
        optimizer.step()

        # Return the losses as floats
        return float(total_loss), float(reconstruction_loss), float(kld)

    def kl_divergence(self, p, q, z):
        """
        Computes the scaled KL Divergence between the prior and approximated distribution of the encoder.
        """
        # Compute log probs for the sampled latent vector z values
        log_qz = q.log_prob(z)
        log_pz = p.log_prob(z)

        # Compute the KLD using the expectation over z values
        kl = log_qz - log_pz
        kl = kl.mean()

        # Scale the KLD with a hyperparameter
        kl *= self.kl_weight
        return kl

    def encode_data_to_sampled_latent_vec(self, x_input) -> Tensor:
        """
        Encodes an input data batch to latent vectors using the forward pass.
        First, the parameters of the Gaussian distribution are computed with the encoder NN.
        Subsequently, we sample from the parameterized Gaussian distribution.
        """
        mu, log_var = self.encoder(x_input)
        p, q, z = self.sample_forward(mu, log_var)
        return z


    def sample_from_prior(self, num_samples: int, device: torch.device):
        """
        Samples data samples from the prior distribution.
        :param num_samples: (int) The number of data vectors which should be drawn.
        :param device: (torch.device) The torch device on which the model is located (e.g. CPU or GPU)
        :return: A batch of the sampled data.
        """
        z = torch.randn(num_samples, self.latent_dim).to(device)
        samples = self.decoder(z)
        return samples

    @torch.no_grad()
    def val_step(self, x_input: Tensor) -> Tuple[float, float, float]:
        # Get latent vectors, the reconstructed data sample, also distributions for the prior and the encoder dist.

        z, x_reconstruct, p, q = self.forward(x_input)

        # Compute the loss reconstruction loss
        reconstruction_loss = F.mse_loss(x_input, x_reconstruct, reduction="mean")
        kld = self.kl_divergence(p, q, z)
        total_loss = reconstruction_loss + kld

        return (float(total_loss), float(reconstruction_loss), float(kld))


class MNISTEncoder(nn.Module):
    def __init__(
            self,
            input_dim: int = 784,
            latent_dim: int = 2,
            hidden_layer_dims=(512, 256 ),
    ):
        super(MNISTEncoder, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Setup input + hidden layers
        layer_dims = (input_dim, ) + hidden_layer_dims
        fc_layers = []
        for layer_idx, dim in enumerate(layer_dims):
            if layer_idx == 0:
                continue
            last_dim = layer_dims[layer_idx - 1]
            fc_layers.append(nn.Linear(in_features=last_dim, out_features=dim))
            fc_layers.append(nn.ReLU())

        self.hidden_layers = nn.Sequential(*fc_layers)

        # Setup two output layers
        self.out_mu = nn.Linear(hidden_layer_dims[-1], latent_dim)
        self.out_var = nn.Linear(hidden_layer_dims[-1], latent_dim)

    def forward(self, x_input: Tensor) -> Tuple[Tensor, Tensor]:
        hidden_act = self.hidden_layers(x_input)
        return self.out_mu(hidden_act), self.out_var(hidden_act)


class MNISTDecoder(nn.Module):
    def __init__(
            self,
            output_dim: int = 784,
            latent_dim: int = 2,
            hidden_layer_dims=(256, 512),
            output_act_fn: Optional[nn.Module] = nn.Tanh()
    ):
        super(MNISTDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim

        # Setup input + hidden layers
        layer_dims = (latent_dim, ) + hidden_layer_dims
        fc_layers = []
        for layer_idx, dim in enumerate(layer_dims):
            if layer_idx == 0:
                continue
            last_dim = layer_dims[layer_idx - 1]
            fc_layers.append(nn.Linear(in_features=last_dim, out_features=dim))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.BatchNorm1d(num_features=dim))
        self.hidden_layers = nn.Sequential(*fc_layers)

        # Setup output activation function
        self.out_layer = nn.Linear(layer_dims[-1], output_dim)
        self.out_act_fn = output_act_fn

    def forward(self, latent_dim_z: Tensor) -> Tensor:
        hidden_act = self.hidden_layers(latent_dim_z)
        out = self.out_layer(hidden_act)
        if self.out_act_fn:
           out = self.out_act_fn(out)
        return out


if __name__ == "__main__":
    # Test the VAE architecture using a standard batch of random values.

    torch.manual_seed(0)

    epochs = 250

    batch_size = 8
    latent_dim = 8 #2
    mnist_dim = 784

    DATA_PATH = Path(".") / "data"
    print("Loading MNIST data...")

    mnist_train_set, mnist_dev_set = utils.get_mnist_train_dev_loaders(DATA_PATH, batch_size=batch_size, flatten_img=True)

    print("...MNIST data loaded")

    #random_batch = torch.randn((batch_size, 784))
    #random_batch = torch.randn((batch_size, 784))
    dataiter = iter(mnist_train_set)
    batch, labels = next(dataiter)

    print(batch.shape)
    print(labels)

    encoder = MNISTEncoder(mnist_dim, latent_dim)
    decoder = MNISTDecoder(mnist_dim, latent_dim)
    vae = VAE(encoder, decoder, latent_dim)

    #random_reconstructed, mu, log_var = vae(random_batch)
    #z, random_reconstructed, mu, log_var = vae(random_batch)
    #assert random_reconstructed.shape == random_batch.shape

    # Check the loss method
    #torch_loss_vals = vae.calc_loss(random_batch)
    #print(torch_loss_vals)
    #exit()
    #torch_loss_vals = vae.loss(random_batch, random_reconstructed, mu, log_var)
    #for loss_val in torch_loss_vals:
    #    assert not torch.isnan(loss_val)

    # Check the train and val step methods.

    optimizer = torch.optim.Adam(vae.parameters())

    for i in range(epochs):
        train_total_loss, train_reconstruction_loss, train_kld = vae.train_step(batch, optimizer)
        val_total_loss, val_reconstruction_loss, val_kld = vae.val_step(batch)

        print(train_total_loss, val_total_loss)

    # Check random samples
    #num_samples = 4
    #random_samples = vae.sample_from_prior(num_samples, torch.device("cpu"))
    #assert random_samples.shape == (4, mnist_dim)

    torch.save(vae.state_dict(), Path(".") / "vae-model-8.pth")
