from .mcmc import McmcNuts
import torch.nn as nn
import pyro.distributions as dist
import pyro.distributions.transforms as T
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from typing import Dict, Any, Tuple, Optional
from ..core.potential import Potential
from ..core.protocol import Protocol
from ..core.loss import Loss

class ConditionalFlow(McmcNuts, nn.Module):
    """Initial condition generator using a Conditional Normalizing Flow trained on MCMC samples."""

    def __init__(
        self,
        dt: float,
        gamma: float,
        mass: float,
        device: torch.device,
        spatial_dimensions: int = 1,
        time_steps: int = 1000,
        beta: float = 1.0,
        starting_bounds: Optional[torch.Tensor] = None,
        samples_per_well: Optional[int] = None,
        num_samples: int = 5000,
        chains_per_well: int = 1,
        warmup_ratio: float = 0.1,
        min_neff: Optional[float] = None,
        run_every_epoch: bool = False,
        flow_layers: int = 4,
        flow_epochs: int = 300,
        flow_batch_size: int = 256,
        flow_training_samples_per_well: int = 500
    ) -> None:
        """Initializes the ConditionalFlow generator.

        Args:
            dt (float): Time step.
            gamma (float): Friction coefficient.
            mass (float): Particle mass.
            device (torch.device): Torch device.
            spatial_dimensions (int): Number of spatial dimensions.
            time_steps (int): Number of time steps.
            beta (float): 1/kT
            starting_bounds (Optional[torch.Tensor]): Bounds for starting positions. Defaults to global bounds if None.
            samples_per_well (int): Samples per well.
            num_samples (int): Total samples if not per well.
            chains_per_well (int): Chains per well.
            warmup_ratio (float): Ratio of warmup steps.
            min_neff (float): Minimum effective sample size.
            run_every_epoch (bool): Whether to run sampling every epoch.
            flow_layers (int): Number of flow layers.
            flow_epochs (int): Number of training epochs for the flow.
            flow_batch_size (int): Batch size for flow training.
            flow_training_samples_per_well (int): Samples per well for training the flow.
        """
        nn.Module.__init__(self)
        super().__init__(
            dt=dt,
            gamma=gamma,
            mass=mass,
            device=device,
            spatial_dimensions=spatial_dimensions,
            time_steps=time_steps,
            beta=beta,
            starting_bounds=starting_bounds,
            samples_per_well=samples_per_well,
            num_samples=num_samples,
            chains_per_well=chains_per_well,
            warmup_ratio=warmup_ratio,
            min_neff=min_neff,
            run_every_epoch=run_every_epoch
        )
        
        if self.spatial_dimensions > 8:
            print('WARNING: when spatial dimensions are greater than 8 the number of samples is forced to 8 * samples_per_well and is taken randomly from the global bounds. You are NOT guaranteed to have samples from each well each epoch nor the normalizing flow to properly learn the entire bitstring space.')
            self.force_random = True
        else:
            self.force_random = False

        self.context_dim = self.spatial_dimensions

        self.original_bounds = self.starting_bounds.clone()

        transforms = []
        if flow_layers < 2:
            raise ValueError("Flow layers must be at least 2 and at least 4 is highly recommended")
        for _ in range(flow_layers):
            c1 = T.conditional_spline(
                self.spatial_dimensions,
                context_dim=self.context_dim, # number of bits in the bitstring
                count_bins=16, #number of bins in the spline
                bound=3.0 #since we are flowing from a N(0, 1) this is ~99.7% of the distribution
            ).to(device)

            transforms.append(c1)
            transforms.append(T.Permute(torch.randperm(self.spatial_dimensions, device=device)))

        self.base_dist = dist.Normal(torch.zeros(self.spatial_dimensions, device=device),
                                     torch.ones(self.spatial_dimensions, device=device))

        self.flow_dist = dist.ConditionalTransformedDistribution(self.base_dist, transforms)
        self.flow_modules = nn.ModuleList([t for t in transforms if isinstance(t, nn.Module)])

        self.is_trained = False

        # saving the mean and std for standardization in the model for saving and loading
        self.register_buffer('data_mean', torch.zeros(self.spatial_dimensions))
        self.register_buffer('data_std', torch.ones(self.spatial_dimensions))

        self.flow_epochs = flow_epochs
        self.flow_batch_size = flow_batch_size
        self.flow_training_well_count = 2**self.spatial_dimensions
        self.flow_training_samples_per_well = flow_training_samples_per_well

        self.hparams.update({
            'force_random': self.force_random,
            'context_dim': self.context_dim,
            'flow_epochs': self.flow_epochs,
            'flow_batch_size': self.flow_batch_size,
            'flow_training_well_count': self.flow_training_well_count,
            'flow_training_samples_per_well': self.flow_training_samples_per_well
        })

    def set_bounds_from_bits(self, target_bitstring: torch.Tensor, loss: Loss) -> None:
        """Updates sampling bounds to target a specific bitstring well.

        Args:
            target_bitstring: Tensor representing the target bitstring (e.g., [0, 1]).
            loss: Loss object containing midpoint information.
        """
        if not hasattr(loss, 'midpoints'):
            raise RuntimeError("Loss object must have .midpoints to define wells.")

        midpoints = loss.midpoints
        global_bounds = self.original_bounds

        new_bounds = []

        for dim_idx in range(self.spatial_dimensions):
            bit = target_bitstring[dim_idx]
            low, high = global_bounds[dim_idx]
            mid = midpoints[dim_idx]

            if bit == 0:
                new_bounds.append([low, mid])
            elif bit == 1:
                new_bounds.append([mid, high])
            else:
                raise ValueError(f"Bit must be 0 or 1, got {bit}")

        # update the bounds used by the parent class
        self.starting_bounds = torch.tensor(new_bounds, device=self.device)

        # runs the parent class in global mode on this subset, basically redoing the per well logic but packed in a way the flow model needs to learn from
        self.samples_per_well = None

    def _train_flow(self, potential: Potential, protocol: Protocol, loss: Loss) -> None:
        """Trains the normalizing flow on MCMC samples."""
        print(f"Generating training data for {self.flow_training_well_count} random wells using inherited NUTS...")

        # save the original number of samples and bounds
        original_samples_per_well = self.samples_per_well
        original_bounds_backup = self.starting_bounds.clone()
        original_num_samples = self.num_samples

        # Set parameters for training data generation
        # We use the user-specified flow_training_samples_per_well
        # Note: _run_multichain_mcmc uses mcmc_num_samples internally when samples_per_well is None
        # But here we are simulating "global" sampling for specific wells by setting bounds manually
        self.num_samples = self.flow_training_samples_per_well * self.chains_per_well

        try:
            all_samples = []
            all_contexts = []


            if not self.force_random:
                indices = torch.arange(self.flow_training_well_count, device=self.device)
                all_bitstrings = ((indices.unsqueeze(1) >> torch.arange(self.spatial_dimensions - 1, -1, -1, device=self.device)) & 1).float()

            # first we generate training for the flow model from MCMC NUTS
            for i in range(self.flow_training_well_count):
                if not self.force_random:
                    target_bits = all_bitstrings[i]
                else:
                    target_bits = torch.randint(0, 2, (self.spatial_dimensions,), device=self.device).float()

                # modify our underlying bounds to point to the new well
                self.set_bounds_from_bits(target_bits, loss)

                # call the parent method that computes the initial positions using mcmc nuts
                samples = self._run_multichain_mcmc(potential, protocol, loss)

                all_samples.append(samples)
                all_contexts.append(target_bits.unsqueeze(0).repeat(samples.shape[0], 1))

                if i % 10 == 0:
                    print(f"  - Sampled well {i+1}/{self.flow_training_well_count}")

            full_data = torch.cat(all_samples, dim=0)
            full_context = torch.cat(all_contexts, dim=0)

            # update our internal mean and std
            self.data_mean = full_data.mean(0)
            self.data_std = full_data.std(0) + 1e-6

            normalized_data = (full_data - self.data_mean) / self.data_std
            dataset = TensorDataset(normalized_data, full_context)
            loader = DataLoader(dataset, batch_size=self.flow_batch_size, shuffle=True)

            # train the flow model
            optimizer = torch.optim.Adam(self.flow_modules.parameters(), lr=1e-3)
            self.flow_modules.train()

            print("Training Conditional Flow...")
            best_loss = float('inf')
            patience = 20
            patience_counter = 0

            with tqdm(range(self.flow_epochs), desc="Flow Training") as pbar:
                for epoch in pbar:
                    epoch_loss = 0
                    for batch_x, batch_context in loader:
                        optimizer.zero_grad()
                        log_prob = self.flow_dist.condition(batch_context).log_prob(batch_x) #condition instructs it to use the bitstring to choose the underlying spline flow
                        loss_val = -log_prob.mean()
                        loss_val.backward()
                        optimizer.step()
                        epoch_loss += loss_val.item()

                    avg_epoch_loss = epoch_loss / len(loader)
                    pbar.set_postfix({'loss': f'{avg_epoch_loss:.4f}'})

                    # Early stopping check
                    if avg_epoch_loss < best_loss:
                        best_loss = avg_epoch_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            print(f"Early stopping at epoch {epoch} with loss {best_loss:.4f}")
                            break

            self.is_trained = True

        finally:
            # restore the original number of samples and bounds
            self.samples_per_well = original_samples_per_well
            self.starting_bounds = original_bounds_backup
            self.num_samples = original_num_samples

    def generate_initial_conditions(self, potential: Potential, protocol: Protocol, loss: Loss) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generates initial conditions using the trained flow model.

        Args:
            potential: The potential energy object.
            protocol: The protocol object.
            loss: The loss object.

        Returns:
            Tuple of (initial_pos, initial_vel, noise).
        """
        if not self.is_trained:
             self._train_flow(potential, protocol, loss)

        if self.samples_per_well is not None and not self.force_random: # per well mode, makes sure to sample from each well a known number of times

            total_samples_per_well = self.samples_per_well * self.chains_per_well

            num_wells = 2 ** self.spatial_dimensions
            indices = torch.arange(num_wells, device=self.device)
            all_bitstrings = ((indices.unsqueeze(1) >> torch.arange(self.spatial_dimensions - 1, -1, -1, device=self.device)) & 1).float()

            context = all_bitstrings.repeat_interleave(total_samples_per_well, dim=0)

            batch_size = context.shape[0]

        else:
            if self.samples_per_well is not None: #per random force
                batch_size = 16 * self.samples_per_well
            else:
                batch_size = self.num_samples
            context = torch.randint(0, 2, (batch_size, self.spatial_dimensions), device=self.device).float()


        # sample from the flow model
        with torch.no_grad():
            conditioned_flow = self.flow_dist.condition(context)
            x_standardized = conditioned_flow.sample(torch.Size([batch_size]))
            initial_pos = x_standardized * self.data_std + self.data_mean # unstandardize
            # log_p_target = self._log_prob(initial_pos.unsqueeze(-1), potential, protocol) # target log probability

            # log_jacobian = torch.log(self.data_std).sum()
            # log_q_flow = conditioned_flow.log_prob(x_standardized) - log_jacobian
            # log_weights = log_p_target - log_q_flow
            # weights = torch.exp(log_weights - log_weights.max())
            # weights = weights / weights.sum()

        #manipulate the parent class to make sure we get the right number of samples for velocity and noise
        original_num_samples = self.num_samples
        self.num_samples = batch_size

        initial_vel = self._get_initial_velocities()
        noise = self._get_noise()

        # restore the original number of samples for the parent class
        self.num_samples = original_num_samples


        return initial_pos, initial_vel, noise
