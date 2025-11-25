from abc import ABC, abstractmethod
from sqlalchemy import false
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from torch.func import vmap, hessian
import math
import itertools
import pyro
from pyro.infer.mcmc import MCMC, NUTS
import pyro.distributions as dist
import pyro.distributions.transforms as T
from tqdm import tqdm
import os


class InitialConditionGenerator(ABC):
    @abstractmethod
    def generate_initial_conditions(self, potential, potential_model, loss):
        #you should return initial positions, initial velocities, and noise
        pass


class McmcNuts(InitialConditionGenerator):
    def __init__(self, params, device):
        self.device = device
        self.spatial_dimensions = params.get('spatial_dimensions', 1)
        self.time_steps = params.get('time_steps', 1000)
        self.mcmc_warmup_ratio = params.get('mcmc_warmup_ratio', 0.1)
        self.mcmc_starting_spatial_bounds = params.get('mcmc_starting_spatial_bounds', torch.tensor([[-5, 5]] * self.spatial_dimensions, device=self.device))
        self.mcmc_chains_per_well = params.get('mcmc_chains_per_well', 1)
        self.min_neff = params.get('min_neff', None)
        self.samples_per_well = params.get('samples_per_well', None)
        if self.samples_per_well is None:
            self.mcmc_num_samples = params.get('mcmc_num_samples', 5000)
        else:
            self.mcmc_num_samples = self.samples_per_well * self.mcmc_chains_per_well * 2**self.spatial_dimensions
        self.beta = params.get('beta', 1.0)
        self.gamma = params['gamma']
        self.dt = params['dt']
        self.noise_sigma = torch.sqrt(torch.tensor(2 * self.gamma / self.beta))
        self.mass = params['mass']
        self.starting_pos = None
        if params.get('run_every_epoch', False):
            print("Warning: Running MCMC every epoch will be very slow and is not recommended for training. Please use the ConditionalFlowBoltzmannGenerator instead if your potential doesn't change at t0.")

    def _get_initial_velocities(self):
        # P(v) = exp(-beta * E_k)
        # E_k = 1/2 * m * v^2
        # P(v) = exp(-beta * 1/2 * m * v^2)
        # generally P(v) = exp(-v^2 * 1/2 / sigma^2)
        # v^2 * 1/2 / sigma^2 = beta * 1/2 * m * v^2
        # sigma^2 = 1 / (beta * m)
        # sigma = sqrt(1 / (beta * m))

        # we draw from randn which produces N(0, 1) and scale by sigma to get N(0, sigma)
        var = 1 / (self.beta * self.mass)
        samples = torch.randn(self.mcmc_num_samples, self.spatial_dimensions, device=self.device) * math.sqrt(var)
        return samples

    def _get_noise(self):
        samples = torch.randn(self.mcmc_num_samples, self.spatial_dimensions, self.time_steps, device=self.device) * self.noise_sigma * math.sqrt(self.dt)
        return samples

    def _run_multichain_mcmc(self, potential, potential_model, loss):
        """Run MCMC sampling, either per-well or global depending on parameters"""
        
        max_attempts = 5
        
        if self.samples_per_well is not None:
            # Per-well sampling mode
            well_bounds = self._partition_bounds_by_midpoints(loss)
            all_samples = []
            
            for well_idx, bounds in enumerate(well_bounds):
                
                samples_per_chain = self.samples_per_well // self.mcmc_chains_per_well
                
                for attempt in range(max_attempts):
                    well_samples = []
                    
                    # Run chains
                    for chain_id in range(self.mcmc_chains_per_well):
                        sampler = MCMC(
                            NUTS(lambda: self._posterior_for_mcmc(bounds[:, 0], bounds[:, 1], potential, potential_model)),
                            num_samples=samples_per_chain,
                            warmup_steps=int(self.mcmc_warmup_ratio * samples_per_chain)
                        )
                        sampler.run()
                        well_samples.append(sampler.get_samples()['x'])
                    
                    # Check Neff if required
                    current_samples = torch.cat(well_samples, dim=0)
                    
                    if self.min_neff is None:
                        break
                        
                    # we stack the chains together to vectorize the Neff check
                    chain_stack = torch.stack(well_samples, dim=0)
                    neff_per_dim = pyro.ops.stats.effective_sample_size(chain_stack, chain_dim=0, sample_dim=1)
                    min_observed_neff = neff_per_dim.min().item()
                    
                    print(f"Well {well_idx} Attempt {attempt+1}: Neff = {min_observed_neff:.2f} (Target: {self.min_neff})")
                    
                    if min_observed_neff >= self.min_neff:
                        break
                    
                    # Need more samples
                    if attempt < max_attempts - 1:
                        ratio = self.min_neff / max(min_observed_neff, 1e-6)
                        # scale factor, bounded to avoid explosion but ensure progress
                        scale_factor = max(1.1, min(ratio, 5.0)) 
                        new_samples_per_chain = int(samples_per_chain * scale_factor)
                        print(f"  - Neff insufficient. Increasing samples per chain from {samples_per_chain} to {new_samples_per_chain}")
                        samples_per_chain = new_samples_per_chain
                    else:
                        print(f"  - WARNING: Max attempts reached for Well {well_idx}. Neff {min_observed_neff:.2f} < {self.min_neff}")
                
                # Print Neff statistics for the final samples of this well
                # Re-calculate since we might have broken out of loop
                chain_stack = torch.stack(well_samples, dim=0)
                neff_per_dim = pyro.ops.stats.effective_sample_size(chain_stack, chain_dim=0, sample_dim=1)
                min_neff = neff_per_dim.min().item()
                mean_neff = neff_per_dim.mean().item()
                print(f"  Well {well_idx} Stats: Min Neff: {min_neff:.2f}, Mean Neff: {mean_neff:.2f}")

                all_samples.append(current_samples)
            
            self.starting_pos = torch.cat(all_samples, dim=0)
        else:
            # Legacy mode: global sampling
            all_samples = []
            num_chains = self.mcmc_chains_per_well
            samples_per_chain = self.mcmc_num_samples // num_chains
            
            bounds_low = self.mcmc_starting_spatial_bounds[:, 0]
            bounds_high = self.mcmc_starting_spatial_bounds[:, 1]
            
            for attempt in range(max_attempts):
                chain_samples_list = []
                
                for chain_id in range(num_chains):
                    sampler = MCMC(
                        NUTS(lambda: self._posterior_for_mcmc(bounds_low, bounds_high, potential, potential_model)),
                        num_samples=samples_per_chain,
                        warmup_steps=int(self.mcmc_warmup_ratio * samples_per_chain)
                    )
                    sampler.run()
                    chain_samples_list.append(sampler.get_samples()['x'])
                
                # Check Neff
                if self.min_neff is None:
                    all_samples = chain_samples_list
                    break
                    
                chain_stack = torch.stack(chain_samples_list, dim=0)
                neff_per_dim = pyro.ops.stats.effective_sample_size(chain_stack, chain_dim=0, sample_dim=1)
                min_observed_neff = neff_per_dim.min().item()
                
                print(f"Global Sampling Attempt {attempt+1}: Neff = {min_observed_neff:.2f} (Target: {self.min_neff})")
                
                if min_observed_neff >= self.min_neff:
                    all_samples = chain_samples_list
                    break
                    
                if attempt < max_attempts - 1:
                    ratio = self.min_neff / max(min_observed_neff, 1e-6)
                    scale_factor = max(1.1, min(ratio, 5.0))
                    new_samples_per_chain = int(samples_per_chain * scale_factor)
                    print(f"  - Neff insufficient. Increasing samples per chain from {samples_per_chain} to {new_samples_per_chain}")
                    samples_per_chain = new_samples_per_chain
                else:
                    print(f"  - WARNING: Max attempts reached. Neff {min_observed_neff:.2f} < {self.min_neff}")
                    all_samples = chain_samples_list

            # Print final stats for global sampling
            if len(all_samples) > 0:
                 chain_stack = torch.stack(all_samples, dim=0)
                 neff_per_dim = pyro.ops.stats.effective_sample_size(chain_stack, chain_dim=0, sample_dim=1)
                 min_neff = neff_per_dim.min().item()
                 mean_neff = neff_per_dim.mean().item()
                 print(f"  MCMC Batch Stats: Min Neff: {min_neff:.2f}, Mean Neff: {mean_neff:.2f}")
            
            self.starting_pos = torch.cat(all_samples, dim=0)
        return self.starting_pos
    
    def _log_prob(self, state_vectors, potential, potential_model):
        #exp(-beta * U), boltzman distribution assumed for posterior
        coeff_at_t0 = potential_model.get_coeff_grid()[:, 0]
        return -self.beta * potential.potential_value(state_vectors, coeff_at_t0)

    def _posterior_for_mcmc(self, bounds_low, bounds_high, potential, potential_model):
        #makes it look like pyro need it to look
        x = pyro.sample("x", pyro.distributions.Uniform(bounds_low, bounds_high).to_event(1))
        pyro.factor("logp", self._log_prob(x.unsqueeze(-1), potential, potential_model))

    def _partition_bounds_by_midpoints(self, loss):
        """Partition the sampling bounds using midpoints to create per-well regions"""

        if not hasattr(loss, 'midpoints'):
            return [self.mcmc_starting_spatial_bounds]
        
        midpoints = loss.midpoints
        bounds = self.mcmc_starting_spatial_bounds
        
        dim_segments = []
        for dim_idx in range(self.spatial_dimensions):
            low = bounds[dim_idx, 0]
            high = bounds[dim_idx, 1]
            mid = midpoints[dim_idx]
            dim_segments.append([[low, mid], [mid, high]])
        well_bounds = []
        for combination in itertools.product(*dim_segments):
            well_bound = torch.stack([torch.tensor(seg, device=self.device) for seg in combination])
            well_bounds.append(well_bound)
        return well_bounds

    def generate_initial_conditions(self, potential, potential_model, loss):
        if self.run_every_epoch:
            initial_pos = self.starting_pos
        else:
            initial_pos = self._run_multichain_mcmc(potential, potential_model, loss)
        initial_vel = self._get_initial_velocities()
        noise = self._get_noise()
        return initial_pos, initial_vel, noise


class ConditionalFlow(McmcNuts, nn.Module):
    # An extension of the MCMC NUTS generator that trains a conditional flow model
    # to efficiently sample from an approximation of the posterior distribution every epoch 
    # far quicker than running MCMC every epoch
    def __init__(self, params, device):
        nn.Module.__init__(self)
        super().__init__(params, device)
        if self.spatial_dimensions > 8:
            print('WARNING: when spatial dimensions are greater than 8 the number of samples is forced to 8 * samples_per_well and is taken randomly from the global bounds. You are NOT guaranteed to have samples from each well each epoch nor the normalizing flow to properly learn the entire bitstring space.')
            self.force_random = True
        else:
            self.force_random = False

        self.context_dim = self.spatial_dimensions 
        
        self.original_bounds = self.mcmc_starting_spatial_bounds.clone()
        
        transforms = []
        flow_layers = params.get('flow_layers', 4)
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

        self.flow_epochs = params.get('flow_epochs', 300)
        self.flow_batch_size = params.get('flow_batch_size', 256)
        self.flow_training_well_count = 2**self.spatial_dimensions
        self.flow_training_samples_per_well = params.get('flow_training_samples_per_well', 500)

    def set_bounds_from_bits(self, target_bitstring, loss):
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
        self.mcmc_starting_spatial_bounds = torch.tensor(new_bounds, device=self.device)
        
        # runs the parent class in global mode on this subset, basically redoing the per well logic but packed in a way the flow model needs to learn from
        self.samples_per_well = None

    def _train_flow(self, potential, potential_model, loss):
        print(f"Generating training data for {self.flow_training_well_count} random wells using inherited NUTS...")
        
        # save the original number of samples and bounds
        original_samples_per_well = self.samples_per_well
        original_bounds_backup = self.mcmc_starting_spatial_bounds.clone()
        original_mcmc_num_samples = self.mcmc_num_samples
        
        # Set parameters for training data generation
        # We use the user-specified flow_training_samples_per_well
        # Note: _run_multichain_mcmc uses mcmc_num_samples internally when samples_per_well is None
        # But here we are simulating "global" sampling for specific wells by setting bounds manually
        self.mcmc_num_samples = self.flow_training_samples_per_well * self.mcmc_chains_per_well

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
                samples = self._run_multichain_mcmc(potential, potential_model, loss)
                
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
            self.mcmc_starting_spatial_bounds = original_bounds_backup
            self.mcmc_num_samples = original_mcmc_num_samples

    def generate_initial_conditions(self, potential, potential_model, loss):
        if not self.is_trained:
             self._train_flow(potential, potential_model, loss)
            
        if self.samples_per_well is not None and not self.force_random: # per well mode, makes sure to sample from each well a known number of times

            total_samples_per_well = self.samples_per_well * self.mcmc_chains_per_well
            
            num_wells = 2 ** self.spatial_dimensions
            indices = torch.arange(num_wells, device=self.device)
            all_bitstrings = ((indices.unsqueeze(1) >> torch.arange(self.spatial_dimensions - 1, -1, -1, device=self.device)) & 1).float()

            context = all_bitstrings.repeat_interleave(total_samples_per_well, dim=0)
            
            batch_size = context.shape[0]
            
        else:
            if self.samples_per_well is not None: #per random force
                batch_size = 16 * self.samples_per_well
            else:
                batch_size = self.mcmc_num_samples
            context = torch.randint(0, 2, (batch_size, self.spatial_dimensions), device=self.device).float()

        
        # sample from the flow model
        with torch.no_grad():
            conditioned_flow = self.flow_dist.condition(context)
            x_standardized = conditioned_flow.sample(torch.Size([batch_size])) 
            initial_pos = x_standardized * self.data_std + self.data_mean # unstandardize
            # log_p_target = self._log_prob(initial_pos.unsqueeze(-1), potential, potential_model) # target log probability
            
            # log_jacobian = torch.log(self.data_std).sum()
            # log_q_flow = conditioned_flow.log_prob(x_standardized) - log_jacobian
            # log_weights = log_p_target - log_q_flow
            # weights = torch.exp(log_weights - log_weights.max())
            # weights = weights / weights.sum()
        
        #manipulate the parent class to make sure we get the right number of samples for velocity and noise
        original_num_samples = self.mcmc_num_samples
        self.mcmc_num_samples = batch_size
        
        initial_vel = self._get_initial_velocities()
        noise = self._get_noise()
        
        # restore the original number of samples for the parent class
        self.mcmc_num_samples = original_num_samples
        

        return initial_pos, initial_vel, noise

class LaplaceApproximation(InitialConditionGenerator):
    def __init__(self, params, centers, device):
        self.centers = centers
        self.device = device
        self.dt = params['dt']
        self.gamma = params['gamma']
        self.mass = params['mass']
        self.beta = params['beta']
        self.spatial_dimensions = params.get('spatial_dimensions', 1)
        self.time_steps = params.get('time_steps', 1000)
        self.mcmc_starting_spatial_bounds = params.get('mcmc_starting_spatial_bounds', None)
        
        if 'num_samples' in params:
            self.num_samples = params['num_samples']
        elif 'mcmc_num_samples' in params:
            self.num_samples = params['mcmc_num_samples']
        elif 'samples_per_well' in params:
             self.num_samples = params['samples_per_well'] * (2**self.spatial_dimensions)
        else:
             self.num_samples = 1000
             
        self.noise_sigma = torch.sqrt(torch.tensor(2 * self.gamma / self.beta))

    def _solve_landscape(self, potential, potential_model):
        coeff_at_t0 = potential_model.get_coeff_grid()[:, 0]
        def potential_kernel(x):
            return potential.potential_value(x, coeff_at_t0)
        
        batched_hessian_func = vmap(hessian(potential_kernel), in_dims = 0)

        hessian_at_centers = batched_hessian_func(self.centers) # Nwells x spatial_dimensions x spatial_dimensions
        sign, logabsdet = torch.linalg.slogdet(hessian_at_centers)
        if not (sign == 1).all():
            raise ValueError(f"Some hessians ({len(hessian_at_centers[sign == -1])}/{len(hessian_at_centers):.2f}%) are not positive definite, some of your bit centers are unstable")
        if not (logabsdet > math.log(1e-5)).all():
            raise ValueError(f"Some hessians ({len(logabsdet[logabsdet <= 0])}/{len(logabsdet):.2f}%) have too near zero determinant (1e-5 cutoff), some of your bit centers are unsuitable for this method. Request better handling of this if needed.")
        # Log P = -E - 0.5 * log|H|
        self.log_weights = -potential_kernel(self.centers) - 0.5 * logabsdet
        self.hessian_at_centers = hessian_at_centers

    def _get_initial_velocities(self):
        # P(v) = exp(-beta * E_k)
        # E_k = 1/2 * m * v^2
        # P(v) = exp(-beta * 1/2 * m * v^2)
        # generally P(v) = exp(-v^2 * 1/2 / sigma^2)
        # v^2 * 1/2 / sigma^2 = beta * 1/2 * m * v^2
        # sigma^2 = 1 / (beta * m)
        # sigma = sqrt(1 / (beta * m))

        # we draw from randn which produces N(0, 1) and scale by sigma to get N(0, sigma)
        var = 1 / (self.beta * self.mass)
        samples = torch.randn(self.num_samples, self.spatial_dimensions, device=self.device) * math.sqrt(var)
        return samples

    def _get_noise(self):
        samples = torch.randn(self.num_samples, self.spatial_dimensions, self.time_steps, device=self.device) * self.noise_sigma * math.sqrt(self.dt)
        return samples

    def _get_initial_positions(self):
        with torch.no_grad():
            chosen_centers = dist.Categorical(self.log_weights).sample((self.num_samples,))
            center_locs = self.centers[chosen_centers]
            chosen_H = self.hessian_at_centers[chosen_centers]

            # Laplaces approximation, x ~ N(x_0, H^-1)
            # precision matrix will invert the hessian internally
            samples = dist.MultivariateNormal(center_locs, precision_matrix = chosen_H).sample()
            return samples

    def generate_initial_conditions(self, potential, potential_model, loss):
        if not hasattr(self, 'log_weights'):
            print("Solving landscape for Laplace approximation...")
            self._solve_landscape(potential, potential_model)
            print("Landscape solved")
        
        initial_pos = self._get_initial_positions()
        initial_vel = self._get_initial_velocities()
        noise = self._get_noise()
        return initial_pos, initial_vel, noise

