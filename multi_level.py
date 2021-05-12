from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import math
import os
import torch
import torch.distributions as dist
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import utils

plt.style.use(['seaborn-white', 'seaborn-paper', 'seaborn-ticks'])
matplotlib.rc('font', family='Latin Modern Roman')

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# # CIFAR10
# x_min = torch.tensor([0, 0, 0]).cuda()
# x_max = torch.tensor([1, 1, 1]).cuda()

# MNIST
x_min = 0
x_max = 1


def greyscale_multilevel_uniform(
        prop,
        x_sample,
        sigma=1.0,
        CUDA=False,
        rho=0.1,
        count_particles=1000,
        count_mh_steps=100,
        debug=True, stats=False):
    # Calculate the mean of the normal distribution in logit space
    # We transform the input from [x_min, x_max] to [epsilon, 1 - epsilon], then to [logit(epsilon), logit(1 - epsilon)]
    # Then we can do the sampling on (-inf, inf)
    if CUDA:
        prior = dist.Uniform(low=torch.max(x_sample-sigma, torch.tensor([x_min]).cuda()), high=torch.min(x_sample+sigma, torch.tensor([x_max]).cuda()))
    else:
        prior = dist.Uniform(low=torch.max(x_sample-sigma, torch.tensor([x_min])), high=torch.min(x_sample+sigma, torch.tensor([x_max])))

    # Parameters
    if CUDA:
        width_proposal = sigma*torch.ones(count_particles).cuda()/30
    else:
        width_proposal = sigma*torch.ones(count_particles)/30

    count_max_levels = 500
    target_acc_ratio = 0.9
    max_width_proposal = 0.1
    min_width_proposal = 1e-8
    width_inc = 1.02
    width_dec = 0.5

    # Sample the initial particles
    # Implements parallel batched accept-reject sampling.
    x = prior.sample(torch.Size([count_particles]))
    L_prev = -math.inf
    L = -math.inf
    l_inf_min = math.inf
    lg_p = 0
    max_val = -math.inf
    levels = []

    # print('Inside valid bounds', x_min, x_max)
    # utils.stats(x[0])
    # print((x >= x_min).all(dim=1) & (x <= x_max).all(dim=1))
    # raise Exception()

    # Loop over levels
    for level_idx in range(count_max_levels):
        if CUDA:
            acc_ratio = torch.zeros(count_particles).cuda()
        else:
            acc_ratio = torch.zeros(count_particles)

        if L >= 0:
            break

        # Calculate current level
        s_x = prop(x).squeeze(-1)
        max_val = max(max_val, s_x.max().item())
        s_sorted, s_idx = torch.sort(s_x)
        L = min(s_sorted[math.floor((1 - rho) * count_particles)].item(), 0)
        if L == L_prev:
            L = 0
        levels.append(L)
        where_keep = s_x >= L
        where_kill = s_x < L
        count_kill = (where_kill).sum()
        count_keep = count_particles - count_kill

        # Print level
        if debug:
            print(f'Level {level_idx + 1} = {L}')

        # Terminate if change in level is below some threshold
        if count_keep == 0:
            return -math.inf, max_val, x, levels

        lg_p += torch.log(count_keep.float()).item() - math.log(count_particles)
        # print('term', torch.log(count_keep.float()).item() - math.log(count_particles))

        # Early termination
        if lg_p < -20:
            return -20., None, x, levels

        # Uniformly resample killed particles below the level
        new_idx = torch.randint(low=0, high=count_keep, size=(count_kill,), dtype=torch.long)
        x = x[where_keep]
        x = torch.cat((x, x[new_idx]), dim=0)
        width_proposal = width_proposal[where_keep]
        width_proposal = torch.cat((width_proposal, width_proposal[new_idx]), dim=0)

        # acc_ratio = torch.zeros(count_kill).cuda()
        # x_temp = x
        # while acc_ratio.mean() < 0.2:
        #  x = x_temp
        if CUDA:
            acc_ratio = torch.zeros(count_particles).cuda()
        else:
            acc_ratio = torch.zeros(count_particles)

        for mh_idx in range(count_mh_steps):
            # Propose new sample
            g_bottom = dist.Uniform(low=torch.max(x - width_proposal.unsqueeze(-1), prior.low),
                                    high=torch.min(x + width_proposal.unsqueeze(-1), prior.high))
            # g_bottom = dist.Normal(x, width_proposal.unsqueeze(-1))

            x_maybe = g_bottom.sample()
            s_x = prop(x_maybe).squeeze(-1)

            # Calculate log-acceptance ratio
            g_top = dist.Uniform(low=torch.max(x_maybe - width_proposal.unsqueeze(-1), prior.low),
                                 high=torch.min(x_maybe + width_proposal.unsqueeze(-1), prior.high))
            # g_top = dist.Normal(x_maybe, width_proposal.unsqueeze(-1))
            lg_alpha = (prior.log_prob(x_maybe) + g_top.log_prob(x) - prior.log_prob(x) - g_bottom.log_prob(x_maybe)).sum(dim=1)
            acceptance = torch.min(lg_alpha, torch.zeros_like(lg_alpha))

            # Work out which ones to accept
            log_u = torch.log(torch.rand_like(acceptance))
            acc_idx = (log_u <= acceptance) & (
                        s_x >= L)  # & (x_maybe >= x_min).all(dim=1) & (x_maybe <= x_max).all(dim=1)
            acc_ratio += acc_idx.float()
            x = torch.where(acc_idx.unsqueeze(-1), x_maybe, x)

        # Adapt the width proposal *for each chain individually*
        acc_ratio /= count_mh_steps

        # DEBUG: See what acceptance ratios are doing
        if stats:
            utils.stats(acc_ratio)
        # input()

        # print(acc_ratio.size())
        width_proposal = torch.where(acc_ratio > 0.124, width_proposal * width_inc, width_proposal)
        width_proposal = torch.where(acc_ratio < 0.124, width_proposal * width_dec, width_proposal)

        L_prev = L
        # input()

    # We return both the estimate of the log-probability of the integral and the set of adversarial examples
    s_x = prop(x).squeeze(-1)
    # max_val = max(max_val, x.max().item())
    max_val = s_x.max().item()
    return lg_p, max_val, x, levels  # , l_inf_min


def multilevel_uniform(
        prop,
        x_sample,
        sigma=1.,
        CUDA=False,
        rho=0.1,
        count_particles=1000,
        count_mh_steps=100,
        debug=True, stats=False):
    # Calculate the mean of the normal distribution in logit space
    # We transform the input from [x_min, x_max] to [epsilon, 1 - epsilon], then to [logit(epsilon), logit(1 - epsilon)]
    # Then we can do the sampling on (-inf, inf)
    prior = dist.Uniform(low=torch.max(x_sample-sigma*(x_max-x_min).view(3,1,1), x_min.view(3,1,1)), high=torch.min(x_sample+sigma*(x_max-x_min).view(3,1,1), x_max.view(3,1,1)))

    # Parameters
    if CUDA:
        width_proposal = sigma*torch.ones(count_particles).cuda()/30
    else:
        width_proposal = sigma*torch.ones(count_particles)/30

    count_max_levels = 500
    target_acc_ratio = 0.9
    max_width_proposal = 0.1
    min_width_proposal = 1e-8
    width_inc = 1.02
    width_dec = 0.5

    # Sample the initial particles
    # Implements parallel batched accept-reject sampling.
    x = prior.sample(torch.Size([count_particles]))
    L_prev = -math.inf
    L = -math.inf
    l_inf_min = math.inf
    lg_p = 0
    max_val = -math.inf
    levels = []

    # print('Inside valid bounds', x_min, x_max)
    # utils.stats(x[0])
    # print((x >= x_min).all(dim=1) & (x <= x_max).all(dim=1))
    # raise Exception()

    # Loop over levels
    for level_idx in range(count_max_levels):
        if CUDA:
            acc_ratio = torch.zeros(count_particles).cuda()
        else:
            acc_ratio = torch.zeros(count_particles)

        if L >= 0:
            break

        # Calculate current level
        s_x = prop(x).squeeze(-1)
        max_val = max(max_val, s_x.max().item())
        s_sorted, s_idx = torch.sort(s_x)
        L = min(s_sorted[math.floor((1 - rho) * count_particles)].item(), 0)
        if L == L_prev:
            L = 0
        levels.append(L)
        where_keep = s_x >= L
        where_kill = s_x < L
        count_kill = (where_kill).sum()
        count_keep = count_particles - count_kill

        # Print level
        if debug:
            print(f'Level {level_idx + 1} = {L}')

        # Terminate if change in level is below some threshold
        if count_keep == 0:
            return -math.inf, max_val, x, levels

        lg_p += torch.log(count_keep.float()).item() - math.log(count_particles)
        # print('term', torch.log(count_keep.float()).item() - math.log(count_particles))

        # Early termination
        if lg_p < -20:
            return -20., None, x, levels

        # Uniformly resample killed particles below the level
        new_idx = torch.randint(low=0, high=count_keep, size=(count_kill,), dtype=torch.long)
        x = x[where_keep]
        x = torch.cat((x, x[new_idx]), dim=0)
        width_proposal = width_proposal[where_keep]
        width_proposal = torch.cat((width_proposal, width_proposal[new_idx]), dim=0)

        # acc_ratio = torch.zeros(count_kill).cuda()
        # x_temp = x
        # while acc_ratio.mean() < 0.2:
        #  x = x_temp
        if CUDA:
            acc_ratio = torch.zeros(count_particles).cuda()
        else:
            acc_ratio = torch.zeros(count_particles)

        for mh_idx in range(count_mh_steps):
            # Propose new sample
            g_bottom = dist.Uniform(low=torch.max(x - width_proposal.view(-1,1,1,1), prior.low),
                                    high=torch.min(x + width_proposal.view(-1,1,1,1), prior.high))
            # g_bottom = dist.Normal(x, width_proposal.unsqueeze(-1))

            x_maybe = g_bottom.sample()
            s_x = prop(x_maybe).squeeze(-1)

            # Calculate log-acceptance ratio
            g_top = dist.Uniform(low=torch.max(x_maybe - width_proposal.view(-1,1,1,1), prior.low),
                                 high=torch.min(x_maybe + width_proposal.view(-1,1,1,1), prior.high))
            # g_top = dist.Normal(x_maybe, width_proposal.unsqueeze(-1))
            lg_alpha = (prior.log_prob(x_maybe) + g_top.log_prob(x) - prior.log_prob(x) - g_bottom.log_prob(x_maybe)).view(count_particles,-1).sum(dim=1)
            acceptance = torch.min(lg_alpha, torch.zeros_like(lg_alpha))

            # Work out which ones to accept
            log_u = torch.log(torch.rand_like(acceptance))
            acc_idx = (log_u <= acceptance) & (s_x >= L)  # & (x_maybe >= x_min).all(dim=1) & (x_maybe <= x_max).all(dim=1)
            acc_ratio += acc_idx.float()
            x = torch.where(acc_idx.view(-1,1,1,1), x_maybe, x)

        # Adapt the width proposal *for each chain individually*
        acc_ratio /= count_mh_steps

        # DEBUG: See what acceptance ratios are doing
        if stats:
            utils.stats(acc_ratio)
        # input()

        # print(acc_ratio.size())
        width_proposal = torch.where(acc_ratio > 0.124, width_proposal * width_inc, width_proposal)
        width_proposal = torch.where(acc_ratio < 0.124, width_proposal * width_dec, width_proposal)

        L_prev = L
        # input()

    # We return both the estimate of the log-probability of the integral and the set of adversarial examples
    s_x = prop(x).squeeze(-1)
    # max_val = max(max_val, x.max().item())
    max_val = s_x.max().item()
    return lg_p, max_val, x, levels  # , l_inf_min
