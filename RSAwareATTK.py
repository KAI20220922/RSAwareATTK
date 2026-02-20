from abc import ABCMeta, abstractmethod

import numpy as np
import torch
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Optional
from smoothing_core import Smooth


class Attacker(metaclass=ABCMeta):
    @abstractmethod
    def attack(self, inputs, targets):
        raise NotImplementedError


class RSAwareATTK(Attacker):
    """
    RSAwareATTK attack

    Parameters
    ----------
    steps : int
        Number of steps for the optimization.
    max_norm : float or None, optional
        If specified, the norms of the perturbations will not be greater than this value which might lower success rate.
    device : torch.device, optional
        Device on which to perform the attack.

    """

    def __init__(self,
                 steps: int,
                 random_start: bool = True,
                 temperature: float = 20,
                 max_norm: Optional[float] = None,
                 device: torch.device = torch.device('cpu')) -> None:
        super(RSAwareATTK, self).__init__()
        self.steps = steps
        self.random_start = random_start
        self.max_norm = max_norm
        self.device = device
        self.T = temperature
        self.iterations_at_end = -1


    def attack(self, model: nn.Module, inputs: torch.Tensor, labels: torch.Tensor,
               noise: torch.Tensor = None, num_noise_vectors=1, targeted: bool = False, no_grad=False, use_clip_range: bool = False) -> torch.Tensor:

        return self._attack_multinoise_RSAwareATTK(model, inputs, labels, noise, num_noise_vectors, targeted, use_clip_range)

    def attack_whitebox(self, smoothedmodel: Smooth,  basemodel: nn.Module, inputs: torch.Tensor, labels: torch.Tensor, N: int, alpha: float, batch: int,
               noise: torch.Tensor = None, num_noise_vectors=1, targeted: bool = False, no_grad=False,
                        use_clip_range: bool = False) -> torch.Tensor:

        return self._attack_multinoise_whitebox_RSAwareATTK(smoothedmodel, basemodel, inputs, labels, N, alpha, batch, noise, num_noise_vectors, targeted, use_clip_range)

    def _attack(self, model: nn.Module, inputs: torch.Tensor, labels: torch.Tensor,
                noise: torch.Tensor = None, targeted: bool = False) -> torch.Tensor:
        """
        Performs the attack of the model for the inputs and labels.

        Parameters
        ----------
        model : nn.Module
            Model to attack.
        inputs : torch.Tensor
            Batch of samples to attack. Values should be in the [0, 1] range.
        labels : torch.Tensor
            Labels of the samples to attack if untargeted, else labels of targets.
        targeted : bool, optional
            Whether to perform a targeted attack or not.

        Returns
        -------
        torch.Tensor
            Batch of samples modified to be adversarial to the model.

        """
        if inputs.min() < 0 or inputs.max() > 1: raise ValueError('Input values should be in the [0, 1] range.')

        batch_size = inputs.shape[0]
        multiplier = 1 if targeted else -1
        delta = torch.zeros_like(inputs, requires_grad=True)

        # Setup optimizers
        optimizer = optim.SGD([delta], lr=self.max_norm / self.steps * 2)

        for i in range(self.steps):

            adv = inputs + delta
            if noise is not None:
                adv = adv + noise
            logits = model(adv)
            pred_labels = logits.argmax(1)
            ce_loss = F.cross_entropy(logits, labels, reduction='sum')
            loss = multiplier * ce_loss

            optimizer.zero_grad()
            loss.backward()
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))

            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])

            optimizer.step()

            delta.data.add_(inputs)
            delta.data.clamp_(0, 1).sub_(inputs)

            delta.data.renorm_(p=2, dim=0, maxnorm=self.max_norm)

        return inputs + delta

    def _attack_multinoise_RSAwareATTK(self, model: nn.Module, inputs: torch.Tensor, labels: torch.Tensor,
                           noise: torch.Tensor = None, num_noise_vectors: int = 1,
                           targeted: bool = False, use_clip_range: bool = False) -> torch.Tensor:
        """
        Performs the attack of the model for the inputs and labels.

        Parameters
        ----------
        model : nn.Module
            Model to attack.
        inputs : torch.Tensor
            Batch of samples to attack. Values should be in the [0, 1] range.
        labels : torch.Tensor
            Labels of the samples to attack if untargeted, else labels of targets.
        targeted : bool, optional
            Whether to perform a targeted attack or not.

        Returns
        -------
        torch.Tensor
            Batch of samples modified to be adversarial to the model.

        """
        clip_min = torch.tensor([-2.1179, -2.0345, -1.8044], device=inputs.device).view(1, -1, 1, 1)
        clip_max = torch.tensor([ 2.2492,  2.4260,  2.6399], device=inputs.device).view(1, -1, 1, 1)
        batch_size = labels.shape[0]
        multiplier = 1 if targeted else -1
        delta = torch.zeros((len(labels), *inputs.shape[1:]), requires_grad=True, device=self.device)

        # Setup optimizers
        optimizer = optim.SGD([delta], lr=self.max_norm / self.steps * 2)

        def logistic_fun(a):
            return 1 / (1 + torch.exp(-a))

        for i in range(self.steps):

            if i > 99:
                abc = 0

            adv = inputs + delta.repeat(1, num_noise_vectors, 1, 1).view_as(inputs)
            if noise is not None:
                adv = adv + noise

            # Logits and their softmax values
            logits = model(adv)
            softmax = F.softmax(logits, dim=1)

            # SIGMOID( (Softmax of the true class - 0.5 ) x temperature )
            x = logistic_fun(self.T * (softmax - 0.5))

            # Resume the processing of the other attacks: average, log, loss etc.
            average_x = x.reshape(-1, num_noise_vectors, logits.shape[-1]).mean(1, keepdim=True).squeeze(1)
            logsofx = torch.log(average_x.clamp(min=1e-20))
            ce_loss = F.nll_loss(logsofx, labels)

            loss = multiplier * ce_loss

            optimizer.zero_grad()
            loss.backward()
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))

            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])

            optimizer.step()

            delta.data.add_(inputs[::num_noise_vectors])
            if use_clip_range:
                delta.data.clamp_(clip_min, clip_max).sub_(inputs[::num_noise_vectors])
            else:
                delta.data.clamp_(0, 1).sub_(inputs[::num_noise_vectors])
            

            delta.data.renorm_(p=2, dim=0, maxnorm=self.max_norm)

        return inputs + delta.repeat(1, num_noise_vectors, 1, 1).view_as(inputs)

    def _attack_multinoise_whitebox_RSAwareATTK(self, smoothedmodel: Smooth, basemodel: nn.Module, inputs: torch.Tensor, labels: torch.Tensor,
                                          N: int, alpha: float, batch: int,
                                          noise: torch.Tensor = None, num_noise_vectors: int = 1,
                                          targeted: bool = False, use_clip_range: bool = False) -> torch.Tensor:
        """
        Performs the attack of the model for the inputs and labels.

        Parameters
        ----------
        model : nn.Module
            Model to attack.
        inputs : torch.Tensor
            Batch of samples to attack. Values should be in the [0, 1] range.
        labels : torch.Tensor
            Labels of the samples to attack if untargeted, else labels of targets.
        targeted : bool, optional
            Whether to perform a targeted attack or not.

        Returns
        -------
        torch.Tensor
            Batch of samples modified to be adversarial to the model.

        """
        clip_min = torch.tensor([-2.1179, -2.0345, -1.8044], device=inputs.device).view(1, -1, 1, 1)
        clip_max = torch.tensor([ 2.2492,  2.4260,  2.6399], device=inputs.device).view(1, -1, 1, 1)
        batch_size = labels.shape[0]
        multiplier = 1 if targeted else -1
        delta = torch.zeros((len(labels), *inputs.shape[1:]), requires_grad=True, device=self.device)

        

        # Setup optimizers
        # optimizer = optim.SGD([delta], lr=self.max_norm / self.steps * 2)
        # MODIFICA PER FAR SI CHE LA STRENGHT CON CUI SI APPLICA L'ATTACCO NON RIMPICCIOLISCA ALL'AUMENTARE DEL NUMERO DI STEP
        # PRESO CASO CON STEP = 400 COME STRENGHT DI RIFERIMENTO
        optimizer = optim.SGD([delta], lr=self.max_norm / 400 * 2)

        def logistic_fun(a):
            return 1 / (1 + torch.exp(-a))

        n_adversarial = 0
        for i in range(self.steps):

            # updating the attack iteration variable so that it can be read at the end of the attack
            self.iterations_at_end = i

            adv = inputs + delta.repeat(1, num_noise_vectors, 1, 1).view_as(inputs)

            # IF ATTACK IS SUCCESSFUL IN CHANGING THE INPUT IMAGE CLASS, BREAK THE LOOP AND RETURN THE ADVERSARIAL IMAGE
            prediction = smoothedmodel.predict(adv[:1], N, alpha, batch)
            if labels.item() != prediction and prediction != -1:
                n_adversarial += 1

                if n_adversarial > 5:
                    print(f'    - Attack successful @step {i}')
                    return inputs + delta.repeat(1, num_noise_vectors, 1, 1).view_as(inputs)

            if noise is not None:
                adv = adv + noise

            # if use_clip_range:
            #     adv = torch.max(torch.min(adv, clip_max), clip_min)

            # Logits and their softmax values (base model)
            logits = basemodel(adv)
            softmax = F.softmax(logits, dim=1)

            # SIGMOID( (Softmax of the true class - 0.5 ) x temperature )
            x = logistic_fun(self.T * (softmax - 0.5))

            # Resume the processing of the other attacks: average, log, loss etc.
            average_x = x.reshape(-1, num_noise_vectors, logits.shape[-1]).mean(1, keepdim=True).squeeze(1)
            logsofx = torch.log(average_x.clamp(min=1e-20))
            ce_loss = F.nll_loss(logsofx, labels)

            loss = multiplier * ce_loss

            optimizer.zero_grad()
            loss.backward()

            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))

            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])

            optimizer.step()

            # INTERRUPT IF THE L2 NORM OF DELTA IS TOO LARGE (MSE>100). L2 is averaged over the three channels
            max_distortion = 1000 * np.sqrt(inputs.shape[-1] * inputs.shape[-2]) / 255

            # L2 norm of delta averaged over the three channels
            l2norm_delta = 0
            for c in range(delta.shape[1]):
                l2norm_delta += (delta[0, c, :, :].pow(2)).sum().sqrt()
            l2norm_delta /= delta.shape[1]

            if l2norm_delta > max_distortion:
                print(f'    - Attack aborted because distortion is too large @step {i}')
                return inputs + delta.repeat(1, num_noise_vectors, 1, 1).view_as(inputs)

            delta.data.add_(inputs[::num_noise_vectors])
            if use_clip_range:
                delta.data.clamp_(clip_min, clip_max).sub_(inputs[::num_noise_vectors])
            else:
                delta.data.clamp_(0, 1).sub_(inputs[::num_noise_vectors])

            # DISABLE RENORM
            #delta.data.renorm_(p=2, dim=0, maxnorm=self.max_norm)

        print(f'    - Attack reached last step {self.steps}')
        return inputs + delta.repeat(1, num_noise_vectors, 1, 1).view_as(inputs)

