import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from globals import FGSM, PGD, ALPHA, EPSILON, NUM_ITER

def denormalize(batch, mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261]):
    """
    Convert a batch of tensors to their original scale.

    Args:
        batch (torch.Tensor): Batch of normalized tensors.
        mean (torch.Tensor or list): Mean used for normalization.
        std (torch.Tensor or list): Standard deviation used for normalization.

    Returns:
        torch.Tensor: batch of tensors without normalization applied to them.
    """
    device = batch.device
    if isinstance(batch, np.ndarray):
        batch = torch.tensor(batch).to(device)
    if isinstance(mean, list):
        mean = torch.tensor(mean).to(device)
    if isinstance(std, list):
        std = torch.tensor(std).to(device)
    return batch * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)


def fgsm_attack(image, data_grad, epsilon = 0.25):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image


    
def fgsm_loss(model, criterion, inputs, labels, defense_args, return_preds = True):
    alpha = defense_args[ALPHA]
    epsilon = defense_args[EPSILON]
    inputs.requires_grad = True
    # Implement the FGSM attack

    original_outputs = model(inputs)
    # Calculate the loss for the original image
    loss = criterion(original_outputs, labels)
    # Calculate the perturbation
    model.zero_grad()
    loss.backward(retain_graph=True)
    data_grad = inputs.grad.data
    # Calculate the loss for the perturbed image
    perturbed_data = fgsm_attack(inputs.clone().detach(), data_grad, epsilon)
    perturbed_outputs = model(perturbed_data)
    perturbed_loss = criterion(perturbed_outputs, labels)
    # Combine the two losses
    loss = (1-alpha)*loss + alpha*perturbed_loss
    # Hint: the inputs are used in two different forward passes,
    # so you need to make sure those don't clash
    if return_preds:
        _, preds = torch.max(original_outputs, 1)
        return loss, preds
    else:
        return loss


def pgd_attack(model, data, target, criterion, args):
    alpha = args[ALPHA]
    epsilon = args[EPSILON]
    num_iter = args[NUM_ITER]

    # Implement the PGD attack
    # Start with a copy of the data
    # Then iteratively perturb the data in the direction of the gradient
    # Make sure to clamp the perturbation to the epsilon ball around the original data
    # Hint: to make sure to each time get a new detached copy of the data,
    # to avoid accumulating gradients from previous iterations
    # Hint: it can be useful to use toch.nograd()
    perturbed_data = data.clone().detach()
    original_data = data.clone().detach()

    for i in range(num_iter):
        perturbed_data.requires_grad = True
        model.zero_grad()
        output = model(perturbed_data)
        loss = criterion(output, target)
        loss.backward()
        with torch.no_grad():
            perturbation = alpha * perturbed_data.grad.sign()
            perturbed_data += perturbation

            eta = torch.clamp(perturbed_data - original_data, min=-epsilon, max=epsilon)
            perturbed_data = torch.clamp(original_data + eta, min=0, max=1)
    return perturbed_data


def test_attack(model, test_loader, attack_function, attack_args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    correct = 0
    criterion = nn.CrossEntropyLoss()
    adv_examples = []
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        data.requires_grad = True # Very important for attack!
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1] 

        # If the initial prediction is wrong, don't attack
        if init_pred.item() != target.item():
            continue

        loss = F.nll_loss(output, target)
        model.zero_grad()
        
        if attack_function == FGSM: 
            # Get the correct gradients wrt the data
            # Perturb the data using the FGSM attack
            # Re-classify the perturbed image
            loss.backward()
            data_grad = data.grad.data
            perturbed_data = fgsm_attack(data, data_grad, attack_args[EPSILON])
            output = model(perturbed_data)

        elif attack_function == PGD:
            # Get the perturbed data using the PGD attack
            # Re-classify the perturbed image
            perturbed_data = pgd_attack(model, data, target, criterion, attack_args)
            output = model(perturbed_data)
        else:
            print(f"Unknown attack {attack_function}")

        # Check for success
        final_pred = output.max(1, keepdim=True)[1] 
        if final_pred.item() == target.item():
            correct += 1
        else:
            # Save some adv examples for visualization later
            if len(adv_examples) < 5:
                original_data = data.squeeze().detach().cpu()
                adv_ex = perturbed_data.squeeze().detach().cpu()
                adv_examples.append( (init_pred.item(), 
                                      final_pred.item(),
                                      denormalize(original_data), 
                                      denormalize(adv_ex)) )

    # Calculate final accuracy
    final_acc = correct/float(len(test_loader))
    print(f"Attack {attack_function}, args: {attack_args}\nTest Accuracy = {correct} / {len(test_loader)} = {final_acc}")
    return final_acc, adv_examples