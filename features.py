import torch
import torch.nn.functional as F
import torchvision.transforms


def odin_fgsm(model, image):
    image.requires_grad = True

    output = model(image)
    if isinstance(output, list):
        output = output[1]  #for njord
    nnOutputs = output
    nnOutputs = nnOutputs
    nnOutputs = nnOutputs[0]
    nnOutputs = nnOutputs - torch.max(nnOutputs)
    nnOutputs = torch.exp(nnOutputs) / torch.sum(torch.exp(nnOutputs))
    nnOutputs = nnOutputs.unsqueeze(0)
    maxIndexTemp = torch.argmax(nnOutputs)
    loss = model.criterion(nnOutputs, torch.ones_like(output)*maxIndexTemp)
    loss.backward()
    data_grad = image.grad.data
    data_grad = data_grad.squeeze(0)
    # perturb image
    perturbed_image = image + data_grad.sign() * 0.1
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

def odin(model, image, feature_transform):
    perturbed_image = odin_fgsm(model, image)
    val =  cross_entropy(model, perturbed_image)
    return val
def cross_entropy(model, image, num_features=1):
    with torch.no_grad():
        out = model(image)
        if isinstance(out, list):
            out = out[1]  #for njord
        return model.criterion(out, torch.ones_like(out)).cpu().numpy()
def grad_magnitude(model, image, num_features=1):
    image.requires_grad = True
    output = model(image)
    if isinstance(output, list):
        output = output[1]  #for njord
    loss = model.criterion(output, torch.ones_like(output)).mean()
    model.zero_grad()
    loss.backward()
    data_grad = image.grad.data
    return torch.norm(data_grad, "fro"  ).item()


def typicality_ks_glow(model, img, num_features=1):
    assert num_features==1
    image = img
    # image = torchvision.transforms.Resize((32,32))(img)
    # image = image * 255
    # n_bins = 2.0 ** 5
    # image = torch.floor(image / 2 ** (8 - 5))
    #
    # image = image / n_bins - 0.5
    return -model.estimate_log_likelihood(image)
