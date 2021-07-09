# Copyright 2021 InterDigital R&D and Télécom Paris.
# Author: Giorgia Cantisani
# License: Apache 2.0

"""Utils for fine tuning
"""
import torch


BN_TYPES = (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)

def _make_trainable(module):
    """Unfreezes a given module.
    Args:
        module: The module to unfreeze
    """
    for param in module.parameters():
        param.requires_grad = True
    module.train()


def _recursive_freeze(module, train_bn=True):
    """Freezes the layers of a given module.
    Args:
        module: The module to freeze
        train_bn: If True, leave the BatchNorm layers in training mode
    """
    children = list(module.children())
    if not children:
        if not (isinstance(module, BN_TYPES) and train_bn):
            for param in module.parameters():
                param.requires_grad = False
            module.eval()
        else:
            # Make the BN layers trainable
            _make_trainable(module)
    else:
        for child in children:
            _recursive_freeze(module=child, train_bn=train_bn)


def freeze(module, n=None, train_bn=True):
    """Freezes the layers up to index n (if n is not None).
    Args:
        module: The module to freeze (at least partially)
        n: Max depth at which we stop freezing the layers. If None, all
            the layers of the given module will be frozen.
        train_bn: If True, leave the BatchNorm layers in training mode
    """
    modules = list(module.modules())
    n_max = len(modules) if n is None else int(n)

    for module in modules[:n_max]:
        _recursive_freeze(module=module, train_bn=train_bn)

    for module in modules[n_max:]:
        _make_trainable(module=module)


def filter_params(module, train_bn=True):
    """Yields the trainable parameters of a given module.
    Args:
        module: A given module
        train_bn: If True, leave the BatchNorm layers in training mode
    Returns:
        Generator
    """
    children = list(module.children())
    if not children:
        if not (isinstance(module, BN_TYPES) and train_bn):
            for param in module.parameters():
                if param.requires_grad:
                    yield param
    else:
        for child in children:
            for param in filter_params(module=child, train_bn=train_bn):
                yield param


def _unfreeze_and_add_param_group(module, optimizer, lr=None, train_bn=True):
    """Unfreezes a module and adds its parameters to an optimizer."""
    _make_trainable(module)
    params_lr = optimizer.param_groups[0]['lr'] if lr is None else float(lr)
    optimizer.add_param_group(
        {'params': filter_params(module=module, train_bn=train_bn),
         'lr': params_lr / 10.,
         })


# Test 
if __name__ == '__main__':
    # Load pretrained model
    klass, args, kwargs, state = torch.load('./demucs/tasnet.th', 'cpu')
    model = klass(*args, **kwargs)
    model.load_state_dict(state)
    
    # Select individual modules
    encoder = model.encoder
    separator = model.separator
    decoder = model.decoder

    # Freeze some layers
    freeze(encoder)
    freeze(separator, n=338)

    # Count trainable parameters
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    nr_parameters = count_parameters(model)
    print(nr_parameters)

    # Initialize an optimizer
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)