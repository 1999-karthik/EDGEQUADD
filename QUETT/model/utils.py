import torch
from .quett_model import QuETT
from .quadratic_components import ASPPQuadraticAdapter


def make_param_groups(model: QuETT, base_lr=1e-4, adapter_lr=6e-4, weight_decay=0.01):
    fused = getattr(model, 'is_fused', False)
    if isinstance(fused, torch.Tensor):
        fused = bool(fused.item())
    if fused:
        return [{'params': list(model.parameters()), 'lr': adapter_lr, 'weight_decay': weight_decay}]
    
    base_params, adapter_params, gate_params = [], [], []

    for m in model.adapters:
        if isinstance(m, ASPPQuadraticAdapter):
            base_params += [m.base.weight, m.base.bias]
            # ASPPQuadraticAdapter uses low-rank decomposition with gating
            if m.rank > 0:
                adapter_params += [m.Ux.weight, m.Vc.weight, m.Wo.weight, m.alpha, m.gate.weight, m.gate.bias, m.gate_temp]
            else:
                adapter_params += [m.quad.weight, m.quad.bias, m.alpha, m.gate.weight, m.gate.bias, m.gate_temp]

    # Add gate conv parameters with L2 regularization
    if hasattr(model, 'gate_conv'):
        gate_params += [model.gate_conv.weight, model.gate_conv.bias]

    base_params    = [p for p in base_params    if p is not None]
    adapter_params = [p for p in adapter_params if p is not None]
    gate_params    = [p for p in gate_params    if p is not None]

    base_ids = {id(p) for p in base_params}
    adap_ids = {id(p) for p in adapter_params}
    gate_ids = {id(p) for p in gate_params}
    grouped_other = [p for p in model.parameters() if id(p) not in base_ids and id(p) not in adap_ids and id(p) not in gate_ids]

    param_groups = [
        {'params': base_params,    'lr': base_lr,    'weight_decay': weight_decay},
        {'params': adapter_params, 'lr': adapter_lr, 'weight_decay': weight_decay},
        {'params': grouped_other,  'lr': adapter_lr, 'weight_decay': weight_decay},
    ]
    
    # Add gate params with L2 regularization if they exist
    if gate_params:
        param_groups.append({'params': gate_params, 'lr': adapter_lr, 'weight_decay': 1e-4})
    
    return param_groups