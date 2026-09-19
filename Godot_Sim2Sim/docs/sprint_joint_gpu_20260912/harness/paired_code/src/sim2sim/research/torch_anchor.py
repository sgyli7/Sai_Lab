"""Frozen, batched Torch execution of the small ONNX graphs used by this project.

Training-side adapter only. Export/deployment retain the original ONNX bytes;
unknown operators fail instead of silently substituting a different policy.
All stored tensor dtypes and explicit Cast nodes are preserved. Each source
must pass CPU ORT versus CUDA parity before its GPU rollouts are accepted.
"""
from pathlib import Path
import hashlib

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
import torch
from torch import nn
from torch.nn import functional as F


class TorchAnchor(nn.Module):
    def __init__(self, path):
        super().__init__()
        self.path = Path(path)
        raw = self.path.read_bytes()
        self.sha256 = hashlib.sha256(raw).hexdigest()
        model = onnx.load_model_from_string(raw)
        onnx.checker.check_model(model)
        if len(model.graph.input) != 1 or len(model.graph.output) != 1:
            raise ValueError('Expected one policy input and output')
        self.input_name = model.graph.input[0].name
        self.output_name = model.graph.output[0].name
        self.constants = {}
        self.operations = []
        supported = {'Sub', 'Add', 'Mul', 'Div', 'Gemm', 'MatMul', 'Elu',
                     'Tanh', 'Clip', 'Cast', 'Identity', 'Relu'}

        def constant(name, value):
            key = 'tensor_' + str(len(self.constants))
            self.register_buffer(key, torch.from_numpy(np.asarray(value).copy()))
            self.constants[name] = key

        for tensor in model.graph.initializer:
            constant(tensor.name, numpy_helper.to_array(tensor))
        known = {self.input_name, *self.constants}
        for node in model.graph.node:
            attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
            if node.domain not in ('', 'ai.onnx') or len(node.output) != 1:
                raise ValueError('Unsupported operator domain/output arity')
            if node.op_type == 'Constant':
                if set(attrs) != {'value'}:
                    raise ValueError('Only tensor-valued Constant is supported')
                constant(node.output[0], numpy_helper.to_array(attrs['value']))
            else:
                if node.op_type not in supported:
                    raise ValueError('Unsupported frozen operator: ' + node.op_type)
                if any(name and name not in known for name in node.input):
                    raise ValueError('Graph input dependency is unavailable')
                if node.op_type == 'Cast':
                    dtypes = {TensorProto.FLOAT: torch.float32, TensorProto.DOUBLE: torch.float64,
                              TensorProto.INT64: torch.int64, TensorProto.BOOL: torch.bool}
                    if attrs['to'] not in dtypes:
                        raise ValueError('Unsupported Cast dtype')
                    attrs['dtype'] = dtypes[attrs['to']]
                self.operations.append((node.op_type, tuple(node.input), node.output[0], attrs))
            known.add(node.output[0])
        if self.output_name not in known:
            raise ValueError('Missing policy output')

    def forward(self, obs):
        values = {name: getattr(self, key) for name, key in self.constants.items()}
        values[self.input_name] = obs
        for kind, names, output, attrs in self.operations:
            args = [values[name] if name else None for name in names]
            x = args[0]
            if kind == 'Sub': y = x - args[1]
            elif kind == 'Add': y = x + args[1]
            elif kind == 'Mul': y = x * args[1]
            elif kind == 'Div': y = x / args[1]
            elif kind == 'MatMul': y = x @ args[1]
            elif kind == 'Gemm':
                a = x.T if attrs.get('transA', 0) else x
                b = args[1].T if attrs.get('transB', 0) else args[1]
                if len(args) == 3 and args[2] is not None:
                    y = torch.addmm(args[2], a, b, beta=attrs.get('beta', 1.), alpha=attrs.get('alpha', 1.))
                else:
                    y = attrs.get('alpha', 1.) * (a @ b)
            elif kind == 'Elu': y = F.elu(x, alpha=attrs.get('alpha', 1.))
            elif kind == 'Tanh': y = torch.tanh(x)
            elif kind == 'Relu': y = torch.relu(x)
            elif kind == 'Identity': y = x
            elif kind == 'Cast': y = x.to(attrs['dtype'])
            elif kind == 'Clip':
                lo = args[1] if len(args) > 1 else attrs.get('min')
                hi = args[2] if len(args) > 2 else attrs.get('max')
                y = torch.clamp(x, min=lo, max=hi)
            else:
                raise RuntimeError('Unvalidated operator')
            values[output] = y
        return values[self.output_name]
