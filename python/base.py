from dataclasses import dataclass
from typing import Optional, Callable
from contextlib import contextmanager

# Similar to config in the beginning of
# eval_frame.py
@dataclass
class DynamoStance:
    stance: str = "default"
    backend: Optional[Callable] = None

_stance = DynamoStance()

def set_stance(stance: DynamoStance) -> DynamoStance:
    global _stance
    prior = _stance
    _stance = stance
    return prior

# "with stance_context("eager"): " runs the function up the the yield
# Then runs the indented code within statement
# Then runs the the code after finally
@contextmanager
def stance_context(stance: str):
    prior = set_stance(DynamoStance(stance=stance))
    try:
        yield
    finally:
        set_stance(prior)

# Returns a function based on the stance
def callback_from_stance():
    if _stance.stance == "eager":
        return None
    elif _stance.stance == "default":
        return _stance.backend

# guards.py:1895
def type_guard(x):
    t = type(x)
    return lambda y: type(y) == t

# guards.py:2121
def value_guard(x):
    val = x
    return lambda y: y == val

# guards.py:2698
def shape_guard(x):
    if hasattr(x, 'shape'):
        shape = x.shape
        return lambda y: hasattr(y, 'shape') and y.shape == shape
    return lambda y: True

def dtype_guard(x):
    if hasattr(x, 'dtype'):
        dtype = x.dtype
        return lambda y: hasattr(y, 'dtype') and y.dtype == dtype
    return lambda y: True

class Guard:
    def __init__(self, predicates: list):
        self.predicates = predicates

    def check(self, x) -> bool:
        return all(p(x) for p in self.predicates)

# X is the value we are building the guard around
def build_guard(x) -> Guard:
    predicates = [type_guard(x)]

    if isinstance(x, bool):
        predicates.append(value_guard(x))
    elif isinstance(x, int) and abs(x) < 100:
        predicates.append(value_guard(x))
    elif isinstance(x, str):
        predicates.append(value_guard(x))

    if hasattr(x, 'shape'):
        predicates.append(shape_guard(x))
    if hasattr(x, 'dtype'):
        predicates.append(dtype_guard(x))

    # Wrapped in a guard so the check can be run
    return Guard(predicates)

_trace: list = []

# The Purpose of SymTensor is somewhat self explanatory
# No operations are actually run and they are all meant to
# be logged instead this is similar to the way it is handled
# in Pytorch and JAX
class SymTensor:
    _counter = 0

    def __init__(self, dim: list, id: str = None, trainable: bool = False, init: str = None):
        self.dim = dim
        self.trainable = trainable
        self.init = init
        if id is None:
            SymTensor._counter += 1
            self.id = f"t{SymTensor._counter}"
        else:
            self.id = id

    # Meant to have the function work with the @ symbol
    # SymTensor has quotes because it hasn't been defined yet
    def __matmul__(self, other: "SymTensor") -> "SymTensor":
        out_dim = [self.dim[0], other.dim[1]]
        out = SymTensor(out_dim)
        _trace.append({"id": out.id, "op": "matmul", "args": [self.id, other.id]})
        return out

# Creates a SymTensor while also appending to the trace
def sym_relu(x: SymTensor) -> SymTensor:
    out = SymTensor(x.dim)
    _trace.append({"id": out.id, "op": "relu", "args": [x.id]})
    return out

def sym_softmax(x: SymTensor) -> SymTensor:
    out = SymTensor(x.dim)
    _trace.append({"id": out.id, "op": "softmax", "args": [x.id]})
    return out

def sym_cross_entropy(x: SymTensor, num_classes: int) -> SymTensor:
    out = SymTensor([1])
    _trace.append({"id": out.id, "op": "cross_entropy", "args": [x.id], "dim": [1, num_classes]})
    return out

def trace(fn, *sym_args) -> list:
    global _trace
    SymTensor._counter = 0
    _trace = []
    for arg in sym_args:
        node = {"id": arg.id, "op": "const", "dim": arg.dim}
        if arg.trainable:
            node["trainable"] = True
        if arg.init:
            node["init"] = arg.init
        _trace.append(node)
    fn(*sym_args)
    return list(_trace)

class OptimizeContext:
    def __init__(self, cache_size_limit=8):
        # pairs a Guard to a compiled function
        self.cache: list[tuple[Guard, object]] = []
        self.cache_size_limit = cache_size_limit

    
    # Call allows OptimizeContext to work as a decorator
    # fn is the function and wrapper is what replaces it
    def __call__(self, fn):
        def wrapper(*args):
            # Get first input
            x = args[0]

            # Call fn directly
            if _stance.stance == "eager":
                print(f"  Eager mode — skipping cache for {x!r}")
                return fn(x)

            # Compile if it is found in the cache
            for guard, compiled in self.cache:
                if guard.check(x):
                    print(f"  Guard passed for {x!r}")
                    return compiled(x)
 
            if len(self.cache) >= self.cache_size_limit:
                print(f"  Cache full — falling back to eager for {x!r}")
                return fn(x)

            print(f"  Compiling for {x!r} ...")
            guard = build_guard(x)
            compiled = fn
            self.cache.append((guard, compiled))
            return compiled(x)

        return wrapper


def optimize():
    return OptimizeContext()

@optimize()
def my_func(x):
    return x * 2

import json
import tensor_frontend

# Define your network in Python
def mnist_net(x, w1, w2):
    h = sym_relu(x @ w1)
    logits = h @ w2
    probs = sym_softmax(logits)
    sym_cross_entropy(probs, 10)

x  = SymTensor([1, 784],   id="x")
w1 = SymTensor([784, 128], id="w1", trainable=True, init="xavier")
w2 = SymTensor([128, 10],  id="w2", trainable=True, init="xavier")

ir = trace(mnist_net, x, w1, w2)
print("=== traced IR ===")
print(json.dumps(ir, indent=2))

# Compile: FusionPass runs, matmul+relu → matmul_relu
model = tensor_frontend.compile(ir)
model.print_graph()

# Train on MNIST
losses = model.train(
    img_path="../data/MNIST/raw/train-images-idx3-ubyte",
    lbl_path="../data/MNIST/raw/train-labels-idx1-ubyte",
    lr=0.01,
    epochs=3,
    n_samples=1000
)
