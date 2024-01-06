from inspect import isfunction


def get_named_function(module):
    """Get the class member in module."""
    return {k: v for k, v in module.__dict__.items() if isfunction(v)}
