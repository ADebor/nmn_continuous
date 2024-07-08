from omegaconf import OmegaConf
from nmn import SReLU
import importlib

# custom resolvers
if not OmegaConf.has_resolver("resolve_default"):
    OmegaConf.register_new_resolver(
        "resolve_default", lambda default, arg: default if arg == "" else arg
    )

if not OmegaConf.has_resolver("if"):
    OmegaConf.register_new_resolver(
        "if", lambda cdt, if_true, if_false: if_true if cdt else if_false
    )


# utils functions
def get_class_from_path(class_path):
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)
