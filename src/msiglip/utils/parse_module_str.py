import importlib


def parse_module_str(module: str):
    legacy_prefixes = {
        "model.": "msiglip.model.",
        "data.": "msiglip.data.",
        "utils.": "msiglip.utils.",
        "solver.": "msiglip.solver.",
    }
    for old_prefix, new_prefix in legacy_prefixes.items():
        if module.startswith(old_prefix):
            module = new_prefix + module[len(old_prefix):]
            break

    from_modules, imported = module.rsplit(".", 1)
    get_module = getattr(importlib.import_module(from_modules), imported)
    return get_module
