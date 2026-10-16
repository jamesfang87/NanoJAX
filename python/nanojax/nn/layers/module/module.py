from __future__ import annotations

import copy

import nanojax as nj


class Module:
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def named_parameters(self, prefix: str = "") -> dict[str, nj.Tensor]:
        params: dict[str, nj.Tensor] = {}
        for name, value in self.__dict__.items():
            if isinstance(value, nj.Tensor):
                params[prefix + name] = value
            elif isinstance(value, Module):
                params.update(value.named_parameters(prefix + name + "."))
        return params

    def parameters(self) -> list[nj.Tensor]:
        return list(self.named_parameters().values())

    def with_parameters(self, replacements: dict[str, nj.Tensor]) -> Module:
        # First create a shallow copy for copy-on-write
        updated = copy.copy(self)

        for path, replacement in replacements.items():
            current = updated
            *parents, attribute = path.split(".")

            for name in parents:
                # Get the parent from its name
                parent = getattr(current, name)

                # Copy on write
                copied_child = copy.copy(parent)
                setattr(current, name, copied_child)

                current = copied_child

            setattr(current, attribute, replacement)

        return updated
