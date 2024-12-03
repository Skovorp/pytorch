# Owner(s): ["module: inductor"]
import importlib
from typing import Any, Callable, List, Optional

import torch
import torch.utils._pytree as pytree
from torch._inductor import config
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)
from torch.testing._internal.inductor_utils import HAS_CPU, HAS_GPU


importlib.import_module("filelock")


@instantiate_parametrized_tests
class CodegenInductorTest(InductorTestCase):
    def run_and_compare(
        self,
        func: Callable[..., Any],
        *args,
        compile_kwargs: Optional[dict] = None,
        config_patches: Optional[dict] = None,
    ):
        """
        Runs the module through Inductor, comparing to eager reference.
        """
        if compile_kwargs is None:
            compile_kwargs = {}
        if config_patches is None:
            config_patches = {}

        def flatten_tensors(tensors):
            flat, spec = pytree.tree_flatten(tensors)
            return flat

        with config.patch(config_patches):
            compiled = torch.compile(func, backend="inductor", **compile_kwargs)
            result, code = run_and_get_code(compiled, *args)

        # Check numerical accuracy
        ref_tensors = flatten_tensors(func(*args))
        actual_tensors = flatten_tensors(result)
        for ref, actual in zip(ref_tensors, actual_tensors):
            self.assertTrue(torch.allclose(ref, actual))

        return result, code

    def count_code(self, substr: str, code: List[str], expected: Optional[int]):
        count = sum(prog.count(substr) for prog in code)
        if expected is not None:
            self.assertEqual(count, expected)

    @parametrize("force_pointwise_cat", [False, True])
    def test_pointwise_cat(self, force_pointwise_cat: bool):
        def func(a, b):
            return torch.cat([a + 1, b + 2], dim=0)

        a = torch.randn(1024, device=torch.device("cpu"))
        b = torch.randn(1024, device=torch.device("cpu"))
        config_patches = {
            "force_pointwise_cat": force_pointwise_cat,
        }
        _, code = self.run_and_compare(
            func,
            a,
            b,
            config_patches=config_patches,
        )

        if force_pointwise_cat:
            self.count_code("= reinterpret_tensor(", code, 0)
        else:
            self.count_code("= reinterpret_tensor(", code, 2)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if HAS_GPU or HAS_CPU:
        run_tests(needs="filelock")
