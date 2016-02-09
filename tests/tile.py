from snowflake.nodes import Stencil, WeightArray, StencilComponent

import numpy as np
from snowflake.stencil_compiler import CCompiler
from snowflake_openmp.compiler import OpenMPCompiler

__author__ = 'nzhang-dev'


def run_test():
    weight_array = WeightArray(
        np.ones((3, 3, 3), dtype=np.float) / 27
    )
    component = StencilComponent(
        "input",
        weight_array
    )
    stencil = Stencil(
        component,
        "output",
        [(1, -1, 1)]*3
    )
    ccompiler = OpenMPCompiler()
    kern = ccompiler.compile(stencil)
    arr = np.arange(66**3, dtype=np.float).reshape((66, 66, 66))
    out = np.zeros_like(arr)
    kern(out, arr)

if __name__ == "__main__":
    run_test()