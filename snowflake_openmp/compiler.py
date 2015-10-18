from __future__ import division, print_function
from ctree.c.nodes import MultiNode, Pragma
import operator
from ctree.cpp.nodes import CppInclude
from snowflake.analytics import validate_stencil, AnalysisError
from snowflake.compiler_nodes import Space
from snowflake.stencil_compiler import CCompiler
import itertools
import numpy as np
from snowflake.vector import Vector

__author__ = 'nzhang-dev'

class OpenMPCompiler(CCompiler):
    class IterationSpaceExpander(CCompiler.IterationSpaceExpander):
        def visit_IterationSpace(self, node):
            result = super(OpenMPCompiler.IterationSpaceExpander, self).visit_IterationSpace(node)
            result = MultiNode([
                Pragma(pragma="omp for nowait", body=[child]) for child in result.body
            ])
            return Pragma(pragma="omp parallel", body=[result], braces=True)

    class LazySpecializedKernel(CCompiler.LazySpecializedKernel):
        def __init__(self, py_ast=None, names=None, target_names=('out',), index_name='index',
                     _hash=None):
            super(OpenMPCompiler.LazySpecializedKernel, self).__init__(
                py_ast, names, target_names, index_name, _hash
            )
            self.parent_cls = OpenMPCompiler

        def transform(self, tree, program_config):
            result = super(OpenMPCompiler.LazySpecializedKernel, self).transform(tree, program_config)
            result.config_target = 'omp'
            result.body.insert(0, CppInclude("omp.h"))
            return result

    def _compile(self, node, index_name, **kwargs):
        if not all(validate_stencil(stencil) for stencil in node.body):
            raise AnalysisError("Analysis found Loop-Carry Dependency in stencil. Cannot proceed")
        return super(OpenMPCompiler, self)._compile(node, index_name, **kwargs)

class TiledOpenMPCompiler(OpenMPCompiler):
    class IterationSpaceExpander(OpenMPCompiler.IterationSpaceExpander):
        def visit_IterationSpace(self, node):
            spaces = node.space.spaces
            if len(self.block_size) > len(self.reference_array_shape):
                raise ValueError("Block size has more dimensions than the array")
            num_blocks = np.array([shape/size for shape, size in
                               itertools.izip_longest(self.reference_array_shape, self.block_size, fillvalue=1)
                               ])
            relative_densities = np.array([
                space.stride for space in spaces
            ])

            totals = np.sum(relative_densities, axis=0)
            scaled = np.round(relative_densities / totals * num_blocks + 0.5) #how many tasks to split into
            node.space.spaces = []
            for scale, space in zip(scaled, spaces):
                for scaling_index in itertools.product(*[range(int(i)) for i in scale]):
                    node.space.spaces.append(
                        Space(
                            space.low + space.stride*scaling_index,
                            space.high,
                            space.stride*scale
                        )
                    )

            result = super(OpenMPCompiler.IterationSpaceExpander, self).visit_IterationSpace(node)
            result = MultiNode([
                Pragma(pragma="omp section", body=[child], braces=True) for child in result.body
            ])
            return Pragma(pragma="omp parallel sections", body=[result], braces=True)

    class LazySpecializedKernel(OpenMPCompiler.LazySpecializedKernel):
        def __init__(self, py_ast=None, names=None, target_names=('out',), index_name='index',
                     _hash=None):
            super(OpenMPCompiler.LazySpecializedKernel, self).__init__(
                py_ast, names, target_names, index_name, _hash
            )
            self.parent_cls = TiledOpenMPCompiler

        def transform(self, tree, program_config):
            result = super(TiledOpenMPCompiler.LazySpecializedKernel, self).transform(tree, program_config)
            result.config_target = 'omp'
            result.body.insert(0, CppInclude("omp.h"))
            return result