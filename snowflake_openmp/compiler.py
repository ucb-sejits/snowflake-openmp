from __future__ import division, print_function
import ast
import copy
from ctree.c.nodes import MultiNode, Pragma, FunctionDecl, For
import operator
from ctree.cpp.nodes import CppInclude
from ctree.frontend import dump
from snowflake.analytics import validate_stencil, AnalysisError
from snowflake.compiler_nodes import Space, IterationSpace, NDSpace
from snowflake.compiler_utils import calculate_ND_volume, is_homogenous_space, StencilShifter
from snowflake.stencil_compiler import CCompiler
import itertools
import numpy as np
from snowflake.vector import Vector

import os

os.environ["GOMP_CPU_AFFINITY"] = "0-7"
os.environ["OMP_NUM_THREADS"] = "8"

__author__ = 'nzhang-dev'

class OpenMPCompiler(CCompiler):
    class IterationSpaceExpander(CCompiler.IterationSpaceExpander):

        def visit_IterationSpace(self, node):
            if is_homogenous_space(node.space) and len(node.space.spaces) > 1:
                node = self._visit_homogenous_space(node)
                # print(dump(node))
            result = super(OpenMPCompiler.IterationSpaceExpander, self).visit_IterationSpace(node)
            return MultiNode([Pragma(pragma="omp task", body=[child], braces=True) for child in result.body])
            # return Pragma(pragma="omp parallel", body=[result], braces=True)

        def _visit_homogenous_space(self, node):
            offset_versions = []

            # so that we can apply all of the spaces simultaneously
            for space in node.space.spaces:
                cp = copy.deepcopy(node.body)
                shifter = StencilShifter(space.low)
                output = [shifter.visit(part) for part in cp]
                offset_versions.extend(output)
            low = Vector((0,) * node.space.ndim)
            high = node.space.spaces[0].high - node.space.spaces[0].low
            stride = node.space.spaces[0].stride
            return IterationSpace(
                space=NDSpace([Space(low, high, stride)]),
                body=offset_versions
            )

    class ParallelForTasks(ast.NodeTransformer):
        def visit_MultiNode(self, node):
            for ind, child in enumerate(node.body):
                if isinstance(child, For):
                    node.body[ind] = Pragma("omp for", body=[child])
            return node

    class MakeSingle(ast.NodeTransformer):
        def visit_MultiNode(self, node):
            new_body = []
            group = []
            for child in node.body:
                if isinstance(child, Pragma):
                    group.append(child)
                else:
                    if group:
                        new_body.append(
                            Pragma("omp master", body=group, braces=True)
                        )
                        group = []
                    new_body.append(child)
            if group:
                new_body.append(
                    Pragma("omp master", body=group, braces=True)
                )
            node.body = new_body
            return node




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
            node = result.find(FunctionDecl)
            node.defn = [Pragma("omp parallel", body=node.defn, braces=True)]
            result = self.parent_cls.MakeSingle().visit(result)
            return self.parent_cls.ParallelForTasks().visit(result)

    def _compile(self, node, index_name, **kwargs):
        # if not all(validate_stencil(stencil) for stencil in node.body):
        #     raise AnalysisError("Analysis found Loop-Carry Dependency in stencil. Cannot proceed")
        return super(OpenMPCompiler, self)._compile(node, index_name, **kwargs)