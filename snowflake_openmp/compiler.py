from __future__ import division, print_function
import ast
import copy
from ctypes import c_long
from ctree.c.nodes import MultiNode, Pragma, FunctionDecl, For, Assign, SymbolRef
import operator
from ctree.cpp.nodes import CppInclude
from ctree.frontend import dump
from snowflake.analytics import validate_stencil, AnalysisError, create_dependency_graph
from snowflake.compiler_nodes import Space, IterationSpace, NDSpace
from snowflake.compiler_utils import calculate_ND_volume, is_homogenous_space, StencilShifter, BlockCombineTransformer
from snowflake.stencil_compiler import CCompiler
import itertools
import numpy as np
from snowflake.vector import Vector
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
        def visit_FunctionDecl(self, node):
            for ind, child in enumerate(node.defn):
                if isinstance(child, For):
                    node.defn[ind] = Pragma("omp for", body=[child])
            return node

    class MakeSingle(ast.NodeTransformer):
        def __init__(self, dependency_graph, stencil_ids, ndim):
            self.graph = dependency_graph  # graph[a][b] = does a depend on b?
            self.stencil_ids = stencil_ids  # a list of the original stencils.
            self.ndim = ndim

        def visit_Pragma(self, node): # They'll be inside an omp parallel pragma
            if node.pragma != "omp parallel":
                return node
            new_body = []
            group = []
            dependency_ids = []
            # def stop():
            #     return Pragma("omp taskwait", []), Pragma("omp barrier", [])
            for stencil_id, child in zip(self.stencil_ids, node.body):
                # Cases:
                # task -> has dependency
                # task -> no dependency
                # parallel task region -> has dependency
                # parallel task region -> no dependency
                has_dependency = any(self.graph[stencil_id][dep] for dep in dependency_ids)
                # print(dependency_ids, stencil_id)
                # print(has_dependency)
                if isinstance(child, Pragma) and child.pragma == "omp task":
                    # task
                    if has_dependency:
                        # if we're in a task and we have a dependency
                        group.append(Pragma("omp taskwait", []))

                        dependency_ids = []  # reset the dependencies
                    dependency_ids.append(stencil_id)
                    group.append(child)

                else:
                    if group:
                        new_body.append(  # offload the previous group if there is one
                            Pragma("omp single", body=group, braces=True)
                        )
                        if has_dependency:
                            new_body.append(Pragma("omp barrier", []))
                            new_body.append(Pragma("omp taskwait", []))
                            dependency_ids = []
                        group = []
                        dependency_ids.append(stencil_id)
                        new_body.append(child)
                    # parallel task region
                    else:
                        if has_dependency:
                            new_body.append(Pragma("omp barrier", []))
                            new_body.append(Pragma("omp taskwait", []))
                            dependency_ids = []
                        dependency_ids.append(stencil_id)
                        new_body.append(child)
            if group:  # Might have remaining single group waiting. Will not have dependencies.
                new_body.append(
                    Pragma("omp single", body=group, braces=True)
                )
            node.body = new_body
            return node

    class Privatize(ast.NodeTransformer):
        def __init__(self):
            self.stack = []

        def visit_For(self, node):
            """
            We match the pattern
            for a
                for b
                    etc...
                        #pragma omp task

            and replace it with
            for a
                for b
                    etc...
                        #pragma omp task firstprivate(a, b, etc...)
            """
            self.stack.append(
                node.init.left.name # node.init is an assign node, assign.left is a symbolref.
            )
            self.generic_visit(node)
            self.stack.pop()
            return node

        def visit_Pragma(self, node):
            if node.pragma == "omp task" and self.stack:
                node.pragma += " firstprivate({})".format(", ".join(self.stack))
            return self.generic_visit(node)



    class LazySpecializedKernel(CCompiler.LazySpecializedKernel):
        def __init__(self, py_ast=None, names=None, target_names=('out',), index_name='index',
                     _hash=None, original=None):
            super(OpenMPCompiler.LazySpecializedKernel, self).__init__(
                py_ast, names, target_names, index_name, _hash, original
            )
            self.parent_cls = OpenMPCompiler

        def transform(self, tree, program_config):
            subconfig, tuning_config = program_config
            name_shape_map = {name: arg.shape for name, arg in subconfig.items()}
            ndim = len(name_shape_map.values()[0])
            print("analyzing", len(self.original.body))
            dependency_graph = create_dependency_graph(self.original, name_shape_map)
            stencil_ids = [hash(s) for s in self.original.body]
            print("done analyzing")
            result = super(OpenMPCompiler.LazySpecializedKernel, self).transform(tree, program_config)
            result.config_target = 'omp'
            result.body.insert(0, CppInclude("omp.h"))
            result = BlockCombineTransformer().visit(result)
            result = self.parent_cls.ParallelForTasks().visit(result)

            node = result.find(FunctionDecl)
            node.defn = [Pragma("omp parallel", body=node.defn, braces=True)]
            result = self.parent_cls.MakeSingle(dependency_graph, stencil_ids, ndim).visit(result)
            result = self.parent_cls.Privatize().visit(result)

            return result