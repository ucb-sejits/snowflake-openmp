from ctree.c.nodes import MultiNode, Pragma
from snowflake.analytics import validate_stencil, AnalysisError
from snowflake.stencil_compiler import CCompiler

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
            return result

    def _compile(self, node, index_name, **kwargs):
        if not all(validate_stencil(stencil) for stencil in node.body):
            raise AnalysisError("Analysis found Loop-Carry Dependency in stencil. Cannot proceed")
        return super(OpenMPCompiler, self)._compile(node, index_name, **kwargs)