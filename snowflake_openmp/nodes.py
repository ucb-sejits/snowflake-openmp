import ast


class ParFor(ast.AST):
    _fields = ['init', 'test', 'incr', 'body']

    def __init__(self, init=None, test=None, incr=None, body=None, pragma=None):
        self.init = init
        self.test = test
        self.incr = incr
        if body is None:
            body = []
        self.body = body
        self.pragma = pragma