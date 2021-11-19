"""accelpy utility module"""

import ast
from subprocess import PIPE, run as subp_run

_builtin_excludes = ["exec", "eval", "breakpoint", "memoryview"]

_accelpy_builtins = dict((k, v) for k, v in __builtins__.items()
                       if k not in _builtin_excludes)

def appeval(text, env):

    if not text:
        return [], {}

    if not isinstance(text, str):
        raise Exception("Not a string")

    val = None
    lenv = {}

    stmts = ast.parse(text).body

    if len(stmts) == 1 and isinstance(stmts[-1], ast.Expr):
        val = eval(text, env, lenv)

    else:
        exec(text, env, lenv)

    return val, lenv


def funcargseval(text, lenv):

    def _p(*argv, **kw_str):
        return list(argv), kw_str

    env = dict(_accelpy_builtins)
    if isinstance(lenv, dict):
        env.update(lenv)

    env["_appeval_p"] = _p
    fargs, out = appeval("_appeval_p(%s)" % text, env)

    return fargs

def shellcmd(cmd, shell=True, stdout=PIPE, stderr=PIPE,
             check=False):

    return subp_run(cmd, shell=shell, stdout=stdout,
                    stderr=stderr, check=check)
