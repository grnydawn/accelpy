"""accelpy utility module"""

import os, abc, ast, json, hashlib
from subprocess import PIPE, run as subp_run

class Object(abc.ABC):

    def __new__(cls, *vargs, **kwargs):

        obj = super(Object, cls).__new__(cls)

        return obj

_builtin_excludes = ["exec", "eval", "breakpoint", "memoryview"]

_accelpy_builtins = dict((k, v) for k, v in __builtins__.items()
                       if k not in _builtin_excludes)

_config = None

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
    (vargs, kwargs), out = appeval("_appeval_p(%s)" % text, env)

    return kwargs

def shellcmd(cmd, shell=True, stdout=PIPE, stderr=PIPE,
             check=False):

    return subp_run(cmd, shell=shell, stdout=stdout,
                    stderr=stderr, check=check)

def which(pgm):
    path=os.getenv('PATH')
    for p in path.split(os.path.pathsep):
        p=os.path.join(p,pgm)
        if os.path.exists(p) and os.access(p,os.X_OK):
            return p

def init_config(cfgdir, overwrite=False):

    libdir = os.path.join(cfgdir, "lib")
    cfgfile = os.path.join(cfgdir, "config")

    for vendor in ["cray", "amd", "nvidia", "intel", "pgi", "ibm", "gnu"]:
        vendor_path = os.path.join(libdir, vendor)
        if not os.path.isdir(vendor_path):
            try:
                os.makedirs(vendor_path)

            except FileExistsError:
                pass

    config = {
        "libdir": libdir,
        "blddir": "",
    }

    if overwrite or not os.path.isfile(cfgfile):
        with open(cfgfile, "w")  as f:
            json.dump(config, f)


def _cfgfile():

    cfgdir = os.path.join(os.path.expanduser("~"), ".accelpy")
    redirect = os.path.join(cfgdir, "redirect")

    if os.path.isfile(redirect):
        with open(redirect) as f:
            cfgdir = f.read().strip()

    return os.path.join(cfgdir, "config")


def get_config(key):

    global _config

    if _config is None:
        with open(_cfgfile()) as f:
            _config = json.load(f)

    return _config[key]


def set_config(key, value, save=False):

    # TODO: multiprocessing check

    if _config is None:
        raise Exception("config file is not initialized.")

    _config[key] = value

    if save:
        with open(_cfgfile(), "w") as f:
            json.dump(_config, f, indent=4)


def fortline_pack(items):

    lines = [""]
    maxlen = 72

    for item in items:
        if len(lines[-1]) + len(item) > maxlen:            

            lines[-1] += " &"
            lines.append("        &, %s" % item)

        elif lines[-1] == "":
            lines[-1] += item

        else:
            lines[-1] += ", " + item

    return lines


def gethash(text, length=10):

    return hashlib.md5(text.encode("utf-8")).hexdigest()[:length]

