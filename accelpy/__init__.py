"""accelpy package"""

import atexit

def _load_config():
    import os, json, tempfile, time

    home = os.path.expanduser("~")

    cfgdir = os.path.join(home, ".accelpy")
    redirect = os.path.join(cfgdir, "redirect")

    if os.path.isfile(redirect):
        with open(redirect) as f:
            cfgdir = f.read().strip()

    if not os.path.isdir(cfgdir):
        os.makedirs(cfgdir)

    libdir = os.path.join(cfgdir, "lib")
    if not os.path.isdir(libdir):
        os.makedirs(libdir)

    _cfgfile = os.path.join(cfgdir, "config")

    if os.path.isfile(_cfgfile):
        with open(_cfgfile) as f:
            config = json.load(f)

        config["session"].clear()

    else:
        config = {
            "libdir": libdir,
            "blddir": "",
            "session": {
                "started_at": time.time(),
                "threads": {}
            }
        }

        with open(_cfgfile, "w") as f:
            json.dump(config, f, indent=4)

    if os.path.isdir(config["blddir"]):
        config["session"]["workdir"] = config["blddir"]

    else:
        config["session"]["tmpdir"] = tempfile.mkdtemp()
        config["session"]["workdir"] = config["session"]["tmpdir"]

    config["session"]["libdir"] = config["libdir"]
    config["session"]["started_at"] = time.time()

    return config


_config = _load_config()


@atexit.register
def _unload_config():
    import os, json, shutil

    if "session" in _config:
        session = _config["session"]

        if "tmpdir" in session:
            tmpdir = session["tmpdir"]

            if os.path.isdir(tmpdir):
                shutil.rmtree(tmpdir)

    _config["session"].clear()

    home = os.path.expanduser("~")

    cfgdir = os.path.join(home, ".accelpy")
    redirect = os.path.join(cfgdir, "redirect")

    if os.path.isfile(redirect):
        with open(redirect) as f:
            cfgdir = f.read().strip()

    _cfgfile = os.path.join(cfgdir, "config")

    try:
        with open(_cfgfile, "w") as f:
            json.dump(_config, f, indent=4)

    except IOError as err:
        print("Warning: can't write config due to read-only file system: %s" % _cfgfile) 


# accelerators
from .accel import Accel, AccelBase

# order
from .order import Order 

# import accelerator per their priority
from .fortran import FortranAccel
from .openacc_fortran import OpenaccFortranAccel
from .openmp_fortran import OpenmpFortranAccel
from .cpp import CppAccel
from .hip import HipAccel
from .cuda import CudaAccel
from .openacc_cpp import OpenaccCppAccel
from .openmp_cpp import OpenmpCppAccel

del _load_config, _unload_config, atexit
