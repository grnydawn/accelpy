"""accelpy package"""

import atexit

def _load_config():
    import os, json, tempfile, time

    home = os.path.expanduser("~")

    cfgdir = os.path.join(home, ".accelpy")
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
    _cfgfile = os.path.join(cfgdir, "config")

    if os.path.isfile(_cfgfile):
        with open(_cfgfile, "w") as f:
            json.dump(_config, f, indent=4)


# accelerators
from .accel import Accel, AccelBase

# import accelerator per their priority
from .cpp import CppAccel
from .fortran import FortranAccel
from .hip import HipAccel
from .cuda import CudaAccel

del _load_config, _unload_config, atexit
