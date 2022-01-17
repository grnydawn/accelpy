"""main entry for accelpy command-line interface"""

import os
from accelpy import _config

cfgdir = os.path.join(os.path.expanduser("~"), ".accelpy")
redirect = os.path.join(cfgdir, "redirect")


def cmd_config(args):

    import json

    if args.dir:
        with open(redirect, "w") as f:
            f.write(args.dir)
    else:
        if os.path.isfile(redirect):
            with open(redirect) as f:
                cfgdir = f.read().strip()

        if args.libdir:
            cfgfile = os.path.join(cfgdir, "config")
            _config["libdir"] = args.libdir

            with open(cfgfile, "w") as f:
                json.dump(_config, f, indent=4)

            
def cmd_cache(args):

    import shutil

    if os.path.isfile(redirect):
        with open(redirect) as f:
            cfgdir = f.read().strip()

    if args.clear_all:
        libdir = _config["libdir"]

        if os.path.isdir(libdir):
            for item in os.listdir(libdir):

                itempath = os.path.join(libdir, item)
                if os.path.isdir(itempath):
                    shutil.rmtree(itempath)

                else:
                    os.remove(itempath)


def main():
    import argparse
    from accelpy.core import version

    parser = argparse.ArgumentParser(description="accelpy command-line tool")
    parser.add_argument("--version", action="version", version="accelpy "+version)
    parser.add_argument("--verbose", action="store_true", help="verbose info")

    cmds = parser.add_subparsers(title='subcommands',
                description='accelpy subcommands', help='additional help')

    p_config = cmds.add_parser('config')
    p_config.add_argument("-d", "--dir", help="set path for config files")
    p_config.add_argument("-l", "--libdir", help="set path for library cache files")
    p_config.set_defaults(func=cmd_config)

    p_cache = cmds.add_parser('cache')
    p_cache.add_argument("-a", "--clear-all", action="store_true",
                            help="clear all caches")
    p_cache.set_defaults(func=cmd_cache)

    argps = parser.parse_args()

    if hasattr(argps, "func"):
        argps.func(argps)

    else:
        parser.print_help()


    return 0
