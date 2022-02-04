"""main entry for accelpy command-line interface"""

import os
from accelpy.util import init_config, get_config, set_config
from accelpy.const import version

def cmd_config(args):

    import json

    if args.dir:

        cfgdir = os.path.join(os.path.expanduser("~"), ".accelpy")
        redirect = os.path.join(cfgdir, "redirect")

        with open(redirect, "w") as f:
            f.write(args.dir)

        init_config(args.dir)

    elif args.libdir:
        set_config("libdir", args.libdir, save=True)

            
def cmd_cache(args):

    import shutil

    if args.clear_all:
        libdir = get_config("libdir")

        if os.path.isdir(libdir):
            for item in os.listdir(libdir):

                itempath = os.path.join(libdir, item)
                if os.path.isdir(itempath):
                    shutil.rmtree(itempath)

                else:
                    os.remove(itempath)


def main():
    import argparse

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
