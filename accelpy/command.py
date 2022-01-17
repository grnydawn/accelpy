"""main entry for accelpy command-line interface"""

import accelpy


def cmd_config(args):
    import os

    if args.dir:
        redirect = os.path.join(os.path.expanduser("~"), ".accelpy", "redirect")
        with open(redirect, "w") as f:
            f.write(args.dir)


def main():
    import argparse
    from accelpy.core import version

    parser = argparse.ArgumentParser(description="accelpy command-line tool")
    parser.add_argument("--version", action="version", version="accelpy "+version)
    parser.add_argument("--verbose", action="store_true", help="verbose info")

    cmds = parser.add_subparsers(title='subcommands',
                description='accelpy subcommands', help='additional help')

    p_info = cmds.add_parser('config')
    p_info.add_argument("-d", "--dir", help="set path for config files")
    p_info.set_defaults(func=cmd_config)

    argps = parser.parse_args()
    argps.func(argps)

    return 0
