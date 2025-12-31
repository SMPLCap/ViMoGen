import argparse
import importlib
import subprocess

mbench_cmd = ['evaluate']

def main():
    parser = argparse.ArgumentParser(prog="mbench", formatter_class=argparse.RawTextHelpFormatter)
    subparsers = parser.add_subparsers(title='mbench subcommands')

    for cmd in mbench_cmd:
        module = importlib.import_module(f'mbench.cli.{cmd}')
        module.register_subparsers(subparsers)
    parser.set_defaults(func=help)
    args = parser.parse_args()
    args.func(args)

def help(args):
    subprocess.run(['mbench', '-h'], check=True)