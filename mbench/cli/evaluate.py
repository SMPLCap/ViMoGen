import os
import shlex
import subprocess
import argparse

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
def register_subparsers(subparser):
    parser = subparser.add_parser('evaluate', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--ngpus",
        type=int,
        default=1,
        help="Number of GPUs to run evaluation on"
        )
    parser.add_argument(
        "--output_path",
        type=str,
        default='./evaluation_results/',
        help="output path to save the evaluation results",
    )
    parser.add_argument(
        "--full_json_dir",
        type=str,
        default=f'{CUR_DIR}/../MBench_full_info.json',
        help="path to save the json file that contains the prompt and dimension information",
    )
    parser.add_argument(
        "--evaluation_path",
        type=str,
        required=True,
        help="folder that contains the model generated results",
    )
    parser.add_argument(
        "--dimension",
        type=str,
        required=True,
        help="list of evaluation dimensions, usage: --dimension <dim_1> <dim_2>",
    )
    parser.add_argument(
        "--load_ckpt_from_local",
        type=bool,
        required=False,
        help="whether load checkpoints from local default paths (assuming you have downloaded the checkpoints locally",
    )

    parser.set_defaults(func=evaluate)

def stringify_cmd(cmd_ls):
    cmd = ""
    for string in cmd_ls:
        cmd += string + " "
    return cmd

## TODO
def evaluate(args):
    args_dict = vars(args)
    abs_path_main = os.path.abspath(f'{CUR_DIR}/../launch/evaluate.py')
    abs_path_json = os.path.abspath(f'{CUR_DIR}/../MBench_full_info.json')
    cmd = [
        'python3',
        abs_path_main,
        '--output_path', os.path.join(args_dict['output_path']),
        '--full_json_dir', abs_path_json,
        '--evaluation_path', args_dict['evaluation_path'],
        '--dimension', args_dict['dimension']
    ]
    subprocess.run(cmd)
