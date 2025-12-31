import torch
import os
from mbench import mbench
from mbench.distributed import dist_init, print0
from datetime import datetime
import argparse
import json

def parse_args():

    CUR_DIR = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description='MBench', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--output_path",
        type=str,
        default='evaluation_results/',
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
        nargs='+',
        required=True,
        help="list of evaluation dimensions, usage: --dimension <dim_1> <dim_2>",
    )
    parser.add_argument(
        "--load_ckpt_from_local",
        type=bool,
        required=False,
        help="whether load checkpoints from local default paths (assuming you have downloaded the checkpoints locally",
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    print(f'args: {args}')
    device = torch.device("cuda")
    my_MBench = MBench(device, args.full_json_dir, args.output_path)
    
    print(f'start evaluation')

    current_time = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

    kwargs = {}

    prompt = []

    if (args.prompt_file is not None) and (args.prompt != "None"):
        raise Exception("--prompt_file and --prompt cannot be used together")
    if (args.prompt_file is not None or args.prompt != "None") and (not args.mode=='custom_input'):
        raise Exception("must set --mode=custom_input for using external prompt")

    if args.prompt_file:
        with open(args.prompt_file, 'r') as f:
            prompt = json.load(f)
        assert type(prompt) == dict, "Invalid prompt file format. The correct format is {\"video_path\": prompt, ... }"
    elif args.prompt != "None":
        prompt = [args.prompt]

    kwargs['imaging_quality_preprocessing_mode'] = args.imaging_quality_preprocessing_mode

    my_VBench.evaluate(
        videos_path = args.videos_path,
        name = f'results_{current_time}',
        prompt_list=prompt, # pass in [] to read prompt from filename
        dimension_list = args.dimension,
        local=args.load_ckpt_from_local,
        read_frame=args.read_frame,
        mode=args.mode,
        **kwargs
    )
    print('done')


if __name__ == "__main__":
    main()
