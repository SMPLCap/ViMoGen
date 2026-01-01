import torch
import os
import argparse
from datetime import datetime

from mbench import MBench


def parse_args():
    """Parse command line arguments"""
    CUR_DIR = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(
        description='MBench: Motion generation benchmark', 
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default='./evaluation_results/',
        help="Output path to save the evaluation results",
    )
    parser.add_argument(
        "--full_info_json",
        type=str,
        default=f'{CUR_DIR}/data/meta_info/MBench_eval_info.json',
        help="Path to the JSON file that contains the prompt and dimension information",
    )
    parser.add_argument(
        "--evaluation_path",
        type=str,
        required=True,
        help="Folder that contains the model generated results",
    )
    parser.add_argument(
        "--dimension",
        nargs='+',
        default=None,
        help="List of evaluation dimensions, usage: --dimension <dim_1> <dim_2>",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for evaluation (cuda or cpu)",
    )
    # Dimension-specific arguments
    parser.add_argument(
        "--gemini_api_key",
        type=str,
        default=None,
        help="Gemini API key for Action_Accuracy evaluation (can also use GEMINI_API_KEY env var)",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    
    print(f'MBench Evaluation Arguments: {args}')

    # Initialize MBench
    device = torch.device(args.device)
    mbench = MBench(
        device=str(device), 
        output_path=args.output_path,
        full_info_dir=args.full_info_json
    )
    
    # Prepare kwargs for evaluation
    kwargs = {}
    if args.gemini_api_key:
        kwargs['api_key'] = args.gemini_api_key
    
    # Generate evaluation name with timestamp
    current_time = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    eval_name = f'mbench_results_{current_time}'
    
    print(f'Starting MBench evaluation...')
    print(f'Evaluation path: {args.evaluation_path}')
    print(f'Dimensions: {args.dimension}')

    # Run evaluation
    try:
        results = mbench.evaluate(
            evaluation_path=args.evaluation_path,
            name=eval_name,
            dimension_list=args.dimension,
            **kwargs
        )
        
        # Print summary
        print('\nEvaluation completed successfully!')
        print(f'Results saved to: {args.output_path}')
        
        # Print brief results summary
        print('\nResults Summary:')
        print('-' * 40)
        for dimension, result in results.items():
            if isinstance(result, dict) and 'error' not in result:
                if 'aggregate' in result:
                    agg = result['aggregate']
                    print(f'{dimension}: {agg.get("mean", "N/A"):.4f} (std: {agg.get("std", 0):.4f})')
                elif 'accuracy' in result:
                    print(f'{dimension}: {result["accuracy"]:.4f}')
                elif 'score' in result:
                    print(f'{dimension}: {result["score"]:.4f}')
                else:
                    print(f'{dimension}: Completed')
            elif isinstance(result, dict) and 'error' in result:
                print(f'{dimension}: Error - {result["error"]}')
            else:
                print(f'{dimension}: {result}')
                
    except Exception as e:
        print(f'Evaluation failed with error: {e}')
        raise


if __name__ == "__main__":
    main()
