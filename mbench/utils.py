import os
import json
import re
from typing import List, Dict, Any, Union
from pathlib import Path


def get_prompt_from_filename(filename: str) -> str:
    """
    Extract prompt from video filename by removing extensions and special characters
    
    Args:
        filename: Video filename
        
    Returns:
        Extracted prompt string
    """
    # Remove file extension
    name = Path(filename).stem
    
    # Remove common suffixes like "-0", "-1", etc.
    name = re.sub(r'-\d+$', '', name)
    
    # Replace underscores and hyphens with spaces
    name = re.sub(r'[_-]', ' ', name)
    
    # Clean up multiple spaces
    name = re.sub(r'\s+', ' ', name).strip()
    
    return name.lower()


def save_json(data: Union[Dict, List], filepath: str) -> None:
    """
    Save data to JSON file
    
    Args:
        data: Data to save
        filepath: Path to save the JSON file
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(filepath: str) -> Union[Dict, List]:
    """
    Load data from JSON file
    
    Args:
        filepath: Path to the JSON file
        
    Returns:
        Loaded data
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)




def validate_video_path(video_path: str) -> bool:
    """
    Validate that a video file exists and has a supported extension
    
    Args:
        video_path: Path to video file
        
    Returns:
        True if valid, False otherwise
    """
    if not os.path.exists(video_path):
        return False
    
    valid_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv'}
    extension = Path(video_path).suffix.lower()
    
    return extension in valid_extensions


def create_video_info_dict(video_path: str, prompt: str, dimensions: List[str]) -> Dict[str, Any]:
    """
    Create a video information dictionary
    
    Args:
        video_path: Path to the video file
        prompt: Text prompt associated with the video
        dimensions: List of evaluation dimensions
        
    Returns:
        Dictionary with video information
    """
    return {
        "prompt_en": prompt,
        "dimension": dimensions,
        "video_list": [video_path]
    }


def format_results_summary(results_dict: Dict[str, Any]) -> str:
    """
    Format evaluation results into a readable summary string
    
    Args:
        results_dict: Dictionary containing evaluation results
        
    Returns:
        Formatted summary string
    """
    summary_lines = ["MBench Evaluation Results Summary", "=" * 40]
    
    for dimension, result in results_dict.items():
        summary_lines.append(f"\n{dimension}:")
        
        if isinstance(result, dict):
            if 'error' in result:
                summary_lines.append(f"  Error: {result['error']}")
            else:
                for key, value in result.items():
                    if isinstance(value, (int, float)):
                        if key.lower() in ['accuracy', 'score', 'metric']:
                            summary_lines.append(f"  {key}: {value:.4f}")
                        else:
                            summary_lines.append(f"  {key}: {value}")
                    elif not isinstance(value, (list, dict)):
                        summary_lines.append(f"  {key}: {value}")
        else:
            summary_lines.append(f"  Result: {result}")
    
    return "\n".join(summary_lines)


def ensure_directory_exists(path: str) -> None:
    """
    Ensure that a directory exists, creating it if necessary
    
    Args:
        path: Directory path to create
    """
    os.makedirs(path, exist_ok=True) 

def load_dimension_info(json_dir, dimension):
    """
    Load video list and prompt information based on a specified dimension from a JSON file.
    
    Parameters:
    - json_dir (str): The directory path where the JSON file is located.
    - dimension (str): The dimension for evaluation
    
    Returns:
    - prompt_dict_ls (list): A list of dictionaries, each containing a prompt and its corresponding evaluation file.
    
    The function reads the JSON file to extract video information. It filters the prompts based on the specified
    dimension.
    
    Notes:
    - The JSON file is expected to contain a list of dictionaries with keys 'dimension', 'evaluation_file', and prompts.
    """
    prompt_dict_ls = []
    full_prompt_list = load_json(json_dir)
    for prompt_dict in full_prompt_list:
        if dimension == prompt_dict['dimension'] and 'evaluation_file' in prompt_dict:
            prompt = prompt_dict['prompt']
            cur_evaluation_file = prompt_dict['evaluation_file']
            entry = {
                'prompt': prompt,
                'evaluation_file': cur_evaluation_file,
                'id': prompt_dict.get('id'),
                'dimension': prompt_dict.get('dimension'),
                'category': prompt_dict.get('category'),
                'motion_duration': prompt_dict.get('motion_duration'),
            }
            if 'auxiliary_info' in prompt_dict:
                entry['auxiliary_info'] = prompt_dict['auxiliary_info']
            prompt_dict_ls.append(entry)
    return prompt_dict_ls
