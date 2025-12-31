"""
VLM Utilities for MBench evaluation.

Provides video preprocessing and Gemini API calling utilities.
"""

import os
import time
from pathlib import Path
from typing import Optional

try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Warning: google-genai not available. Install it with: pip install google-genai")


def call_gemini_api(
    video_bytes: bytes,
    prompt: str,
    api_key: str,
    fps: int = 5,
    retry_times: int = 3,
    client=None
) -> str:
    """
    Call Gemini API with video and text prompt.
    
    Args:
        video_bytes: Video file content as bytes
        prompt: Text prompt for the model
        api_key: Gemini API key
        fps: FPS for video sampling (default: 5)
        retry_times: Number of retry attempts
        client: Optional pre-initialized Gemini client
        
    Returns:
        Model response text
        
    Raises:
        RuntimeError: If all retry attempts fail
    """
    if not GEMINI_AVAILABLE:
        raise ImportError("google-genai package is required but not installed")
    
    if not api_key:
        raise ValueError("API key is required")
        
    client = client or genai.Client(api_key=api_key)
    text = prompt.strip()

    for i in range(retry_times):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=types.Content(
                    role="user",
                    parts=[
                        types.Part(text=text),
                        types.Part(
                            inline_data=types.Blob(data=video_bytes, mime_type="video/mp4"),
                            video_metadata=types.VideoMetadata(fps=fps)
                        )
                    ],
                ),
            )
            if response.text:
                return response.text
        except Exception as e:
            print(f"Attempt {i+1} failed: {e}")
            if i < retry_times - 1:
                time.sleep(2 ** i)  # Exponential backoff
            
    raise RuntimeError(f"Gemini API failed after {retry_times} attempts")