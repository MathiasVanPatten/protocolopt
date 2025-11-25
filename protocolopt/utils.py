import torch
import sys
import os

def robust_compile(model, compile_mode=True):
    """
    Attempts to compile a model using torch.compile(backend="inductor").
    
    Respects the compile_mode flag and falls back to eager execution 
    if on macOS, if compilation fails, or if compile_mode is False.
    """
    if not compile_mode:
        return model

    is_mac = sys.platform == 'darwin'

    if is_mac:
        print("Mac detected: Skipping torch.compile to avoid compiler dependency issues.")
        return model
    
    try:
        return torch.compile(model, backend="inductor") 
    except Exception as e:
        print(f"WARNING: torch.compile failed (likely missing C compiler). Falling back to eager execution.\nError: {e}")
        return model

