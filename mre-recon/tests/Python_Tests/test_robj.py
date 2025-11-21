import torch
import sys
import os

test_file_path = os.path.abspath(__file__) # gets the full, absolute path to test_robj.py

test_dir = os.path.dirname(test_file_path) # gets the folder test_robj.py its in

project_root = os.path.dirname(os.path.dirname(test_dir)) # gets the root folder path

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.regularizers.robj import Robj

def Robj_tests():
    """Run test cases (static method equivalent)."""
    print("\n=== Python Robj Test Cases ===\n")
    
    # Test 1: Quadratic
    print("Test 1: Quadratic Regularization")
    mask = torch.ones((32, 32), dtype=torch.bool)
    R = Robj(mask, beta=10.0, potential='quad')
    
    torch.manual_seed(42)
    num_pixels = mask.sum().item()
    x = torch.randn(num_pixels, 1, dtype=torch.float32)
    
    penalty = R.penal(x)
    gradient = R.cgrad(x)
    
    print(f"  Penalty:     {penalty.item():.10e}")
    print(f"  Grad norm:   {torch.norm(gradient).item():.10e}")
    print(f"  Grad[0]:     {gradient[0].item():.10e}")
    print(f"  Grad[99]:    {gradient[99].item():.10e}")
    print(f"  Grad[499]:   {gradient[499].item():.10e}\n")
    
    # Test 2: Total Variation
    print("Test 2: Total Variation")
    R_tv = Robj(mask, beta=10.0, delta=0.01, potential='approxTV')
    
    penalty_tv = R_tv.penal(x)
    gradient_tv = R_tv.cgrad(x)
    
    print(f"  Penalty:     {penalty_tv.item():.10e}")
    print(f"  Grad norm:   {torch.norm(gradient_tv).item():.10e}")
    print(f"  Grad[0]:     {gradient_tv[0].item():.10e}")
    print(f"  Grad[99]:    {gradient_tv[99].item():.10e}")
    print(f"  Grad[499]:   {gradient_tv[499].item():.10e}\n")

    # Test 3: Denom
    print("Test 3: Denominator")
    ddir = torch.randn(num_pixels, 1, dtype=torch.float32)
    denom_quad = R.denom(ddir, x)
    denom_tv = R_tv.denom(ddir, x)
    
    print(f"  Denom (Quad): {denom_quad.item():.10e}")
    print(f"  Denom (TV):   {denom_tv.item():.10e}\n")

    print("=== Tests Complete ===")


if __name__ == "__main__":
    Robj_tests()