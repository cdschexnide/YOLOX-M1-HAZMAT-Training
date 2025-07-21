#!/usr/bin/env python3

import sys
import os
sys.path.append('.')

from exps.hazmat.yolox_s_hazmat_m1 import Exp

def test_dataloader():
    print("Creating experiment...")
    exp = Exp()
    
    print("Testing data loader...")
    try:
        print("Getting data loader...")
        loader = exp.get_data_loader(batch_size=1, is_distributed=False)
        print(f"Dataset length: {len(exp.dataset)}")
        print("Data loader created successfully!")
        
        print("Testing batch loading...")
        for i, batch in enumerate(loader):
            print(f"Loaded batch {i}")
            if i >= 2:  # Test first 3 batches
                break
        print("Data loading test passed!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_dataloader()