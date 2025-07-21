#!/usr/bin/env python3

import sys
import traceback
import os

# Add current directory to path
sys.path.append('.')

def debug_training():
    try:
        print("Starting debug training...")
        
        # Import after adding to path
        from exps.hazmat.yolox_s_hazmat_simple import Exp
        
        print("Creating experiment...")
        exp = Exp()
        
        print("Testing data loader creation...")
        loader = exp.get_data_loader(batch_size=2, is_distributed=False)
        
        print(f"Dataset length: {len(exp.dataset)}")
        print("Data loader created successfully!")
        
        print("Testing batch iteration...")
        for i, batch in enumerate(loader):
            print(f"Successfully loaded batch {i}")
            if i >= 1:  # Test first 2 batches
                break
                
        print("Data loading test completed successfully!")
        
    except Exception as e:
        print(f"ERROR CAUGHT: {e}")
        print("Full traceback:")
        traceback.print_exc()
        return False
        
    return True

if __name__ == "__main__":
    success = debug_training()
    if success:
        print("All tests passed! Training should work.")
    else:
        print("Issues found - need to fix before training.")