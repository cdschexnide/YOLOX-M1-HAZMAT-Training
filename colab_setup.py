#!/usr/bin/env python3
"""
Google Colab Setup Script for YOLOX Hazmat Detection Training
Handles dependency conflicts and environment setup automatically
"""

import os
import sys
import subprocess
import importlib
import pkg_resources
from typing import List, Tuple

class ColabSetupManager:
    """Manages Colab environment setup with dependency conflict prevention"""
    
    def __init__(self):
        self.conflicts_detected = []
        self.installation_log = []
    
    def log(self, message: str, level: str = "INFO"):
        """Log setup progress"""
        print(f"[{level}] {message}")
        self.installation_log.append(f"{level}: {message}")
    
    def check_gpu(self) -> bool:
        """Check GPU availability"""
        try:
            import torch
            gpu_available = torch.cuda.is_available()
            if gpu_available:
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                self.log(f"GPU detected: {gpu_name} ({gpu_memory:.1f} GB)")
                return True
            else:
                self.log("No GPU detected - will use CPU training", "WARN")
                return False
        except ImportError:
            self.log("PyTorch not available for GPU check", "WARN")
            return False
    
    def check_preinstalled_packages(self) -> dict:
        """Check versions of pre-installed packages"""
        packages_to_check = [
            'torch', 'torchvision', 'numpy', 'opencv-python', 
            'PIL', 'scipy', 'matplotlib'
        ]
        
        versions = {}
        self.log("Checking pre-installed packages...")
        
        for package in packages_to_check:
            try:
                if package == 'PIL':
                    import PIL
                    version = PIL.__version__
                else:
                    version = pkg_resources.get_distribution(package).version
                versions[package] = version
                self.log(f"  {package}: {version}")
            except:
                versions[package] = "Not installed"
                self.log(f"  {package}: Not installed")
        
        return versions
    
    def install_system_packages(self):
        """Install required system packages"""
        self.log("Installing system packages...")
        
        commands = [
            ["apt-get", "update", "-qq"],
            ["apt-get", "install", "-qq", "-y", "libglib2.0-0", "libsm6", 
             "libxext6", "libxrender-dev", "libgl1-mesa-glx"]
        ]
        
        for cmd in commands:
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                self.log(f"Successfully ran: {' '.join(cmd)}")
            except subprocess.CalledProcessError as e:
                self.log(f"Failed to run: {' '.join(cmd)} - {e}", "ERROR")
    
    def lock_numpy_version(self):
        """Lock NumPy to 1.x to prevent conflicts"""
        self.log("Locking NumPy to version 1.x...")
        
        try:
            # Force install NumPy 1.x without dependencies
            subprocess.run([
                sys.executable, "-m", "pip", "install", 
                "--no-deps", "numpy==1.24.3"
            ], check=True, capture_output=True)
            
            # Verify installation
            import numpy as np
            version = np.__version__
            
            if version.startswith('1.'):
                self.log(f"NumPy locked to: {version}")
                return True
            else:
                self.log(f"NumPy lock failed - got version {version}", "ERROR")
                self.conflicts_detected.append(f"NumPy version {version} != 1.x")
                return False
                
        except Exception as e:
            self.log(f"NumPy installation failed: {e}", "ERROR")
            return False
    
    def install_opencv(self):
        """Install OpenCV compatible with NumPy 1.x"""
        self.log("Installing OpenCV...")
        
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install", 
                "opencv-python==4.8.0.76", "numpy<2.0"
            ], check=True, capture_output=True)
            
            # Test import
            import cv2
            import numpy as np
            
            self.log(f"OpenCV installed: {cv2.__version__}")
            self.log(f"NumPy still at: {np.__version__}")
            return True
            
        except Exception as e:
            self.log(f"OpenCV installation failed: {e}", "ERROR")
            return False
    
    def install_ml_packages(self):
        """Install ML packages with conflict prevention"""
        self.log("Installing ML packages...")
        
        packages = [
            "loguru==0.7.0",
            "tqdm==4.65.0", 
            "tabulate==0.9.0",
            "ninja==1.11.1",
            "thop==0.1.1.post2209072238",
            "tensorboard>=2.12.0,<2.15.0"
        ]
        
        for package in packages:
            try:
                subprocess.run([
                    sys.executable, "-m", "pip", "install", 
                    package, "numpy<2.0"
                ], check=True, capture_output=True)
                self.log(f"Installed: {package}")
            except Exception as e:
                self.log(f"Failed to install {package}: {e}", "WARN")
        
        # Install pycocotools separately with no-build-isolation
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install", 
                "pycocotools", "--no-build-isolation"
            ], check=True, capture_output=True)
            self.log("Installed: pycocotools")
        except Exception as e:
            self.log(f"pycocotools installation failed: {e}", "WARN")
    
    def test_critical_imports(self) -> bool:
        """Test that all critical packages can be imported"""
        self.log("Testing critical imports...")
        
        test_imports = [
            ('torch', 'PyTorch'),
            ('torchvision', 'TorchVision'), 
            ('numpy', 'NumPy'),
            ('cv2', 'OpenCV'),
            ('loguru', 'Loguru'),
            ('tqdm', 'TQDM'),
            ('pycocotools', 'PycocoTools')
        ]
        
        success = True
        for module, name in test_imports:
            try:
                importlib.import_module(module)
                self.log(f"✅ {name} import successful")
            except ImportError as e:
                self.log(f"❌ {name} import failed: {e}", "ERROR")
                success = False
        
        return success
    
    def determine_optimal_batch_size(self) -> int:
        """Determine optimal batch size based on available GPU"""
        try:
            import torch
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                
                if gpu_memory >= 15:  # T4 or better
                    batch_size = 32
                    self.log(f"GPU has {gpu_memory:.1f}GB - using batch size {batch_size}")
                elif gpu_memory >= 10:
                    batch_size = 24
                    self.log(f"GPU has {gpu_memory:.1f}GB - using batch size {batch_size}")
                else:
                    batch_size = 16
                    self.log(f"GPU has {gpu_memory:.1f}GB - using batch size {batch_size}")
                    
                return batch_size
            else:
                self.log("No GPU - using CPU batch size 4")
                return 4
        except:
            self.log("Could not determine GPU specs - using conservative batch size 8")
            return 8
    
    def setup_dataset_paths(self):
        """Setup and verify dataset paths"""
        self.log("Setting up dataset paths...")
        
        # Expected dataset location in Google Drive
        dataset_path = "/content/drive/MyDrive/hazmat_dataset/VOCdevkit/VOC2007"
        
        if os.path.exists(dataset_path):
            # Count dataset files
            try:
                annotations_dir = f"{dataset_path}/Annotations"
                images_dir = f"{dataset_path}/JPEGImages"
                
                if os.path.exists(annotations_dir) and os.path.exists(images_dir):
                    annotation_count = len([f for f in os.listdir(annotations_dir) if f.endswith('.xml')])
                    image_count = len([f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg'))])
                    
                    self.log(f"Dataset found: {annotation_count} annotations, {image_count} images")
                    
                    if annotation_count == image_count and annotation_count > 2000:
                        self.log("✅ Dataset structure validated")
                        return True
                    else:
                        self.log("⚠️ Dataset structure issues detected", "WARN")
                        return False
                else:
                    self.log("❌ Dataset subdirectories not found", "ERROR")
                    return False
                    
            except Exception as e:
                self.log(f"Dataset validation error: {e}", "ERROR")
                return False
        else:
            self.log("❌ Dataset not found at expected path", "ERROR")
            self.log("Please upload dataset to: /content/drive/MyDrive/hazmat_dataset/")
            return False
    
    def run_complete_setup(self) -> bool:
        """Run the complete setup process"""
        self.log("🚀 Starting YOLOX Colab Setup...")
        
        # Step 1: Check environment
        self.check_gpu()
        versions = self.check_preinstalled_packages()
        
        # Step 2: Install system packages
        self.install_system_packages()
        
        # Step 3: Lock NumPy version (critical!)
        if not self.lock_numpy_version():
            self.log("❌ NumPy version locking failed - this will cause conflicts!", "ERROR")
            return False
        
        # Step 4: Install packages in order
        if not self.install_opencv():
            self.log("❌ OpenCV installation failed", "ERROR")
            return False
            
        self.install_ml_packages()
        
        # Step 5: Test imports
        if not self.test_critical_imports():
            self.log("❌ Critical import tests failed", "ERROR")
            return False
        
        # Step 6: Setup dataset
        dataset_ok = self.setup_dataset_paths()
        
        # Step 7: Determine batch size
        batch_size = self.determine_optimal_batch_size()
        
        # Summary
        self.log("\n📋 Setup Summary:")
        self.log(f"  GPU Available: {self.check_gpu()}")
        self.log(f"  Dataset Ready: {dataset_ok}")
        self.log(f"  Recommended Batch Size: {batch_size}")
        
        if self.conflicts_detected:
            self.log("\n⚠️ Conflicts Detected:")
            for conflict in self.conflicts_detected:
                self.log(f"  - {conflict}")
        
        success = len(self.conflicts_detected) == 0 and dataset_ok
        
        if success:
            self.log("\n✅ Setup completed successfully!")
            self.log("Ready to start YOLOX training on Colab!")
        else:
            self.log("\n❌ Setup completed with issues - check logs above")
        
        return success

def main():
    """Main setup function for Colab"""
    setup_manager = ColabSetupManager()
    return setup_manager.run_complete_setup()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)