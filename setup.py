#!/usr/bin/env python3
"""
Setup script for Enhanced YOLO Object Detection
This script will help you set up the environment and download required files.
"""

import os
import sys
import subprocess
import urllib.request
from pathlib import Path

def print_status(message, status="INFO"):
    """Print colored status messages"""
    colors = {
        "INFO": "\033[94m",      # Blue
        "SUCCESS": "\033[92m",   # Green
        "WARNING": "\033[93m",   # Yellow
        "ERROR": "\033[91m",     # Red
        "RESET": "\033[0m"       # Reset
    }
    
    color = colors.get(status, colors["INFO"])
    reset = colors["RESET"]
    print(f"{color}[{status}] {message}{reset}")

def check_python_version():
    """Check if Python version is compatible"""
    print_status("Checking Python version...")
    
    if sys.version_info < (3, 7):
        print_status("Python 3.7 or higher is required!", "ERROR")
        return False
    
    version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print_status(f"Python version {version} - OK", "SUCCESS")
    return True

def install_requirements():
    """Install required Python packages"""
    print_status("Installing required packages...")
    
    packages = [
        "opencv-python>=4.5.0",
        "numpy>=1.19.0",
        "flask>=2.0.0",
        "werkzeug>=2.0.0",
        "scipy>=1.7.0"
    ]
    
    for package in packages:
        try:
            print_status(f"Installing {package}...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", package
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print_status(f"✓ {package} installed", "SUCCESS")
        except subprocess.CalledProcessError:
            print_status(f"✗ Failed to install {package}", "ERROR")
            return False
    
    return True

def create_directories():
    """Create necessary directories"""
    print_status("Creating directories...")
    
    directories = [
        "templates",
        "uploads", 
        "processed",
        "static"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print_status(f"✓ Directory '{directory}' created", "SUCCESS")
    
    return True

def download_yolo_files():
    """Download YOLO model files"""
    print_status("Downloading YOLO model files...")
    
    files_to_download = {
        "yolov4-tiny.weights": "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights",
        "yolov4-tiny.cfg": "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg",
        "coco.names": "https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names"
    }
    
    for filename, url in files_to_download.items():
        if os.path.exists(filename):
            print_status(f"✓ {filename} already exists", "SUCCESS")
            continue
            
        try:
            print_status(f"Downloading {filename}...")
            urllib.request.urlretrieve(url, filename)
            
            # Check file size
            size = os.path.getsize(filename)
            size_mb = size / (1024 * 1024)
            print_status(f"✓ {filename} downloaded ({size_mb:.1f} MB)", "SUCCESS")
            
        except Exception as e:
            print_status(f"✗ Failed to download {filename}: {e}", "ERROR")
            return False
    
    return True

def create_html_template():
    """Create the HTML template if it doesn't exist"""
    print_status("Checking HTML template...")
    
    template_path = "templates/index.html"
    
    if os.path.exists(template_path):
        print_status("✓ HTML template already exists", "SUCCESS")
        return True
    
    print_status("HTML template not found. Please create templates/index.html", "WARNING")
    print_status("You can use the provided HTML template from the artifacts", "INFO")
    
    return True

def create_launch_script():
    """Create a launch script"""
    print_status("Creating launch script...")
    
    launch_script_content = '''#!/usr/bin/env python3
"""
Launch script for Enhanced YOLO Object Detection
"""

import sys
import os

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and run the main application
try:
    from app import app, detector
    
    print("🚀 Starting Enhanced YOLO Detection Server...")
    print("📊 Features: Object Detection, Tracking, Analytics, Downloads")
    print("🌐 Access: http://localhost:5000")
    print("📚 API Documentation: http://localhost:5000/api/docs")
    print("="*60)
    
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)
    
except KeyboardInterrupt:
    print("\\n🛑 Server stopped by user")
    if 'detector' in globals():
        detector.stop_camera()
except Exception as e:
    print(f"❌ Error starting server: {e}")
    if 'detector' in globals():
        detector.stop_camera()
'''
    
    with open("launch.py", "w") as f:
        f.write(launch_script_content)
    
    # Make executable on Unix systems
    if os.name != 'nt':
        os.chmod("launch.py", 0o755)
    
    print_status("✓ Launch script created (launch.py)", "SUCCESS")
    return True

def create_requirements_txt():
    """Create requirements.txt file"""
    print_status("Creating requirements.txt...")
    
    requirements_content = """# Enhanced YOLO Object Detection Requirements
opencv-python>=4.5.0
numpy>=1.19.0
flask>=2.0.0
werkzeug>=2.0.0
scipy>=1.7.0
pillow>=8.0.0
psutil>=5.8.0
"""
    
    with open("requirements.txt", "w") as f:
        f.write(requirements_content)
    
    print_status("✓ requirements.txt created", "SUCCESS")
    return True

def run_tests():
    """Run basic functionality tests"""
    print_status("Running basic tests...")
    
    try:
        # Test imports
        import cv2
        import numpy as np
        import flask
        
        print_status(f"✓ OpenCV version: {cv2.__version__}", "SUCCESS")
        print_status(f"✓ NumPy version: {np.__version__}", "SUCCESS")
        print_status(f"✓ Flask version: {flask.__version__}", "SUCCESS")
        
        # Test camera access
        try:
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                print_status("✓ Camera access test passed", "SUCCESS")
                cap.release()
            else:
                print_status("⚠ Camera not accessible (this is OK if no camera)", "WARNING")
        except:
            print_status("⚠ Camera test failed (this is OK if no camera)", "WARNING")
        
        return True
        
    except ImportError as e:
        print_status(f"✗ Import test failed: {e}", "ERROR")
        return False

def main():
    """Main setup function"""
    print_status("🚀 Enhanced YOLO Object Detection Setup", "INFO")
    print_status("="*50, "INFO")
    
    # Step 1: Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Step 2: Install requirements
    if not install_requirements():
        print_status("Failed to install requirements. Please install manually:", "ERROR")
        print_status("pip install opencv-python numpy flask werkzeug scipy", "ERROR")
        sys.exit(1)
    
    # Step 3: Create directories
    if not create_directories():
        sys.exit(1)
    
    # Step 4: Download YOLO files
    if not download_yolo_files():
        print_status("Failed to download YOLO files. Please download manually.", "ERROR")
        sys.exit(1)
    
    # Step 5: Create additional files
    create_html_template()
    create_launch_script()
    create_requirements_txt()
    
    # Step 6: Run tests
    if not run_tests():
        print_status("Some tests failed. The application might still work.", "WARNING")
    
    # Final message
    print_status("="*50, "SUCCESS")
    print_status("🎉 Setup completed successfully!", "SUCCESS")
    print_status("", "INFO")
    print_status("Next steps:", "INFO")
    print_status("1. Make sure templates/index.html exists (use the provided template)", "INFO")
    print_status("2. Run: python app.py", "INFO")
    print_status("3. Or run: python launch.py", "INFO")
    print_status("4. Open browser: http://localhost:5000", "INFO")
    print_status("", "INFO")
    print_status("Features available:", "INFO")
    print_status("- Real-time object detection", "INFO")
    print_status("- Video/Image processing", "INFO")
    print_status("- Results download", "INFO")
    print_status("- Analytics export", "INFO")
    print_status("- Heatmap generation", "INFO")

if __name__ == "__main__":
    main()