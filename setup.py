"""
Setup script for Q-Pilot system.
Installs dependencies and initializes the environment.
"""
import os
import sys
import subprocess
import platform


def check_python_version():
    """
    Check if Python version is compatible
    """
    major, minor = sys.version_info[:2]
    if major < 3 or (major == 3 and minor < 7):
        print("⚠️  Warning: Python 3.7+ is recommended for Q-Pilot")
        return False
    return True


def install_dependencies():
    """
    Install required dependencies
    """
    print("📦 Installing dependencies...")

    # Read requirements
    requirements_file = "requirements.txt"
    if not os.path.exists(requirements_file):
        print(f"❌ Requirements file {requirements_file} not found")
        return False

    try:
        # Install requirements
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", requirements_file
        ])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        return False


def download_sample_data():
    """
    Download sample NGSIM data (placeholder)
    """
    print("📂 Setting up sample data...")

    # Create data directory
    os.makedirs("data", exist_ok=True)

    # For demo purposes, we'll just create a README
    readme_content = """
    NGSIM Dataset
    =============

    Place the NGSIM dataset files in this directory.

    Expected format:
    - Trajectory data with columns: Vehicle_ID, Frame_ID, Local_X, Local_Y, v_Vel, v_Acc, Lane_ID
    - File format: CSV

    For more information, visit: https://www.fhwa.dot.gov/publications/research/operations/07030/
    """

    with open("data/README.md", "w") as f:
        f.write(readme_content)

    print("✅ Sample data directory created")


def setup_directories():
    """
    Setup required directories
    """
    print("📁 Setting up directories...")

    directories = [
        "data",
        "models",
        "training",
        "evaluation",
        "results",
        "notebooks",
        "configs",
        "dashboard",
        "research_module"
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)

    print("✅ Directories setup completed")


def create_environment_file():
    """
    Create environment configuration file
    """
    print("⚙️  Creating environment configuration...")

    env_content = """
    # Q-Pilot Environment Configuration

    # Data paths
    DATA_PATH=data/
    MODEL_PATH=models/
    RESULTS_PATH=results/

    # Training parameters
    BATCH_SIZE=32
    EPOCHS=50
    LEARNING_RATE=0.001

    # Quantum parameters
    NUM_QUBITS=4

    # Dashboard settings
    DASHBOARD_PORT=8501
    """

    with open(".env", "w") as f:
        f.write(env_content.strip())

    print("✅ Environment configuration created")


def main():
    """
    Main setup function
    """
    print("🔧 Q-Pilot Setup Script")
    print("=" * 30)

    # Check Python version
    if not check_python_version():
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Setup cancelled")
            return

    # Setup directories
    setup_directories()

    # Install dependencies
    print("\nInstalling dependencies...")
    if not install_dependencies():
        print("⚠️  Dependency installation failed. Please install manually using:")
        print("   pip install -r requirements.txt")

    # Download sample data
    download_sample_data()

    # Create environment file
    create_environment_file()

    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Prepare your dataset (place NGSIM data in 'data/' directory)")
    print("2. Run training: python main.py --mode train")
    print("3. Launch dashboard: python main.py --mode dashboard")
    print("4. Or run full system: python main.py")


if __name__ == "__main__":
    main()