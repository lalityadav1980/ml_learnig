import os
import platform
import subprocess
import sys
import json  # Use json module for parsing

# Define environment name
env_name = "lstm_env"

# Detect OS and architecture
system_os = platform.system()
machine = platform.machine()
is_mac_m1_m2 = system_os == "Darwin" and machine in ["arm64", "aarch64"]


# Get the latest Python version available in Conda
def get_latest_python_version():
    try:
        output = subprocess.check_output(["conda", "search", "python", "--json"])
        available_versions = sorted(
            [pkg["version"] for pkg in json.loads(output.decode())["python"] if pkg["version"].startswith("3.")],
            key=lambda s: list(map(int, s.split('.'))),
            reverse=True
        )
        return available_versions[0] if available_versions else "3.11"  # Default to 3.11 if unknown
    except Exception as e:
        print(f"Error fetching latest Python version: {e}")
        return "3.11"  # Fallback to a reasonable default


python_version = get_latest_python_version()


# Function to check if Conda environment exists
def conda_env_exists(env_name):
    try:
        result = subprocess.run(["conda", "env", "list", "--json"], capture_output=True, text=True, check=True)
        envs = json.loads(result.stdout).get("envs", [])
        return any(os.path.basename(env) == env_name for env in envs)
    except Exception as e:
        print(f"Error checking Conda environments: {e}")
        return False


# Function to get the current active Conda environment
def get_current_conda_env():
    return os.environ.get("CONDA_DEFAULT_ENV", "")


# Function to remove an existing Conda environment
def remove_conda_env(env_name):
    current_env = get_current_conda_env()
    if current_env == env_name:
        print(f"Cannot remove the currently active environment '{env_name}'. Please deactivate it first.")
        sys.exit(1)  # Exit the script gracefully
    if conda_env_exists(env_name):
        print(f"Environment '{env_name}' exists. Removing...")
        try:
            subprocess.run(["conda", "env", "remove", "-n", env_name, "-y"], check=True)
            print(f"Environment '{env_name}' removed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Failed to remove environment '{env_name}': {e}")
            sys.exit(1)
    else:
        print(f"Environment '{env_name}' does not exist. No need to remove.")


# Function to create a Conda environment
def create_conda_env():
    print(f"Creating Conda environment '{env_name}' with Python {python_version}...")
    try:
        subprocess.run(["conda", "create", "-n", env_name, f"python={python_version}", "-y"], check=True)
        print(f"Environment '{env_name}' created successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to create environment '{env_name}': {e}")
        sys.exit(1)


# Function to install packages
def install_packages():
    print(f"Installing necessary libraries in '{env_name}'...")

    base_packages = [
        "numpy", "pandas", "matplotlib", "scikit-learn",
        "keras"
    ]

    try:
        # Install base packages using Conda
        subprocess.run(["conda", "install", "-n", env_name, "-y"] + base_packages, check=True)

        # Ensure pip is installed
        subprocess.run(["conda", "install", "-n", env_name, "-y", "pip"], check=True)

        # Prepare pip packages
        pip_packages = [
            "tensorflow.keras",
            "logging"
        ]

        if is_mac_m1_m2:
            # Use pip for macOS-specific TensorFlow versions
            pip_packages.extend(["tensorflow-macos", "tensorflow-metal"])
        else:
            pip_packages.append("tensorflow")  # Regular TensorFlow for other OS

        # Install pip packages within the environment
        subprocess.run(["conda", "run", "-n", env_name, "pip", "install"] + pip_packages, check=True)

        print(f"All required packages installed in '{env_name}'.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install packages: {e}")
        sys.exit(1)


# Main execution
if __name__ == "__main__":
    current_env = get_current_conda_env()
    print(f"Current Conda environment: '{current_env}'")

    remove_conda_env(env_name)  # Remove existing env if it exists and not active
    create_conda_env()  # Create new env with latest Python version
    install_packages()  # Install required packages

    print(f"✅ Conda virtual environment '{env_name}' is ready! 🚀")
    print(f"To activate it, run: conda activate {env_name}")
