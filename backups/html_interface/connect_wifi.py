import subprocess

def connect_to_wifi(network_name):
    network_interface = "en0"  # Replace with the appropriate network interface for your Mac

    try:
        subprocess.run(["networksetup", "-setairportnetwork", network_interface, network_name], check=True)
        print(f"Connected to Wi-Fi network: {network_name}")
    except subprocess.CalledProcessError:
        print(f"Failed to connect to Wi-Fi network: {network_name}")