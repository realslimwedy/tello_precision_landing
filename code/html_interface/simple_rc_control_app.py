import subprocess
from flask import Flask, render_template
from djitellopy import Tello
import threading

app = Flask(__name__)
me = None

def connect_to_wifi(network_name):
    network_interface = "en0"  # Replace with the appropriate network interface for your Mac

    try:
        subprocess.run(["networksetup", "-setairportnetwork", network_interface, network_name], check=True)
        print(f"Connected to Wi-Fi network: {network_name}")
    except subprocess.CalledProcessError:
        print(f"Failed to connect to Wi-Fi network: {network_name}")


@app.route('/')
def hello_world():
    return render_template('index.html', title="Tello Drone Control")

@app.route('/connect-tello-wifi', methods=['POST'])
def connect_tello_wifi():
    print('connecting to TELLO WIFI ...')
    connect_to_wifi("TELLO-9C7357")
    return 'OK'

@app.route('/connect-home-wifi', methods=['POST'])
def connect_home_wifi():
    print('connecting to Home WIFI ...')
    connect_to_wifi("Leev_Marie")
    return 'OK'

@app.route('/connect-drone', methods=['POST'])
def connect_drone():
    global me
    print('connecting to drone ...')
    if me is not None:
        print('Drone already connected')
        return 'OK'
    try:
        me = Tello()
        print('Drone connected successfully')
        return 'OK'
    except Exception as e:
        print(f'Error initializing drone: {str(e)}')
        return 'Failed'

@app.route('/takeoff', methods=['POST'])
def takeoff():
    global me
    print('takeoff ...')
    if me is None:
        print('Drone not connected')
        return 'Failed'
    try:
        me.takeoff()
        print('Drone took off')
        return 'OK'
    except Exception as e:
        print(f'Error during takeoff: {str(e)}')
        return 'Failed'

@app.route('/land', methods=['POST'])
def land():
    global me
    print('land ...')
    if me is None:
        print('Drone not connected')
        return 'Failed'
    try:
        me.land()
        print('Drone landed')
        return 'OK'
    except Exception as e:
        print(f'Error during landing: {str(e)}')
        return 'Failed'

def run_drone():
    global me
    try:
        me = Tello()
        print('Drone object initialized')
        me.connect()
        print('Drone connected')
    except Exception as e:
        print(f'Error initializing drone: {str(e)}')

if __name__ == '__main__':
    # Start the Tello drone in a separate thread
    drone_thread = threading.Thread(target=run_drone)
    drone_thread.start()
    print('Drone thread started')

    app.run(debug=True, port=5001)
    print('App run started')
