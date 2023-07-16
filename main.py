# Required imports for controlling tello drone
from djitellopy import tello
from matplotlib import image
import tello_module as tm

# Required imports for video streaming, face tracking and keyboard control
import cv2
import time
import uuid  # For generating unique filenames
import os  # For creating directories
import re  # For speech detection

# Required imports for speech detection
from ssl import ALERT_DESCRIPTION_UNKNOWN_PSK_IDENTITY
import pyaudio  # For audio recording
import websockets  # A websocket is used to connect to the AssemblyAI API
import asyncio  # For asynchronous programming, in particular for the websocket
import base64  # For encoding audio data. enconding means converting binary data to ASCII characters
import json  # For parsing the JSON data that is received from the AssemblyAI API

# Required imports for hand gesture control
import mediapipe as mp  # For hand tracking, in particular for the holistic model
from tensorflow.python.keras.models import load_model