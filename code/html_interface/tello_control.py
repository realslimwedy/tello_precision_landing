from djitellopy import Tello

# Create a Tello instance
tello = Tello()

# Connect to the Tello drone
tello.connect()

# Main control loop
while True:
    try:
        # Receive command from user input
        command = input("Enter a command (takeoff, land, forward, back, left, right, up, down, flip): ")

        # Process the command
        if command == "takeoff":
            tello.takeoff()
        elif command == "land":
            tello.land()
        elif command == "forward":
            tello.move_forward(30)
        elif command == "back":
            tello.move_back(30)
        elif command == "left":
            tello.move_left(30)
        elif command == "right":
            tello.move_right(30)
        elif command == "up":
            tello.move_up(30)
        elif command == "down":
            tello.move_down(30)
        elif command == "flip":
            tello.flip_right()
        else:
            print("Invalid command. Try again.")

    except KeyboardInterrupt:
        break

# Disconnect from the Tello drone
tello.disconnect()
