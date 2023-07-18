import cv2 as cv
from apriltag_module import apriltag_center_area


cap = cv.VideoCapture(0)  # Use the default webcam (change the index if needed)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Call the main function from apriltag_detection.py
    img, [center_x, center_y], area = apriltag_center_area(frame)
        

    
    # Display the frame
    cv.imshow('Webcam', img)
    cv.waitKey(1)
    
    key = cv.waitKey(1)
    if key == 27:  # Press Esc to exit
        break

    # Print the tag ID, center coordinates, and area
    print("Center X:", center_x)
    print("Center Y:", center_y)
    print("Area:", area)

    
cap.release()
cv.destroyAllWindows()
