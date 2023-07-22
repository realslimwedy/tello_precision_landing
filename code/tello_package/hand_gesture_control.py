import cv2
import mediapipe as mp
import numpy as np

def mediapipe_detection(image, model):
    """
    Calculates landmarks using one of mediapipe's model.
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_landmarks(image, results):
    """
    Draws hand landmarks determined by the mediapipe holistic model.
    """
    # Create short cuts
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic

    # Draw landmarks
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

def get_bbox_coords(results):
    """
    Gets bbox coordinates (min x,y and max x,y) based on mediapipe model 
    (using the extracted keypoints/landmarks)
    """
    try:
        x_min = min(res.x for res in results.right_hand_landmarks.landmark)
        y_min = min(res.y for res in results.right_hand_landmarks.landmark)
        x_max = max(res.x for res in results.right_hand_landmarks.landmark)
        y_max = max(res.y for res in results.right_hand_landmarks.landmark)
        return x_min, y_min, x_max, y_max
    except: 
        return 0, 0, 0, 0

def draw_bbox(image, results, res, class_id):
    """
    Draws boundary boxes based on mediapipe output. 
    Extracts min x,y and max x,y values from landmarks for this.
    """
    actions = ['fist', 'palm', 'index', 'ok', 'thp_up']
    coords = get_bbox_coords(results)
    
    # Draw bbox
    cv2.rectangle(image,
                  tuple(np.multiply(coords[:2], list(res)).astype(int)),
                  tuple(np.multiply(coords[2:], list(res)).astype(int)),
                  (0,0,255), 2)

    # Draw label box
    cv2.rectangle(image,
                    tuple(np.add(np.multiply(coords[:2], list(res)).astype(int), [0, -30])),
                    tuple(np.add(np.multiply(coords[:2], list(res)).astype(int), [80, 0])),
                    (0,0,255), -1)

    # Put text in label
    cv2.putText(image, f'{actions[class_id]}', 
                tuple(np.add(np.multiply(coords[:2], list(res)).astype(int), [0, -5])),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

def extract_keypoints(results):
    """
    Extracts key points from results vector delivered by holistic 
    mediapipe model.
    """
    rhand = np.zeros(21*3)
    if results.right_hand_landmarks:
        rhand = np.array([[result.x, result.y, result.z] 
                          for result in results.right_hand_landmarks.landmark]).flatten()
        
    return rhand

def predict_hand_gesture(image, mp_model, hand_model, res):
    """
    Wrapper function - combines the above functions to make a prediction 
    and return class_id, class_prob as well as center and area of bbox.
    Also comprises drawing functions (optional, can be deactivated).
    """ 
    # Use mediapipe model to predict landmarks for hands
    image, results = mediapipe_detection(image, mp_model)
    
    # Preprocess landmarks/keypoints for model from right hand
    keypoints = extract_keypoints(results)
    keypoints = keypoints.reshape(1,-1)

    # Make a prediction with the loaded model
    y_pred = hand_model.predict(keypoints)
    class_id = np.argmax(y_pred)
    class_prob = np.max(y_pred)

    # Draw hand_landmarks and bbox optional
    draw_landmarks(image, results)
    draw_bbox(image, results, res, class_id)
    
    # Calculate bbox params -> center_point and area for tracking
    coords = get_bbox_coords(results)
    coords = tuple(np.multiply(coords, list(2 * res)).astype(int))
    
    center = [(coords[0] + coords[2]) // 2, (coords[1] + coords[3]) // 2]
    area = (coords[2] - coords[0]) * (coords[3] - coords[1]) // 1
    
    return image, [class_id, class_prob], [center, area]

def convert_hand_signal_to_rc_params(rc_params, hand_pred, hand_data, pid, error, prv_error):
    
    # hand_actions = ['fist', 'palm', 'index', 'ok', 'thumb_up']
    # hand_id =          0       1       2       3       4
    
    # Unpack input params
    lr, fb, ud, yv = rc_params
    hand_id, hand_prob = hand_pred
    [cx, cy], area = hand_data
    
    lr_err, fb_err, ud_err, yv_err = error
    lr_err_prv, fb_err_prv, ud_err_prv, yv_err_prv = prv_error
    prp, itg, dif = pid

    # Define speed for movements
    speed = 20
    
    if cx != 0: # Only change rc params if a hand is in the frame
        if hand_prob > 0.8:
            # Move drone via hand gestures (change rc-params)
            if hand_id == 0: # 'fist'
                print(f'fist, p={hand_prob}')
                fb = speed

            if hand_id == 1: # 'palm'
                print(f'palm, p={hand_prob}')
                fb = -speed
            
            if hand_id == 2: # 'index'
                print(f'index, p={hand_prob}')
                # Only control lr, ud, rest is 0
                fb, yv = 0, 0
                
                # Adjust lr and ud params using pid-control
                # Left/right
                lr = prp * lr_err + dif * (lr_err - lr_err_prv)
                lr = int(np.clip(lr, -10, 10))

                # Up/down
                ud = prp * ud_err + dif * (ud_err - ud_err_prv)
                ud = int(np.clip(ud, -25, 25))

            if hand_id == 3: # 'ok'
                print(f'ok, p={hand_prob}')
                # Only control yv, rest is 0
                fb, ud, lr = 0, 0, 0
                
                # Adjust lr and ud params using pid-control
                # Turn left/right
                yv = prp * yv_err + dif * (yv_err - yv_err_prv)
                yv = int(np.clip(yv, -25, 25))

            if hand_id == 4: # 'thumb_up'
                print(f'thumb up, p={hand_prob}')
                # Stop all
                lr, fb, ud, yv = 0, 0, 0, 0

    return (lr, fb, ud, yv)
    
    