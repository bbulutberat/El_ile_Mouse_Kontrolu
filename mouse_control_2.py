import cv2
import numpy as np
import mediapipe as mp
import pyautogui

class MouseControl():
    
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        mp_hands = mp.solutions.hands
        self.hands = mp_hands.Hands(static_image_mode=False,
                                     max_num_hands=1,
                                     min_detection_confidence=0.5,
                                     min_tracking_confidence=0.5)
        self.lk_params = dict(winSize  = (15, 15), 
                         maxLevel = 2, 
                         criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.mp_control = False
        self.p0 = None

    def main(self):
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)

            if self.mp_control == False:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(rgb)
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        h, w, _ = frame.shape
                        lm = hand_landmarks.landmark[8]  
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        self.p0 = np.array([[cx, cy]], dtype=np.float32).reshape(-1, 1, 2)
                        self.frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        self.mask = np.zeros_like(frame)
                        self.mp_control = True
            
            if self.p0 is not None:
                img = self.optic_flow()
            
            else:
                img = frame

            cv2.imshow("MouseControl", img)
            key = cv2.waitKey(1)
            if key == 27:
                break
            
    def optic_flow(self):
        ret, frame2 = self.cap.read()
        frame2 = cv2.flip(frame2, 1)
        frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        p1, st, err = cv2.calcOpticalFlowPyrLK(self.frame_gray, frame2_gray, self.p0, None, **self.lk_params)

        if p1 is not None:
            a, b = self.p0.ravel()
            c, d = p1.ravel()
            self.mask = cv2.line(self.mask, (int(a), int(b)), (int(c), int(d)), (0,255,0), 2)
            self.frame2 = cv2.circle(frame2, (int(c), int(d)), 5, (0,255,0), -1)
            img = cv2.add(frame2, self.mask)
            self.mousemove(c, d, frame2)
            self.frame_gray = frame2_gray.copy()
            self.p0 = p1.reshape(-1, 1, 2)
            return img
    
    def mousemove(self, x, y, frame2):
        screen_w, screen_h = pyautogui.size()
        frame_h, frame_w = frame2.shape[:2]
        move_x = int(x * (screen_w / frame_w))
        move_y = int(y * (screen_h / frame_h))
        pyautogui.moveTo(move_x, move_y)


if __name__ == "__main__":
    mc = MouseControl()
    mc.main()
