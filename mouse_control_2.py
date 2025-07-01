import cv2
import mediapipe as mp
import numpy as np
import pyautogui

class MouseControl():

    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        mp_hands = mp.solutions.hands
        self.hands = mp_hands.Hands(static_image_mode=False,
                                     max_num_hands=1,
                                     min_detection_confidence=0.5,
                                     min_tracking_confidence=0.5)
        
    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)

            cv2.imshow("MouseKontrol", frame)
            key = cv2.waitKey(30)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    h, w, _ = frame.shape
                    lm = hand_landmarks.landmark[8]  
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    p0 = np.array([[cx, cy]], dtype=np.float32).reshape(-1, 1, 2)
                    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    self.optic_flow(p0, prev_gray, frame)
            if key == 27:
                break
        self.cap.release()
        cv2.destroyAllWindows()

    def optic_flow(self, p0, prev_gray, frame):
        lk_params = dict(winSize  = (15, 15), 
                         maxLevel = 2, 
                         criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        
        mask = np.zeros_like(frame)  

        while True:
            ret, next_frame = self.cap.read()
            if not ret:
                break

            next_frame = cv2.flip(next_frame, 1)
            next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

            p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, next_gray, p0, None, **lk_params)

            if p1 is not None :
                a, b = p0.ravel()
                c, d = p1.ravel()
                mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0,255,0), 2)
                next_frame = cv2.circle(next_frame, (int(c), int(d)), 5, (0,255,0), -1)
                self.mousemove(c, d, next_frame)
                img = cv2.add(next_frame, mask)

                cv2.imshow('MouseKontrol', img)
                k = cv2.waitKey(30)
                prev_gray = next_gray.copy()
                p0 = p1.reshape(-1, 1, 2)

            if k == 27:
                    break      

    def mousemove(self, x, y, frame):
        screen_w, screen_h = pyautogui.size()
        frame_h, frame_w = frame.shape[:2]
        move_x = int(x * (screen_w / frame_w))
        move_y = int(y * (screen_h / frame_h))
        pyautogui.moveTo(move_x, move_y)

if __name__ == "__main__":
    mc = MouseControl()
    mc.run()
