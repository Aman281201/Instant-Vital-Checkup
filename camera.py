import cv2
import mediapipe as mp
import time

class VideoCamera(object):
    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        self.video = cv2.VideoCapture(0)
        # If you decide to use video.mp4, you must have this file in the folder
        # as the main.py.
        # self.video = cv2.VideoCapture('video.mp4')
    
    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        print("gg")
        mpDraw = mp.solutions.drawing_utils
        mpPose = mp.solutions.pose
        pose = mpPose.Pose()    
        success, img = self.video.read()
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        minheight =1000
        maxheight=0
        
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(imgRGB)
        if results.pose_landmarks:
            mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
            for id, lm in enumerate(results.pose_landmarks.landmark):
                h, w, c = img.shape
                if(id==32 or id==30 or id==29 or id==31):
                    minheight=min(minheight,lm.y)
                if(id==2 or id==5 ):
                    maxheight=max(maxheight,lm.y)
        #         print(id, lm)

                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

        cv2.putText(img, str(  6.5+ ( (-maxheight+minheight)*283.25)     ), (70, 450), cv2.FONT_HERSHEY_PLAIN, 2,
                (255, 0, 0), 2)
        # cv2.imshow("Image", img)


        
        ret, jpeg = cv2.imencode('.jpg', img)
        return jpeg.tobytes()



def gen(camera):
    while True:

        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
