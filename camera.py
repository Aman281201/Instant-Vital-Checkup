import cv2
import mediapipe as mp
import time
from matplotlib import pyplot as plt
import numpy as np
from statistics import mode

# mpDraw = mp.solutions.drawing_utils
# mpPose = mp.solutions.pose
# pose = mpPose.Pose()  
class VideoCamera(object):
    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        self.video = cv2.VideoCapture(0)
        # self.video.set(4,1920)
        # self.video.set(3,1080)
        self.video.set(5,30)
        self.beats = [0]*120
        self.secs = [time.time()]*120
        self.fig  = plt.figure()
        self.graph = self.fig.add_subplot(111)
        self.l = []
        self.ret = True

        #self.video.set(5,30)
        # If you decide to use video.mp4, you must have this file in the folder
        # as the main.py.
        # self.video = cv2.VideoCapture('video.mp4')
    
    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        # mpDraw = mp.solutions.drawing_utils
        # mpPose = mp.solutions.pose
        # pose = mpPose.Pose()  
        # print("gg")
       
        success, img = self.video.read()
        mpDraw = mp.solutions.drawing_utils
        mpPose = mp.solutions.pose
        pose = mpPose.Pose()

        # cap = cv2.VideoCapture(0)

        # while True:
        minheight =1000
        maxheight=0
        # success, img = cap.read()
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = img[0:, 200:1000]
        # img = cv2.resize(img, (1500,1500))

        results = pose.process(imgRGB)
        if results.pose_landmarks:
                mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
                for id, lm in enumerate(results.pose_landmarks.landmark):
                    h, w, c = img.shape
                    print(str(h)+ " " + str(w) + " " + str(c))
                    if(id==32 or id==30 or id==29 or id==31):
                        minheight=min(minheight,lm.y)
                    if(id==2 or id==5 ):
                        maxheight=max(maxheight,lm.y)
                    

                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

        cv2.putText(img, str(  6.5+ ( (-maxheight+minheight)*283.25)     ), (70, 450), cv2.FONT_HERSHEY_PLAIN, 2,
                        (255, 0, 0), 2)
            # cv2.imshow("Image", img)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

                
        ret, jpeg = cv2.imencode('.jpg', img)
        return jpeg.tobytes()
    
    def get_pulse_frame(self):
        self.ret, img = self.video.read()

        # img = cv2.resize(img, (1500,1500))

       
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_img = img[300:400, 660:960]
    
        self.beats = self.beats[1:] + [np.average(face_img)]
        self.secs = self.secs[1:] + [time.time()]
        self.graph.plot(self.secs,self.beats)
    
        #GRAPH PLOTTING  (https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.html)
        #self.fig.canvas.draw()
        # beat_monitor = np.fromstring(self.fig.canvas.tostring_rgb(),dtype=np.uint8, sep='')
        # print(beat_monitor.shape)
        # beat_monitor = beat_monitor.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
       # plt.cla()
        #GRAPH PLOTTING
        print(self.beats[-1])
        cv2.putText(face_img, str(  self.beats[-1]), (30, 30), cv2.FONT_HERSHEY_PLAIN, 2,
                        (255, 0, 0), 2)
        # cv.imshow("hearbeat",beat_monitor)
        self.l.append(np.average(self.beats))
        # cv.imshow('skin_frame', face_img)
        # if cv.waitKey(1) & 0xFF == ord('q'):
        #     break

        ret, jpeg = cv2.imencode('.jpg', face_img)
        return jpeg.tobytes()

    def get_gen_age_frame(self,padding,count,a,g,faceNet):
        # success, img = self.video.read()

        
        hasFrame,frame=self.video.read()
        # if not hasFrame:
        #     cv2.waitKey()
        #     continue
        
        resultImg,faceBoxes=highlightFace(faceNet,frame)
        # if not faceBoxes:
        #     print("No face detected")

        for faceBox in faceBoxes:
            face=frame[max(0,faceBox[1]-padding):
                    min(faceBox[3]+padding,frame.shape[0]-1),max(0,faceBox[0]-padding)
                    :min(faceBox[2]+padding, frame.shape[1]-1)]
            
            blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
            genderNet.setInput(blob)
            genderPreds=genderNet.forward()
            gender=genderList[genderPreds[0].argmax()]
            print(f'Gender: {gender}')
            g.append(gender)
            
            ageNet.setInput(blob)
            agePreds=ageNet.forward()
            age=ageList[agePreds[0].argmax()]
            print(f'Age: {age[1:-1]} years')

            a.append(f'Age: {age[1:-1]} years')

            count+=1
            cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
        
        ret, jpeg = cv2.imencode('.jpg', resultImg)
        return count,a,g,jpeg.tobytes()

def highlightFace(net, frame, conf_threshold=0.7):
        frameOpencvDnn=frame.copy()
        frameHeight=frameOpencvDnn.shape[0]
        frameWidth=frameOpencvDnn.shape[1]
        blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

        net.setInput(blob)
        detections=net.forward()
        # [,frame,no of detections,[classid,class score,conf,x,y,h,w]
        faceBoxes=[]
        for i in range(detections.shape[2]):
            confidence=detections[0,0,i,2]
            if confidence>conf_threshold:
                x1=int(detections[0,0,i,3]*frameWidth)
                y1=int(detections[0,0,i,4]*frameHeight)
                x2=int(detections[0,0,i,5]*frameWidth)
                y2=int(detections[0,0,i,6]*frameHeight)
                faceBoxes.append([x1,y1,x2,y2])
                cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (0,255,0), int(round(frameHeight/150)), 8)
        return frameOpencvDnn,faceBoxes

def gen0(camera):

    abhinn(camera)
    # faceProto="face/opencv_face_detector.pbtxt"
    # faceModel="face/opencv_face_detector_uint8.pb"
    # ageProto="age/age_deploy.prototxt"
    # ageModel="age/age_net.caffemodel"
    # genderProto="gender/gender_deploy.prototxt"
    # genderModel="gender/gender_net.caffemodel"

    # MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
    # ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    # genderList=['Male','Female']

    # faceNet=cv2.dnn.readNet(faceModel,faceProto)
    # ageNet=cv2.dnn.readNet(ageModel,ageProto)
    # genderNet=cv2.dnn.readNet(genderModel,genderProto)

    # padding=20
    # count = 0
    # a = []
    # g = []


    # while count<50:
        
    #     count,a,g,frame = camera.get_gen_age_frame(padding,count,a,g,faceNet)
    #     yield (b'--frame\r\n'
    #         b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    # print("final after mode")
    # print(mode(a))
    # print(mode(g))
       
        
def abhinn(camera):
    def highlightFace(net, frame, conf_threshold=0.7):
        frameOpencvDnn=frame.copy()
        frameHeight=frameOpencvDnn.shape[0]
        frameWidth=frameOpencvDnn.shape[1]
        blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

        net.setInput(blob)
        detections=net.forward()
        # [,frame,no of detections,[classid,class score,conf,x,y,h,w]
        faceBoxes=[]
        for i in range(detections.shape[2]):
            confidence=detections[0,0,i,2]
            if confidence>conf_threshold:
                x1=int(detections[0,0,i,3]*frameWidth)
                y1=int(detections[0,0,i,4]*frameHeight)
                x2=int(detections[0,0,i,5]*frameWidth)
                y2=int(detections[0,0,i,6]*frameHeight)
                faceBoxes.append([x1,y1,x2,y2])
                cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (0,255,0), int(round(frameHeight/150)), 8)
        return frameOpencvDnn,faceBoxes


    faceProto="face/opencv_face_detector.pbtxt"
    faceModel="face/opencv_face_detector_uint8.pb"
    ageProto="age/age_deploy.prototxt"
    ageModel="age/age_net.caffemodel"
    genderProto="gender/gender_deploy.prototxt"
    genderModel="gender/gender_net.caffemodel"

    MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
    ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    genderList=['Male','Female']

    faceNet=cv2.dnn.readNet(faceModel,faceProto)
    ageNet=cv2.dnn.readNet(ageModel,ageProto)
    genderNet=cv2.dnn.readNet(genderModel,genderProto)

    padding=20
    count = 0
    a = []
    g = []
    while True:
        if count >=50:
            break
        hasFrame,frame=camera.video.read()
        if not hasFrame:
            cv2.waitKey()
            continue
        
        resultImg,faceBoxes=highlightFace(faceNet,frame)
        if not faceBoxes:
            print("No face detected")

        for faceBox in faceBoxes:
            face=frame[max(0,faceBox[1]-padding):
                    min(faceBox[3]+padding,frame.shape[0]-1),max(0,faceBox[0]-padding)
                    :min(faceBox[2]+padding, frame.shape[1]-1)]
            
            blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
            genderNet.setInput(blob)
            genderPreds=genderNet.forward()
            gender=genderList[genderPreds[0].argmax()]
            print(f'Gender: {gender}')
            g.append(gender)
            
            ageNet.setInput(blob)
            agePreds=ageNet.forward()
            age=ageList[agePreds[0].argmax()]
            print(f'Age: {age[1:-1]} years')

            a.append(f'Age: {age[1:-1]} years')

            count+=1
            cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
        cv2.imshow("Detecting age and gender", resultImg)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
    # do a bit of cleanup
    print("final after mode")
    print(mode(a))
    print(mode(g))
    cv2.destroyAllWindows()
    video.release()
    #out.release()




def gen2(camera):
    while True:

        frame = camera.get_pulse_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def gen(camera):
    while True:
        
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

