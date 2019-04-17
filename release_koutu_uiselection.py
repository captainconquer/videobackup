# -*- coding: utf-8 -*-
import tkinter
import cv2
import PIL.Image, PIL.ImageTk
import time
import os
import threading
import sys
import collections
import datetime
import threading
from queue import Queue
from CountsPerSec import CountsPerSec
import configparser
import logging
#logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')  
rootDir = "pic_sample"
logofile = "logo.jpg"
detecting = "detecting.jpg"
#localimage = "LocalImage"
localimage = 'C:\\FaceRec\\regisiter'
def putIterationsPerSec(frame, iterations_per_sec,threadinfo,pos,scale=2,source = ""):
    """
    Add iterations per second text to lower-left corner of a frame.
    """

    if source == "":
        cv2.putText(frame, "{:.0f} iterations/sec in thread:{}".format(iterations_per_sec,threadinfo),
            (10, pos), cv2.FONT_HERSHEY_TRIPLEX,scale, (255, 255, 255), 3)
    else:
        cv2.putText(frame, "{:.0f} iterations/sec in thread:{} from source:{}".format(iterations_per_sec, threadinfo, source),
                    (10, pos), cv2.FONT_HERSHEY_TRIPLEX, scale, (255, 255, 255), 3)
        #cv2.putText(frame," from source:{}".format(source),
         #           (10, pos+70), cv2.FONT_HERSHEY_TRIPLEX, scale, (255, 255, 255), 3)
    return frame

def queue_img_get(queues,outputdirs,camera_macs,uiswitch='close',DebugMode='False', \
                  queues2displays=[],queueresults=[], videowidth=640,videoheight=480):
    logging.info('img get from queue thread started')
    if DebugMode=='True':
        cps = CountsPerSec().start()
    print
    while True:
        for queue, queuedisp, queueresult,outputdir,camera_mac in zip(queues, queues2displays,queueresults,outputdirs,camera_macs):
            frame = queue.get()
            if frame is not None:
                if DebugMode=='True':
                    logging.info(DebugMode)
                    putIterationsPerSec(frame, cps.countsPerSec(), 'get from queue', 200)
                    cps.increment()
                result = recognition(img=frame, outputdir=outputdir,camera_mac=camera_mac, fx=1, fy=1,blur=11, isOpen=DebugMode)
                if DebugMode=='True':
                    #recoginized frame to display
                    resizedframe = cv2.resize(frame, (videowidth, videoheight), interpolation=cv2.INTER_CUBIC)
                    queuedisp.put(resizedframe)
                    queuedisp.get() if queuedisp.qsize() > 1 else None
                    
                    queueresult.put(result)
                    queueresult.get() if queueresult.qsize() > 1 else None
        
def queue_img_put(q, ip,uiswitch='close',DebugMode='False',queuedisp=[],videowidth=640,videoheight=480):
    #VideoCapture = MyVideoCapture('rtsp://admin:asb#1234@192.168.7.64:554/h264/ch1/main/av_stream')
    logging.info('img put to queue thread started')
    VideoCapture = None
    VideoCapture = MyVideoCapture(ip)
    if DebugMode=='True':
        cps = CountsPerSec().start()
    while True:
        is_opened, frame = VideoCapture.get_frame()
        
        if is_opened:
            if DebugMode=='True':
                cps.increment()
                frame = putIterationsPerSec(frame, cps.countsPerSec(),'from',100,2,ip)
            q.put(frame)
            q.get() if q.qsize() > 1 else None
            
            if uiswitch=='open':
                resizedframe = cv2.resize(frame, (videowidth, videoheight), interpolation=cv2.INTER_CUBIC)
                #original frame to display
                queuedisp.put(resizedframe)
                queuedisp.get() if queuedisp.qsize() > 1 else None
        else:
            time.sleep(3)
            logging.debug('Camera disconnected try to recreate video  = %s',VideoCapture.video_source)
            VideoCapture = MyVideoCapture(ip)
        #time.sleep(0.04)
class App:
    def __init__(self, window, window_title, camera_ips,videowidth,videoheight,DebugMode):
        self.window = window
        self.lablelist = []
        self.window.startposX = 0
        self.window.startposY = 40
        self.window.scdistenceX = 20
        self.window.splitratio = 0.9
        self.window.title(window_title)
        self.window.attributes('-fullscreen', True)
        self.window.attributes("-alpha", 1)
        # self.window.screenwidth = 683
        # self.window.screenheight = 384
        self.window.screenwidth = window.winfo_screenwidth()
        self.window.screenheight = window.winfo_screenheight()

        self.logowidth = int(self.window.screenwidth/10)-8
        self.logohight = int((402/1133)*self.logowidth)

        self.facedistance = 10
        self.facewidth = int((self.window.screenwidth-self.logowidth-self.facedistance*9)/10)
        self.facehight = self.facewidth

        self.Canvasfacewidth = self.window.screenwidth
        self.Canvasfacehight = int(self.facehight*(1+0.4))

        self.Canvaswidth = self.window.screenwidth
        self.Canvashight = self.window.screenheight-self.Canvasfacehight
        self.videodistenceX = 20

        self.CanvasStartX = 0
        self.CanvasStartY = 0

        self.CanvasfaceStartX = self.CanvasStartX
        self.CanvasfaceStartY = self.Canvashight

        self.facestartX = self.CanvasfaceStartX
        self.facestartY = self.CanvasfaceStartY+int((self.Canvasfacehight-self.facehight)*0.25)

        self.logostartX = self.CanvasfaceStartX+(self.Canvasfacewidth-self.logowidth)
        self.logostartY = self.Canvasfacehight-self.logohight


        self.buttonwidth = 5
        self.buttonhight = 5
        self.buttonstartX = self.window.screenwidth-60
        self.buttonstartY = self.facestartY

        self.videowidth = int((self.Canvaswidth-self.videodistenceX)/2)
        self.videoheight = int((videoheight/videowidth)*self.videowidth)
        self.videostartposY = int((self.Canvashight-self.videoheight)/2)

        self.iconcontainer = []
        self.lablecontainer = []

        self.imgcontainer = []
        cv_img = cv2.imread(detecting)
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        # cv_img = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(cv_img))
        for i in range(0, 10):
            self.imgcontainer.append((cv_img, 'name'))

        # Create a canvas that can fit the above video source size
        self.canvas = tkinter.Canvas(window, width = self.Canvaswidth, height = self.Canvashight,bg = 'black')
        self.canvas.place(x = self.CanvasStartX,y=self.CanvasStartY)
        self.canvasface = tkinter.Canvas(window, width=self.Canvasfacewidth,
                                     height=self.Canvasfacehight, bg='DeepSkyBlue')
        self.canvasface.place(x=self.CanvasfaceStartX,y=self.CanvasfaceStartY)
        #display logo
        self.logo = cv2.imread(logofile)
        self.logo = cv2.resize(self.logo, (self.logowidth, self.logohight), interpolation=cv2.INTER_CUBIC)
        self.logo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(self.logo))
        self.canvasface.create_image(self.logostartX, self.logostartY, image=self.logo, anchor=tkinter.NW)
        #display detecting picture
        self.detectingpic = cv2.imread(detecting)
        self.cpses = []
        self.cpses.append(CountsPerSec().start())
        self.cpses.append(CountsPerSec().start())

        
        for i in range(0, 10):
            cv_img = cv2.resize(self.detectingpic, (self.facewidth, self.facewidth), interpolation=cv2.INTER_CUBIC)
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            self.iconcontainer.append(PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(cv_img)))
            lable = tkinter.Label(self.window, text='', image=self.iconcontainer[i], compound=tkinter.TOP,
                          background='LightBlue')
            lable.place(x=(self.facestartX + i * (self.facewidth + self.facedistance)),y=self.facestartY)
            self.lablecontainer.append(lable)
        
        #display video 1 and video 2
        self.videolables = []
        self.videolables.append(tkinter.Label(self.window, text='', image=self.logo, compound=tkinter.TOP,background='LightBlue'))
        self.videolables[0].place(x = 0,y=self.videostartposY)

        # self.videolables.append(tkinter.Label(self.window, text='', image=self.logo, compound=tkinter.TOP, background='LightBlue'))
        # self.videolables[1].place(x=self.videowidth+self.window.scdistenceX, y=self.videostartposY)


        # Button that lets the user exit App
        self.btn_snapshot=tkinter.Button(window, text="退出", command=self.exitApp)
        self.btn_snapshot.place(x=self.buttonstartX,y=self.buttonstartY)

        # Button that selects camera
        self.cameraSelects=[
            self.cameraSelect1,
            self.cameraSelect2,
            self.cameraSelect3,
            self.cameraSelect4,
            self.cameraSelect5,
            self.cameraSelect6,
            self.cameraSelect7,
            self.cameraSelect8
        ]
        self.btn_cameraSelect = []
        for index in range(len(camera_ips)):
            self.btn_cameraSelect.append(tkinter.Button(window, text=camera_ips[index], command=self.cameraSelects[index])\
                                         .place(x=self.videowidth+self.window.scdistenceX, y=self.videostartposY+index*50))


        # After it is called once, the update method will be automatically called every delay milliseconds
        self.reservedtime = time.time()

        self.photos = [0,0]
        self.indexes = [0,1]
        self.pos = 0
        self.DebugMode = DebugMode
        # self.update()
        # self.window.mainloop()

    def exitApp(self):
        # Get a frame from the video source
        self.window.destroy()
        sys.exit()

    def cameraSelect1(self):
        self.pos = 0
    def cameraSelect2(self):
        self.pos = 1
    def cameraSelect3(self):
        self.pos = 2
    def cameraSelect4(self):
        self.pos = 3
    def cameraSelect5(self):
        self.pos = 4
    def cameraSelect6(self):
        self.pos = 5
    def cameraSelect7(self):
        self.pos = 6
    def cameraSelect8(self):
        self.pos = 7

    def update(self):
        for index in range(1):
            PosStart = index+self.pos
            if queues2displays[PosStart].empty() is False:
                    frame = queues2displays[PosStart].get(False)
                    if frame is not None:
                        if self.DebugMode=='True':
                            self.cpses[index].increment()
                            frame = putIterationsPerSec(frame, self.cpses[index].countsPerSec(), 'display', 300,0.8)
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        self.photos[index] = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
                        self.videolables[index].config(image=self.photos[index])
                        # self.video1lable.config(image=self.photos[index])

            if queueresults[index].empty() is False:
                result = queueresults[index].get(False)
                if result is not None:
                    # print("display face",'result len ', len(result))
                    for list_img_name in result:
                        for j in range(0, 10):
                            #print('list_img_name', list_img_name[1], 'self.imgcontainer[j][1]', self.imgcontainer[j][1])
                            if list_img_name[1] == self.imgcontainer[j][1]:
                                # print("replace exist face")
                                self.imgcontainer[j] = list_img_name#(list_name[i], list_img[i])
                                break
                        if j == 9:
                            self.imgcontainer.insert(0,list_img_name)
                            self.imgcontainer.pop()


        # for queues2display, queueresult, index in zip(queues2displays, queueresults, self.indexes):
        #     if queues2display.empty() is False:
        #         frame = queues2display.get(False)
        #         if frame is not None:
        #             self.cpses[index].increment()
        #
        #             frame = putIterationsPerSec(frame, self.cpses[index].countsPerSec(), 'display', 300,0.8)
        #             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #             self.photos[index] = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
        #             self.videolables[index].config(image=self.photos[index])
        #             # self.video1lable.config(image=self.photos[index])
        #
        #     if queueresult.empty() is False:
        #         result = queueresult.get(False)
        #         if result is not None:
        #             # print("display face",'result len ', len(result))
        #             for list_img_name in result:
        #                 for j in range(0, 10):
        #                     #print('list_img_name', list_img_name[1], 'self.imgcontainer[j][1]', self.imgcontainer[j][1])
        #                     if list_img_name[1] == self.imgcontainer[j][1]:
        #                         # print("replace exist face")
        #                         self.imgcontainer[j] = list_img_name#(list_name[i], list_img[i])
        #                         break
        #                 if j == 9:
        #                     self.imgcontainer.insert(0,list_img_name)
        #                     self.imgcontainer.pop()

        #if (time.time()-self.reservedtime) >= 1:
        for i in range(10):
            img_tuple = self.imgcontainer[i]
            cv_img = cv2.resize(img_tuple[0], (self.facewidth, self.facewidth), interpolation=cv2.INTER_CUBIC)
            self.iconcontainer[i]=PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(cv_img))
            self.lablecontainer[i].config(image=self.iconcontainer[i], text=img_tuple[1],font=12)
        #    self.reservedtime = time.time()

        self.window.after(1, self.update)

class MyVideoCapture:
    def __init__(self, video_source=0):
        # Open the video source
        self.vid1 = cv2.VideoCapture(video_source)
        self.width = 1664
        self.height = 768
        self.video_source = video_source
        if not self.vid1.isOpened():
            #raise ValueError("Unable to open video source", video_source)
            logging.warning("Unable to open video source %s", video_source)
            return

        # Get video source width and height

        self.width = int(self.vid1.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.vid1.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logging.info('video created, video_source = %s',video_source)

    def get_frame(self):
        ret = False

        if self.vid1.isOpened():
            ret, frame = self.vid1.read()
            #frame = cv2.imread('5.jpg')
            #print('get frame from video source = %s, ret',self.video_source)

            if ret:

                # Return a boolean success flag and the current frame converted to BGR
                return (ret, frame)
            else:
                return (ret,None)
        else:
            logging.warning("Camera disconnected, video_source = %s",self.video_source)
            return (ret,None)

    # Release the video source when the object is destroyed
    def __del__(self):
        if self.vid1.isOpened():
            self.vid1.release()
            logging.info('video released, video_source = %s',self.video_source)



if __name__ == '__main__':
    import extract_feature
    import face_preprocess
    import numpy as np
    import mtcnn_detector
    import cv2
    import time
    import mxnet as mx
    import argparse
    import sklearn
    import os
    import json
    from PIL import Image, ImageDraw, ImageFont
    
    
    conf = configparser.ConfigParser()
    conf.read('C:\\unre\\config2.txt')
    
    th = conf.get('threshold','t')
    register = conf.get('register', 'Flag')
    m_size = conf.get('minsize', 'm')
    
     

    parser = argparse.ArgumentParser(description='face model test')
    # general
    parser.add_argument('--image-size', default='112,112', help='')
    parser.add_argument('--model', default='C:\\unre\\face\\face\\models\\model,0',
                        help='path to load model.')
    parser.add_argument('--ga-model', default='', help='path to load model.')
    parser.add_argument('--cpu', default=0, type=int, help='cpu id')
    parser.add_argument('--det', default=0, type=int,
                        help='mtcnn option, 1 means using R+O, 0 means detect from begining')
    parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
    parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
    args = parser.parse_args()

    threshold = float(th)
    image_size = (112, 112)
    ctx = mx.gpu(0)
    #model_path = 'D:\study\disk\insightface-master\insightface-master\models\model,0'
   
    model_path = 'C:\\unre\\face\\face\\models\\model,0'
    #model_path = 'C:\\unre\\models\\model,0'
    min_size = int(m_size)
    mt = mtcnn_detector.MtcnnDetector(model_folder='C:\\unre\\mtcnn-model\\',
                                      minsize=min_size,
                                      threshold=[0.6, 0.7, 0.9],
                                      factor=0.5,
                                      num_worker=10,
                                      accurate_landmark=True,
                                      ctx=ctx)


    def get_model(ctx, image_size, model_str, layer):
        _vec = model_str.split(',')
        assert len(_vec) == 2
        prefix = _vec[0]
        epoch = int(_vec[1])
        logging.info('loading %s %s', prefix, epoch)
        sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
        all_layers = sym.get_internals()
        sym = all_layers[layer + '_output']
        model = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
        # model.bind(data_shapes=[('data', (args.batch_size, 3, image_size[0], image_size[1]))], label_shapes=[('softmax_label', (args.batch_size,))])
        model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
        model.set_params(arg_params, aux_params)
        return model


    model = get_model(ctx, image_size, model_path, 'fc1')


    def faceDetect(face_img):
        if face_img is None:
            return None
        return mt.detect_face(face_img)


    def faceAlign(face_img, ret):

        if ret is None:
            return None
        bbox, points = ret
        # if bbox.shape[0]==0:
        #  return None

        bbox = bbox[0:4]
        points = points.reshape((2, 5)).T

        nimg = face_preprocess.preprocess(face_img, bbox, points, image_size='112,112')
        nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
        
        return nimg
        
        aligned = np.transpose(nimg, (2, 0, 1))
        logging.info('aligned shape: %s', aligned.shape)
        return aligned


    def faceFeature(face_aligned):
        # print('shape: ', face_aligned.shape)
        # if len(face_aligned.shape) == 3:
        # input_blob = np.expand_dims(face_aligned, axis=0)
        # elif len(face_aligned.shape) == 4:
        input_blob = face_aligned

        data = mx.nd.array(input_blob)
        db = mx.io.DataBatch(data=(data,))

        model.forward(db, is_train=False)
        embeddings = model.get_outputs()[0].asnumpy()
        # print('shape of embedding: ', embeddings.shape)

        # embeddings = sklearn.preprocessing.normalize(embeddings).flatten()
        embeddings = sklearn.preprocessing.normalize(embeddings)

        return embeddings


    def findNearestClassForImage(face_descriptor, faceLabel):
        temp = face_descriptor - data
        e = np.linalg.norm(temp, axis=1, keepdims=True)
        min_distance = e.min()
        #print('distance: ', min_distance)
        index = np.argmin(e)
        #if min_distance > 1.2:
        #    print('name: ',faceLabel[index])
        if min_distance > threshold:
            return 'other', min_distance
        index = np.argmin(e)
        
        return faceLabel[index], min_distance

    def get_time_stamp():
        ct = int(round(time.time() * 1000))
#        local_time = time.localtime(ct)
#        data_head = time.strftime("%Y%m%d%H%M%S", local_time)
#        data_secs = (ct - int(ct)) * 1000
#        time_stamp = "%s%03d" % (data_head, data_secs)
#        return time_stamp
        return ct
    
    def roi(a, b):  # returns None if rectangles don't intersect
        dx = min(a[2], b[2]) - max(a[0], b[0])
        dy = min(a[3], b[3]) - max(a[1], b[1])
    
        if (dx>=0) and (dy>=0):
            return [max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]),min(a[3], b[3])]     
        else:
            return None
    
    def recognition(img, outputdir,camera_mac, fx = 1, fy = 1, blur=7, isOpen=False):
        isOpen = False
        width = img.shape[1]
        height = img.shape[0]
        
        detect = faceDetect(img)
        if detect is None:
            return None

        rets, points = detect
        # print('len of rets: ', rets.shape)
        aligns = []
        aligns_img = []

        for k in range(len(rets)):
            r = rets[k]
            p = points[k]

            # print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            #    k, r[0], r[1], r[2], r[3]) )

            aligned = faceAlign(img, (r, p))
            aligns_img.append(aligned)
            aligns.append(np.transpose(aligned, (2, 0, 1)))
            if isOpen:
                cv2.rectangle(img, (int(r[0]), int(r[1])), (int(r[2]), int(r[3])), (0, 255, 0), 2)

        faces = np.array(aligns)
        features = faceFeature(faces)
        # print('shape of features: ', features.shape)
        if isOpen:
            temp_img = Image.fromarray(img)
            draw = ImageDraw.Draw(temp_img)  # 图片上打印
            font = ImageFont.truetype("simhei.ttf", 25, encoding="utf-8")  # 参数1：字体文件路径，参数2：字体大小

        preds = []

        # ==============================================================================
        #
        #         flag = False
        #         if time.time() - globaltime> 1:
        #             flag = True
        #             globaltime = time.time()
        #
        # ==============================================================================
        rets_new = rets.astype(np.int16)
        for i in range(features.shape[0]):
            r = rets_new[i]

            class_pre, min_distance = findNearestClassForImage(features[i, :], label)

            # preds.append(class_pre)
            
            if class_pre is not 'other':
                logging.debug('face detected: %s',class_pre)
                
                # cv2.putText(img, class_pre, (int(r[0]), int(r[1])), cv2.FONT_HERSHEY_SIMPLEX, 2,
                # (0, 255, 0), 2, cv2.LINE_AA)
                if isOpen:
                    draw.text((int(rets[i][0]), int(rets[i][1]) - 25), class_pre, (0, 255, 0),
                          font=font)  # 参数1：打印坐标，参数2：文本，参数3：字体颜色，参数4：字体
                #preds.append((aligns_img[i], class_pre))
                dx = r[2] - r[0]
                dy = r[3] - r[1]
                x0 = int( max( (r[0] + r[2])/2 - fx*dx, 0 ) )
                y0 = int( max( (r[1] + r[3])/2 - fy*dy, 0 ) )
                x1 = int( min( (r[0] + r[2])/2 + fx*dx, width)  )
                y1 = int( min( (r[1] + r[3])/2 + fy*dy, height) )
            
                new_roi_location = [x0,y0,x1,y1]
                new_roi = img[y0:y1, x0:x1, :].copy()    
                    
                for iii in range(features.shape[0]):
                    if iii != i:
                        roi_mask = roi(new_roi_location, rets_new[iii])
                        #if roi_mask == None:
                        #    continue
                        if roi_mask is not None:
                            tmp = [roi_mask[0]-x0, roi_mask[1]-y0, roi_mask[2]-x0, roi_mask[3]-y0]
                            print('tmp: ', tmp)
                            new_roi[tmp[1]:tmp[3], tmp[0]:tmp[2], :] = cv2.blur(new_roi[tmp[1]:tmp[3], tmp[0]:tmp[2], :], (blur,blur))
                            #cv2.imwrite(outputdir+ '_Koutu_' +namedic[class_pre] + '_' + str(round(min_distance,2)) + '_'  + '.jpg',new_roi)
            
                preds.append((aligns_img[i], class_pre, new_roi))
                    
                # if flag is True:_pre, new_roi))
                    
                # if flag is True:
                timestring = get_time_stamp()
                if class_pre in namedic.keys():
                    logging.debug('save results')
#                    cv2.imwrite(outputdir+ '_' +namedic[class_pre] + '_' + str(round(min_distance,2)) + '_' + timestring + '.jpg',
#                                cv2.cvtColor(aligns_img[i], cv2.COLOR_BGR2RGB))
#                    cv2.imwrite(outputdir+ '_Koutu_' +namedic[class_pre] + '_' + str(round(min_distance,2)) + '_' + timestring + '.jpg',new_roi)
                    #cv2.imwrite(outputdir+ '_' +class_pre + '_' + str(round(min_distance,2)) + '_' + timestring + '.jpg',
                     #           cv2.cvtColor(aligns_img[i], cv2.COLOR_BGR2RGB))
                    cv2.imwrite(outputdir+class_pre+'-' +str(camera_mac)+'-' + str(round(min_distance,2)) + '-' + str(timestring) + '.jpg',new_roi)
#
#                    cv2.imencode('.jpg',cv2.cvtColor(aligns_img[i], cv2.COLOR_BGR2RGB))[1].tofile(outputdir+ '_' +class_pre + '_' + str(round(min_distance,2)) + '_' + timestring + '.jpg')
#                    cv2.imencode('.jpg',new_roi)[1].tofile(outputdir+ '_Koutu_' +class_pre + '_' + str(round(min_distance,2)) + '_' + timestring + '.jpg')

            # cv2.rectangle(img, (int(r[0]), int(r[1])), (int(r[2]), int(r[3])), (0, 255, 0), 2)

        #img[:] = temp_img
        # print('shape of face: ', faces[0].shape)
        # cv2.imshow('image', img)
        return preds
    
    if register is 'True':
        import dlib_label_1128
        
        

    labelFile = open('label2.txt', 'r')
    label = json.load(labelFile)  # 载入本地人脸库的标签
    labelFile.close()

    data = np.loadtxt('faceData2.txt', dtype=float)
    
   
    globaltime = time.time()
    namedic = collections.OrderedDict()
    mapname = 0
    for filename in os.listdir(localimage):
        if '.jpg' in filename or '.png' in filename:
            if '_' in filename:
                labelName = filename.split('_')[0]
            else:
                labelName = filename.split('.')[0]
            mapname+=1
            logging.debug('current label: %s', labelName)
            namedic[labelName] = 'k'+str(mapname)
            

    outputdirRoot = 'C:\\FaceRec\\rec_result\\'
    ips = conf.options('camera ip')
    username = conf.get('login', 'username')
    password = conf.get('login', 'password')
    camera_ips = []
    camera_macs = []
    outputdirs = []
    Currentdate = time.strftime('%Y%m%d', time.localtime(time.time()))

    outputdir = outputdirRoot+Currentdate+'\\'
    logging.debug('to outputdir = %s',outputdir)
    
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
        logging.debug('created outputdir = %s',outputdir)
    for ip in ips:
        cameraip = conf.get('camera ip', ip)
        outputname = outputdir
#        if not os.path.exists(outputdir):
#            os.makedirs(outputdir)        
        outputdirs.append(outputname)
        source = 'rtsp://{}:{}@{}:554/h264/ch1/main/av_stream'.format(username,password,cameraip)
        camera_ips.append(source)
        cameramac = conf.get('camera mac',ip)
        camera_macs.append(cameramac)

    DebugMode = conf.get('Debug Data', 'DebugMode')
    vid = MyVideoCapture(camera_ips[0])
    videowidth = vid.width
    videoheight = vid.height    
    if vid.vid1.isOpened():
        vid.vid1.release()
    logging.info('video width and height: %s %s',videowidth,videoheight)
    
    uiswitch = conf.get('ui','uiswitch')
    
    queues = [Queue(maxsize=2) for _ in camera_ips]
    processes = []

    if uiswitch == 'open':
        myApp = App(tkinter.Tk(), "视频监控", camera_ips,videowidth,videoheight,DebugMode)
        queues2displays = [Queue(maxsize=2) for _ in camera_ips]
        queueresults = [Queue(maxsize=2) for _ in camera_ips]
        for queue, camera_ip, queuedisp in zip(queues, camera_ips,queues2displays):
            processes.append(threading.Thread(target=queue_img_put, args=(queue,camera_ip,uiswitch,DebugMode,queuedisp,myApp.videowidth, myApp.videoheight)))
        processes.append(threading.Thread(target=queue_img_get, \
                                          args=(queues,outputdirs,camera_macs,\
                                            uiswitch,DebugMode, queues2displays,queueresults, myApp.videowidth, myApp.videoheight)))
    else:
        for queue, camera_ip in zip(queues, camera_ips):
            processes.append(threading.Thread(target=queue_img_put, args=(queue, camera_ip)))
        processes.append(threading.Thread(target=queue_img_get,args=(queues,outputdirs,camera_macs)))

    [setattr(process, "daemon", True) for process in processes]  # process.daemon = True
    [process.start() for process in processes]
    if uiswitch == 'close':
        [process.join() for process in processes]
    if uiswitch == 'open':
        myApp.update()
        myApp.window.mainloop()
    logging.info('system exit')
    # Create a window and pass it to the Application object
