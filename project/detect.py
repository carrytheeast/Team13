# detec.py

import argparse
# from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, scale_coords, set_logging, non_max_suppression, strip_optimizer, check_imshow
from utils.torch_utils import select_device, TracedModel

from gaze_estimation import GazeEstimation
from object_selection import ObjectSelection
from utils2 import gridLine, gazePointLine, gazeText, warningText
from plot import printPlot

def detect(args):    

    #################################################
    # yolov7 by WongKinYiu with modifications START #
    #################################################

    source, weights, view_img, imgsz, trace, save_path, mode = args.source, args.weights, args.view_img, args.img_size, not args.no_trace, args.save_path, args.mode
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    fps_cnt_list, total_fps_cnt_list, fps_obj_list = [], [], []
    first_cnt = 0
  
    # Initialize
    set_logging()
    device = select_device(args.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA
#     half = False # If you couldn't see the bbox in yolov7, Set the over line to annotation and Work this line 
    
    # Load model
    model = attempt_load(weights, map_location=device)# load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, args.img_size)

    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    ###############################################
    # yolov7 by WongKinYiu with modifications END #
    ###############################################

    ###################################################
    # our code START : Gaze Estimation with face mesh #
    ###################################################

    # Get names and colors
    # our model have 12 classes
    names = ['book','pen(pencil)','laptop','tabletcomputer','keyboard','cellphone','mouse','pencilcase','wallet','desklamp','airpods','stopwatch']
    colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in names]

    # Gaze estimation model with mediapipe
    gaze_model = GazeEstimation(args)
    object_select_model = ObjectSelection(args)
    
    for path, img, im0s, vid_cap in dataset:        
        first_cnt+=1
        if webcam:
            im0s = im0s[0]
            im0f = im0s.copy()
        else:
            im0f = im0s
        if first_cnt==1:
            if vid_cap:
                w = round(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = round(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = vid_cap.get(cv2.CAP_PROP_FPS)
            else:
                fps, w, h = 30, im0s.shape[1], im0s.shape[0]    # webcam frame
            out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

        with gaze_model.mp_face_mesh.FaceMesh( max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
            
            # Draw the image.
            im0f.flags.writeable = False
            im0f = cv2.cvtColor(im0f, cv2.COLOR_BGR2RGB)

            # Load the face mesh model.
            results = face_mesh.process(im0f)
            x = im0f.shape[1] # height
            y = im0f.shape[0] # width
            object_select_model.font_size = int((x+y/2)*0.03) # draw.text font size
            
            # Draw the face mesh annotations on the image.
            im0f.flags.writeable = True
            im0f = cv2.cvtColor(im0f, cv2.COLOR_RGB2BGR)   

            # Gaze estimation with face mesh   
            if results.multi_face_landmarks:
                for face_landmarks in (results.multi_face_landmarks):

                    # Drawing base line(facemesh)
                    if mode:
                        gaze_model.drawingBaseLine(im0s, face_landmarks)

                    # Make DataFrames
                    gaze_model.makeDataFrames(face_landmarks, x, y) # iris_df, eyes_df 저장                

                    # Gaze Point Estimation
                    range_w = int(x * .07) # 좌측부터 2,3번째 그리드의 x좌표 간격에 각각 +,- 값
                    eye_list, gaze_line_x, dir_total, box1_x1, box1_x2 = gaze_model.gazeDirection(range_w, x, y)                

                    # EAR ratio
                    ear, gaze_line_y, box1_y1, box1_y2 = gaze_model.earRatio(face_landmarks, eye_list, y)                

    #################################################
    # our code END : Gaze Estimation with face mesh #
    #################################################

    #################################################
    # yolov7 by WongKinYiu with modifications START #
    #################################################

                img = torch.from_numpy(img).to(device)
                img = img.half() if half else img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)

                # Warmup
                if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
                    old_img_b = img.shape[0]
                    old_img_h = img.shape[2]
                    old_img_w = img.shape[3]
                    for i in range(3):
                        model(img, augment=args.augment)[0]

                # Inference
                pred = model(img, augment=args.augment)[0]

                # Apply NMS
                pred = non_max_suppression(pred, args.conf_thres, args.iou_thres, classes=args.classes, agnostic=args.agnostic_nms)

    ###############################################
    # yolov7 by WongKinYiu with modifications END #
    ###############################################

    #############################################################
    # our code START : Object Selection and Display information #
    #############################################################

                if mode:
                    # Grid line
                    gridLine(gaze_model.eyes_df, im0s, range_w, eye_list[0], eye_list[2], x, y)            
                    # Gaze text
                    gazeText(im0s, dir_total, ear, x, y)

                # Process detections
                for i, det in enumerate(pred):  # detections per image
                    
                    if len(det): # 감지되는 물건이 있을 때                                
                        if webcam:  # batch_size >= 1
                            p, s, im0s, frame = path[i], '%g: ' % i, im0s, dataset.count
                        else:
                            p, s, im0s, frame = path, '',im0s, getattr(dataset, 'frame', 0)

                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()

                        # 1 frame 내의 bbox 모두
                        box1 = [box1_x1, box1_y1, box1_x2, box1_y2]
                        im0s, cell_phone_xyxy = object_select_model.IOUandTime(im0s, ear, names, colors, det, fps, box1, x, y)

                        # gaze point line
                        if ear != 'UP': # up은 시선 표시 X
                            if mode:
                                gazePointLine(im0s, face_landmarks, gaze_line_x, gaze_line_y, x, y, cell_phone_xyxy, object_select_model.top_iou_obj)                
                        else:
                            im0s = warningText(im0s, x, y)

                    else: # 감지되는 물건이 없을 때
                        im0s = object_select_model.noObject(im0s, x, y)

                    object_select_model.total_fps_cnt += 1/fps

                    fps_cnt_list.append(object_select_model.fps_cnt)
                    if object_select_model.top_iou_obj == 'close':
                        object_select_model.top_iou_obj = 'sleep'
                    fps_obj_list.append(object_select_model.top_iou_obj)
                    total_fps_cnt_list.append(object_select_model.total_fps_cnt) 
                    if webcam:
                        cv2.imshow('A EYE',im0s)
                    out.write(im0s)

                    # Print statement
                    print('statement: dir: {}, ear: {}, obj: {}'.format(dir_total, ear, object_select_model.top_iou_obj))
                        
            else: # facemesh 안될 때
                im0s = object_select_model.noFaceMesh(im0s, x, y)

                object_select_model.total_fps_cnt += 1/fps
                fps_cnt_list.append(object_select_model.fps_cnt)
                fps_obj_list.append('None')
                total_fps_cnt_list.append(object_select_model.total_fps_cnt) 
                if webcam:
                    cv2.imshow('A EYE',im0s)
                out.write(im0s)

    print('save as', save_path)
    out.release()
    return fps_cnt_list, total_fps_cnt_list, fps_obj_list
    
    ###########################################################
    # our code END : Object Selection and Display information #
    ###########################################################   

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--weights', nargs='+', type=str, default='best.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='study02.mp4', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')           
    parser.add_argument('--save-path', type=str, default='run.mp4', help='file/dir/URL/glob')
    parser.add_argument('--mode', type=int, default=0, help='play mode') # 0 for user, 1 for developer
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()    
    with torch.no_grad():
        if args.update:  # update all models (to fix SourceChangeWarning)
            for args.weights in ['best.pt']:
                detect(args)
                strip_optimizer(args.weights)
        else:
            fps_cnt_list, total_fps_cnt_list, fps_obj_list = detect(args) 
            printPlot(fps_cnt_list, total_fps_cnt_list, fps_obj_list)
