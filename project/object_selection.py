# object_selection.py 
import cv2
import numpy as np
from collections import Counter
from PIL import ImageFont, ImageDraw, Image

from utils.plots import plot_one_box
from utils2 import IoU

class ObjectSelection:
    def __init__(self, args):

        self.mode = args.mode

        self.top_iou = None # 1 frame에 보고 있는 물체
        self.top_iou_for10fps = [] # 최근 10 frame동안 보고 있는 물체들
        self.top_iou_obj = None # 최근 10 frame동안 가장 많이 본 물체

        self.fps_cnt = 0.0 # 순공시간 = 1/fps
        self.total_fps_cnt = 0.0 # 전체시간

        self.font_size = 0

    def displayTime(self, draw, x, y):
        if self.fps_cnt: # 현재 시간
            org=(int(x*0.7),int(y*0.3))
            draw.text(org, "순공부시간\n{}:{}:{:.3f}".format(int(self.fps_cnt//60),int(self.fps_cnt//1),self.fps_cnt%1), 
                font=ImageFont.truetype("./Nanum_hand_seongsil.ttf", self.font_size), fill=(255,255,255))              
        if self.total_fps_cnt: # 전체시간
            org=(int(x*0.7),int(y*0.1))
            draw.text(org, "전체시간\n{}:{}:{:.3f}".format(int(self.total_fps_cnt//60),int(self.total_fps_cnt//1),self.total_fps_cnt%1), 
                font=ImageFont.truetype("./Nanum_hand_seongsil.ttf", self.font_size), fill=(255,255,255))  

    def IOUandTime(self, image, ear,  names, colors, det, fps, box1, x, y):
        
        study_obj = ['tabletcomputer','book','laptop','pen(pencil)']
        no_study_obj = ['cellphone']
        iou_key = []
        iou_val = []
        book_head = y*3/4 
        iou_xyxy = []
        cell_phone_xyxy = 0

        # Box Drawing --------------------------------------------------------------------------
        for *xyxy, conf, cls in reversed(det):            
            # 1 frame 내의 bbox별 xyxy 박스 그리기
            xyxy_ = []
            for _ in xyxy:
                xyxy_.append(_.item())

            # 사용물품이 화면 중앙에서 사용 될 때, 탐지 X 보완(ex. cell phone)
            if names[int(cls.item())] =='book':
                book_head = xyxy_[1] # 공부X 물품 기준선

            if (names[int(cls.item())] in no_study_obj) and (xyxy_[1] < book_head):
                cell_phone_xyxy = xyxy # 휴대폰의 위치

            label = f'{names[int(cls)]} {conf:.2f}'
            
            # 객체별 bbox 그리기
            if self.mode:
                plot_one_box(xyxy, image, label=label, color=colors[int(cls)], line_thickness=1)

            # iou
            if box1:
                box2 = xyxy_
                iou = IoU(box1,box2)
                iou_key.append(names[int(cls.item())])
                iou_val.append(iou)
                iou_xyxy.append(xyxy_)

        # Object Selection and Display --------------------------------------------------------------------------
        if iou_key or iou_val:
            self.top_iou = iou_key[np.argmax(iou_val)] # 1 frame의 가장 높은 값 명사로 저장됨(그리드 기준)
            if cell_phone_xyxy: # 예외 정보(사용물품 화면중앙에서 사용 될 때의 보완점)
                self.top_iou = 'cellphone'            
            self.top_iou_for10fps.append(self.top_iou)
        
        image = Image.fromarray(image) # 한글사용 가능하게 변경(draw.text형식과 같이 움직여야함, cv2line 그릴 때는 array화 시켜야함)
        draw = ImageDraw.Draw(image)
        
        if self.mode:
            if self.top_iou: # 현재 보는 거
                org=(int(x*0.1),int(y*0.1))
                draw.text(org, "지금 보는 물체:\n"+self.top_iou, 
                    font=ImageFont.truetype("./Nanum_hand_seongsil.ttf", self.font_size), fill=(255,255,255))
            # writing => top_iou_obj(10fps동안 빈도수 1등)
            if self.top_iou_obj: 
                org=(int(x*0.1),int(y*0.3))
                draw.text(org, "일정시간동안 보는 물체:\n"+self.top_iou_obj, 
                    font=ImageFont.truetype("./Nanum_hand_seongsil.ttf", self.font_size), fill=(255,255,255))

        # 10개의 프레임 중에 가장 높은 사물
        if self.top_iou_for10fps:
            counter_top_iou = Counter(self.top_iou_for10fps)
            self.top_iou_obj = list(counter_top_iou.keys())[(np.argmax(list(counter_top_iou.values())))] # 명사로 저장됨

        # 최근 10개의 프레임
        if len(self.top_iou_for10fps) == 10: 
            self.top_iou_for10fps = self.top_iou_for10fps[1:]                      
        
        if self.top_iou_obj in study_obj:
            self.fps_cnt += 1/fps # 순공시간, 1단위: 1초 
            if ear == 'CLOSE': # 특이사항: 졸음 시간
                org = (int(x*0.35),int(y*0.45))
                draw.text(org, "혹시 졸고 계신가요?", 
                    font=ImageFont.truetype("./Nanum_hand_seongsil.ttf", self.font_size), fill=(255,255,255))
                self.fps_cnt -= 1/fps

        # Time Display --------------------------------------------------------------------------
        self.displayTime(draw, x, y)              
        
        image = np.array(image) # 한글사용 불가능하게 변경
        return image, cell_phone_xyxy           

    # Detection된 Obejct가 없을 때
    def noObject(self, image, x, y):        
        image = Image.fromarray(image) # 한글사용 가능하게 변경
        draw = ImageDraw.Draw(image)        
        self.displayTime(draw, x, y)       

        org=(int(x*0.45),int(y*0.2))
        draw.text(org, "오브젝트가 없습니다.", 
            font=ImageFont.truetype("./Nanum_hand_seongsil.ttf", self.font_size), fill=(255,255,255))
        
        image = np.array(image) # 한글사용 불가능하게 변경
        return image

    # Detection된 Face Mesh가 없을 때
    def noFaceMesh(self, image, x, y):
        image = Image.fromarray(image) # 한글사용 가능하게 변경
        draw = ImageDraw.Draw(image)
        self.displayTime(draw, x, y)            

        # table text
        org=(int(x*0.6),int(y*0.65))
        draw.text(org,'아래에 책상선을 맞춰주세요.', 
            font=ImageFont.truetype("./Nanum_hand_seongsil.ttf", self.font_size), fill=(255,255,255))
        # head circle text
        org=(int(x*0.3),int(y*0.45))
        draw.text(org, "얼굴이 보이도록 화면을 조정해주세요.", 
            font=ImageFont.truetype("./Nanum_hand_seongsil.ttf", self.font_size), fill=(255,255,255))
        image = np.array(image) # 한글사용 불가능하게 변경
                
        # table line                
        cv2.line(image,(0,int(y*0.75)),(x,int(y*0.75)),(255,255,255),1)
        # head circle
        cv2.circle(image,(int(x*0.5), int(y*0.3)), int(x*0.08), (255, 255, 255), 1, cv2.LINE_AA)

        return image