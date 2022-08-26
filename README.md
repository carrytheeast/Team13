![m_logo](https://user-images.githubusercontent.com/77409431/186919386-8161a62b-0816-40fb-8a07-adc7688f55de.png)
![logo2](https://user-images.githubusercontent.com/77409431/186919405-930152c2-e72a-44ea-9209-c231b5e42ba9.png)

# A.Eye
한국 데이터 산업진흥원에서 주최한 데이터 청년 캠퍼스

고려대학교 소속 13조입니다.

This is Korea University Team 13 of Data Campus hosted by Korea Data Agency.

## Demo in colab

<a href="https://colab.research.google.com/drive/1Zx0zZMmj5Zyuf6RDV4EzPnjmeupha7fS?hl=ko#scrollTo=NfANEW0mu8oN"><img src="https://img.shields.io/badge/Demo-blue?style=flat-square&logo=googlecolab&#logoColor=white&link=https://colab.research.google.com/drive/1Zx0zZMmj5Zyuf6RDV4EzPnjmeupha7fS?hl=ko#scrollTo=NfANEW0mu8oN"/></a> 

```python
# You can clone github
!git clone https://github.com/carrytheeast/Team13

# Go to code folder
%cd Team13/project

# Download weights. If not working, You can try click to download release on this web. 
!wget https://github.com/carrytheeast/Team13/releases/download/v0.1/best.pt

# pip install required packages
!pip install -r requirements.txt
```

## Inference

On video:

```python
# detect.py
!python detect.py --weights best.pt --source your_video.mp4 --save_path your/path/name.mp4 --mode 0
```

## Woking

No Detect Face and Eyes:

Draw guide lines.

![사람이 없을 때](https://user-images.githubusercontent.com/98952505/186854480-c0510379-d948-4d12-9ff8-0ba8733d920b.png)


### User Mode

On Working:

User can check the belows.

- pure study time

- total time

![정상 작동중](https://user-images.githubusercontent.com/98952505/186854500-1ae1a163-fc6e-422e-86dc-ae9670082b6e.png)


When EAR is CLOSE, We show  the sentence at middle. (1)

![졸고있음_유저](https://user-images.githubusercontent.com/98952505/186854533-800e0d9e-efaf-46bd-aee6-f4ddae8aaf65.png)


### Developer Mode

On Working:

Developer can check the belows.

- Gaze Directions
- Gaze Lines
- Selected Area
- Objects Labels
- Object Bboxes
- Time

On Working:

![book응시 개발자모드](https://user-images.githubusercontent.com/98952505/186854565-6e91dbe6-c487-4eb9-bcb6-1fd3d8e3cd88.png)


![labtop응시 개발자모드](https://user-images.githubusercontent.com/98952505/186854582-4d2eb123-1bb4-4efc-97c3-ae5d93763621.png)


When EAR is CLOSE, We show the sentence at middle. (2)

![졸고있음](https://user-images.githubusercontent.com/98952505/186854614-11744ae4-cac0-40b6-8928-69e56623c841.png)


## Result Visualtization

After detect, Save video.mp4 and fig.jpg

![fig1 (4)](https://user-images.githubusercontent.com/98952505/186854344-e695787e-595f-4ecf-b2a5-726a5fc62e18.png)

## detec.py

`detec.py` supports the following options:

```python
usage: detect3.py [-h] [--weights WEIGHTS [WEIGHTS ...]] [--source SOURCE]
                  [--img-size IMG_SIZE] [--conf-thres CONF_THRES]
                  [--iou-thres IOU_THRES] [--device DEVICE] [--view-img]
                  [--classes CLASSES [CLASSES ...]] [--agnostic-nms]
                  [--augment] [--update] [--no-trace] [--save-path SAVE_PATH]
                  [--mode MODE]
```

## References

- [https://github.com/WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7)
- [https://github.com/google/mediapipe](https://github.com/google/mediapipe)
- [https://github.com/albumentations-team/albumentations](https://github.com/albumentations-team/albumentations)
