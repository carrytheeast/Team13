# A.Eye
한국 데이터 산업진흥원에서 주최한 데이터 청년 캠퍼스
고려대학교 소속 13조입니다.
### detec.py

`[detec.py](http://detec.py)` supports the following options:

```python
usage: detect.py [-h] [--weights WEIGHTS [WEIGHTS ...]] [--source SOURCE]
                 [--img-size IMG_SIZE] [--conf-thres CONF_THRES]
                 [--iou-thres IOU_THRES] [--device DEVICE] [--view-img]
                 [--classes CLASSES [CLASSES ...]] [--agnostic-nms]
                 [--augment] [--update] [--no-trace] [--save-path SAVE_PATH]
                 [--mode MODE]
```

### Result Visualtization

After detect, Save video.mp4 and fig.jpg

![fig1 (4)](https://user-images.githubusercontent.com/98952505/186854344-e695787e-595f-4ecf-b2a5-726a5fc62e18.png)


### Demo in colab

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

### Inference

On video:

```python
# detect.py
!python detect.py --weights [best.pt](http://best.pt/) --source your_video.mp4 --save_path your_path--mode 0
```

### Detecting

No Detect Face and Eyes:

Draw guide line

![사람이 없을 때](https://user-images.githubusercontent.com/98952505/186854480-c0510379-d948-4d12-9ff8-0ba8733d920b.png)


### UserMode

On Working:

User can get theirs pure study time and total time.

![정상 작동중](https://user-images.githubusercontent.com/98952505/186854500-1ae1a163-fc6e-422e-86dc-ae9670082b6e.png)


When ear is CLOSE, We show it like the sentence at middle. (1)

![졸고있음_유저](https://user-images.githubusercontent.com/98952505/186854533-800e0d9e-efaf-46bd-aee6-f4ddae8aaf65.png)


### Developer Mode

On Working:

Developer can check the belows.

- Gaze Directions
- Gaze Line
- Objects
- Object Bbox
- Time

On Working:

![book응시 개발자모드](https://user-images.githubusercontent.com/98952505/186854565-6e91dbe6-c487-4eb9-bcb6-1fd3d8e3cd88.png)


![labtop응시 개발자모드](https://user-images.githubusercontent.com/98952505/186854582-4d2eb123-1bb4-4efc-97c3-ae5d93763621.png)


When ear is CLOSE, We show it like the sentence at middle. (2)

![졸고있음](https://user-images.githubusercontent.com/98952505/186854614-11744ae4-cac0-40b6-8928-69e56623c841.png)


### References

- [https://github.com/WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7)
- [https://github.com/google/mediapipe](https://github.com/google/mediapipe)
- [https://github.com/albumentations-team/albumentations](https://github.com/albumentations-team/albumentations)
