한국 데이터 산업진흥원에서 주최한 데이터 청년캠퍼스
고려대학교소속 13조입니다.

### Result

![fig1 (4)](https://user-images.githubusercontent.com/98952505/186835435-860c5de3-6dd9-49ba-9294-3126207ef0f3.png)


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

![사람이 없을 때](https://user-images.githubusercontent.com/98952505/186838991-7306c5d3-a0c5-411d-9452-d7c75aa170ca.png)

### UserMode

On Working:

User can check pure study time and total time 

![정상 작동중](https://user-images.githubusercontent.com/98952505/186839008-dc2d36a4-d81b-4aa5-9413-ed2be6b07991.png)

### Developer Mode

On Working:

Developer can check Gaze Directions, Gaze Line, Objects, Time, Object Bbox

![book응시 개발자모드](https://user-images.githubusercontent.com/98952505/186839030-a144ce05-00e5-47bf-af6c-e1302c4c8c0c.png)

![labtop응시 개발자모드](https://user-images.githubusercontent.com/98952505/186839043-991a97f5-c17a-4bf3-8e06-f8fff0cd756f.png)

### References

- [https://github.com/WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7)
- [https://github.com/google/mediapipe](https://github.com/google/mediapipe)
- [https://github.com/albumentations-team/albumentations](https://github.com/albumentations-team/albumentations)
