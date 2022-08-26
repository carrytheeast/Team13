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
!wget https://github.com/carrytheeast/Team13/release/download/v0.1/best.pt

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

![사람이 없을 때.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/d78b6c99-583f-472e-af61-e59529a30a17/%EC%82%AC%EB%9E%8C%EC%9D%B4_%EC%97%86%EC%9D%84_%EB%95%8C.png)

### UserMode

On Working:

User can check pure study time and total time 

![정상 작동중.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/41bd9dea-4b29-4931-acbb-be38e4bd7230/%EC%A0%95%EC%83%81_%EC%9E%91%EB%8F%99%EC%A4%91.png)

### Developer Mode

On Working:

Developer can check Gaze Directions, Gaze Line, Objects, Time, Object Bbox

![book응시 개발자모드.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/f38b59f3-0cf4-4e81-aa52-54c40ce9be01/book%EC%9D%91%EC%8B%9C_%EA%B0%9C%EB%B0%9C%EC%9E%90%EB%AA%A8%EB%93%9C.png)

![labtop응시 개발자모드.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/dfb0c957-ca03-474b-bf83-f23c32fc30ad/labtop%EC%9D%91%EC%8B%9C_%EA%B0%9C%EB%B0%9C%EC%9E%90%EB%AA%A8%EB%93%9C.png)













### References

- [https://github.com/WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7)
- [https://github.com/google/mediapipe](https://github.com/google/mediapipe)
- [https://github.com/albumentations-team/albumentations](https://github.com/albumentations-team/albumentations)
