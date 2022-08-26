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

### References

- [https://github.com/WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7)
- [https://github.com/google/mediapipe](https://github.com/google/mediapipe)
- [https://github.com/albumentations-team/albumentations](https://github.com/albumentations-team/albumentations)
