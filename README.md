한국 데이터 산업진흥원에서 주최한 데이터 청년캠퍼스
고려대학교소속 13조입니다.

## Code 설명

기존의 .ipynb 형식의 코드를 .py 형식으로 바꾸면서 main과 utils로 나누었습니다. 

from utils import * 로 import하여 사용하기 때문에 크게 신경 쓰지 않아도 됩니다. 

**그래도 main.py와 utils.py는 같은 폴더에 넣어주세요**

아래는 각 .py파일에 대한 설명입니다. 

`[main.py](https://www.notion.so/main-py-0d44d7e6b3e5479dbce08da2ac7af4a5)` :  메인

`[utils.py](https://www.notion.so/utils-py-1d6af85ebb2b4b8e8c65f6da52188667)` : 기타 함수 (작성 예정)

### 실행

아래 코드로 실행할 수 있습니다. 

- cmd창에서도 실행할 수 있지만 별도로 파이썬이나 모듈들의 설치가 귀찮은 경우 **주피터 노트북의 .ipynb**에서 실행시키면 됩니다. (**같은 폴더에 있어야 합니다.**) 현재 colab에서 실행할 경우 오류가 발생합니다.

```python
!python main.py
```

현재까지 옵션을 2가지 구현했습니다.  (`--source-path`와 `--output-path`)

**default** (옵션을 입력하지 않고 위의 코드로 실행시키면 기본값이 들어갑니다.)

- source path는 shorts12.mp4로 설정되어 있습니다. **같은 폴더에 해당 영상에 있어야 합니다.**
- save path는 run.mp4로 설정되어 있습니다. 같은 폴더에 저장됩니다.

**옵션을 변경하면 폴더와 파일명을 정해줄 수 있습니다.**

아래의 코드로 short24.mp4를 입력으로 받고 output 폴더에 new_run.mp4로 저장됩니다.

```python
!python main.py --source-path shorts24.mp4 --save-path output/new_run.mp4
```
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
