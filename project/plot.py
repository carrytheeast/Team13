# plot.py 

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
# import matplotlib.font_manager as fm

def printPlot(fps_cnt_list, total_fps_cnt_list, fps_obj_list):

    # # 한글 폰트 사용을 위해서 세팅
    # fontprop = fm.FontProperties(fname="NanumGothic.ttf")

    df = pd.DataFrame({'FPS':fps_cnt_list, 'TOTAL':total_fps_cnt_list,'USING_OBJECT':fps_obj_list})
    f, ax = plt.subplots(1,2, figsize=(16,9)) 

    x =df['TOTAL']
    y= df['FPS']
    z = df['USING_OBJECT']

    ax1 = sns.scatterplot(x=x,y=y,ax=ax[0], hue=z) 

    ax1.set_title('Actual study time for total hours',fontsize=20)
    ax1.set_xlabel('Total Hours', fontsize=13)
    ax1.set_ylabel('Actual Study Time', fontsize=13)

    ax2 = sns.countplot(x= df['USING_OBJECT'], ax=ax[1])

    ax2.set_title('Gaze Frequency',fontsize=20)
    ax2.set_ylabel('Gaze Frequency(10fps)', fontsize=13)
    ax2.set_xlabel('Object Name', fontsize=13)

    plt.savefig('fig1.png', dpi=100)
    plt.show()    
    print('save as', 'fig1.png')