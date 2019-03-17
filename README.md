# 3D Net Visualization Tools

## Demo

**For an input video, this project will show attention map in video and frames.**

### saved video

**Video with Clip_step 1**

![gif](https://github.com/FingerRec/3DNet_Visualization/raw/master/resources/step_1.gif)

**Video with Clip_step 8**

![gif_2](https://github.com/FingerRec/3DNet_Visualization/raw/master/resources/step_8.gif)


**Video withClip_step 16**

![gif_3](https://github.com/FingerRec/3DNet_Visualization/raw/master/resources/step_16.gif)


### saved img

![image](https://github.com/FingerRec/3DNet_Visualization/raw/master/resources/img_1.png)

## Require:
pytorch
opencv
numpy
skvideo

## Run:
### 1.download pretrained model
download pretrained MFNet on UCF101 from [google_drive](https://goo.gl/mML2gv) and put it into directory pretrained_model,
which is from [MFNet](https://github.com/cypw/PyTorch-MFNet)
### 2.run shell script
```bash
bash demo.sh
```
Tip: in main.py, if set clip_steps is 1, will generate a video the same length as origin.

## To Do List
- plot labels and prob
- add text
- support i3d and c3d



## Acknowledgment
This project is highly based on [SaliencyTubes](https://github.com/alexandrosstergiou/Saliency-Tubes-Visual-Explanations-for-Spatio-Temporal-Convolutions) 
, [MF-Net](https://github.com/cypw/PyTorch-MFNet) and [st-gcn](https://github.com/yysijie/st-gcn).