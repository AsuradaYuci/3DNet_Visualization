# 3D Net Visualization Tools

## Demo

**For an input video, this project will show attention map in video and frames.**

### saved video

Video can't be show here, there are some gif.

**Video with Clip_step 1**

![gif](https://github.com/FingerRec/3DNet_Visualization/raw/master/resources/step_1.gif)

**Video with Clip_step 4**

![gif_2](https://github.com/FingerRec/3DNet_Visualization/raw/master/resources/step_4.gif)


**Video withClip_step 16**

![gif_3](https://github.com/FingerRec/3DNet_Visualization/raw/master/resources/step_16.gif)


### saved img

**heatmap**

![heatmap_image](https://github.com/FingerRec/3DNet_Visualization/raw/master/resources/heatmap_1.png)

**focus map**

![focus_image](https://github.com/FingerRec/3DNet_Visualization/raw/master/resources/focusimg_1.png)

## Require:
- pytorch
- opencv
- numpy
- skvideo

## Run:
### 1.create pretrain_model dir
```bash
git clone https://github.com/FingerRec/3DNet_Visualization.git
cd 3DNet_Visualization
mkdir pretrained_model
```

### 2.download pretrained model
download pretrained MFNet on UCF101 from [google_drive](https://goo.gl/mML2gv) and put it into directory pretrained_model,
which is from [MFNet](https://github.com/cypw/PyTorch-MFNet)

### 3.run demo

```bash
bash demo.sh
```

The generate video and imgs will be put in dir output/imgs and output/video.

Tip: in main.py, if set clip_steps is 1, will generate a video the same length as origin.

### 4.test own video

the details in demo.sh as follow, change --video and --label accorading to your video, please refer to resources/classInd.txt for label information for UCF101 videos.

```bash
python main.py --num_classes 101 \
--classes_list resources/classInd.txt \
--model_weights pretrained_model/MFNet3D_UCF-101_Split-1_96.3.pth \
--video test_videos/v_ApplyEyeMakeup_g01_c01.avi \
--frames_num 16 --label 0 --clip_steps 16 \
--output_dir output
```

Tip:UCF101 dataset is support now, for HMDB and Kinetics et al. Just download a pretrained model and change --classes_list

## To Do List
- [ ] support s3d, i3d and c3d
- [ ] visualize filters
- [ ] support multi fc layers or full convolution networks

## Acknowledgment
This project is highly based on [SaliencyTubes](https://github.com/alexandrosstergiou/Saliency-Tubes-Visual-Explanations-for-Spatio-Temporal-Convolutions) 
, [MF-Net](https://github.com/cypw/PyTorch-MFNet) and [st-gcn](https://github.com/yysijie/st-gcn).