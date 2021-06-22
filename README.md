# Real Time 3D Human Pose Estimation in Operating Rooms

![3D Pose Estimation](rcnn_seffpose_test.gif)

## Description

Two models implemented

* Pose 2D
   * Keypoint RCNN
   * OpenPose

* Pose 3D
   * SeffPose

## Installation
1. Clone this repository
```bash
git clone https://github.com/rob-med-usp/human_pose.git
```
2. Install pytorch for your machine using the link:

<https://pytorch.org/get-started/locally/>

3. Install OpenCV, Matplotlib, Numpy and Pillow.

```bash
pip install opencv-contrib-python matplotlib numpy Pillow
```

4. Download pre-trained model from this link.

<https://drive.google.com/file/d/1CSpx5hGD18y8Wp_RysoxUhIPvgYuzl9c/view?usp=sharing>

5. Move the pre-trained model to the models/seffpose/ folder.

## Usage
For image:
```bash
python rcnn_seffpose_image.py
```
For video:
```python
python rcnn_seffpose_video.py
```
For webcam:
```bash
python rcnn_seffpose_webcam.py
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
