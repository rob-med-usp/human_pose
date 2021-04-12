# import cv2
# import matplotlib
# import torch
# import torchvision
# import numpy as np
# import argparse
# import time

# from PIL import Image
# from torchvision.transforms import transforms as transforms

# # pairs of edges for 17 of the keypoints detected ...
# # ... these show which points to be connected to which point ...
# # ... we can omit any of the connecting points if we want, basically ...
# # ... we can easily connect less than or equal to 17 pairs of points ...
# # ... for keypoint RCNN, not  mandatory to join all 17 keypoint pairs
# edges = [
#     (0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10),
#     (5, 7), (7, 9), (5, 11), (11, 13), (13, 15), (6, 12),
#     (12, 14), (14, 16), (5, 6)
# ]

# def draw_keypoints(outputs, image):
#     # the `outputs` is list which in-turn contains the dictionaries
#     for i in range(len(outputs[0]['keypoints'])):
#         keypoints = outputs[0]['keypoints'][i].cpu().detach().numpy()
#         # proceed to draw the lines if the confidence score is above 0.9
#         if outputs[0]['scores'][i] > 0.975:
#             keypoints = keypoints[:, :].reshape(-1, 3)
#             for p in range(keypoints.shape[0]):
#                 # draw the keypoints
#                 cv2.circle(image, (int(keypoints[p, 0]), int(keypoints[p, 1])),
#                             3, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
#                 # uncomment the following lines if you want to put keypoint number
#                 # cv2.putText(image, f"{p}", (int(keypoints[p, 0]+10), int(keypoints[p, 1]-5)),
#                 #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
#             for ie, e in enumerate(edges):
#                 # get different colors for the edges
#                 rgb = matplotlib.colors.hsv_to_rgb([
#                     ie/float(len(edges)), 1.0, 1.0
#                 ])
#                 rgb = rgb*255
#                 # join the keypoint pairs to draw the skeletal structure
#                 cv2.line(image, (keypoints[e, 0][0], keypoints[e, 1][0]),
#                         (keypoints[e, 0][1], keypoints[e, 1][1]),
#                         tuple(rgb), 2, lineType=cv2.LINE_AA)
#         else:
#             continue
#     return image

# transform = transforms.Compose([
#     transforms.ToTensor()
# ])

# # initialize the model
# model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True,
#                                                                num_keypoints=17)
# # set the computation device
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# # load the modle on to the computation device and set to eval mode
# model.to(device).eval()

# cam = cv2.VideoCapture(0)

# while True:

#     t = time.time()

#     ret, frame = cam.read()

#     if not ret:
#         print('Waiting for camera')
#         continue

#     #pil_image = Image.fromarray(frame).convert('RGB')
#     #frame_resized = frame
#     #dsize = (int(0.2*frame.shape[1]), int(0.2*frame.shape[0]))
#     #frame = cv2.resize(frame,dsize) 
#     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    

#     # transform the image
#     #image = transform(pil_image)
#     image = transform(rgb)

#     # add a batch dimension
#     image = image.unsqueeze(0).to(device)

#     with torch.no_grad():
#         outputs = model(image)
    
#     # visualize skeleton and fps
#     output_image = draw_keypoints(outputs, frame)
#     cv2.putText(output_image, str(round(1/(time.time()-t),2)), (30, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0))
#     print(round(1/(time.time()-t),2))

#     # visualize the image
#     cv2.imshow('Keypoint image', output_image)
#     if cv2.waitKey(1) == 27:
#             break

import pose2D_rcnn as pose2D

pose2D = pose2D.Pose2D_RCNN()

pose2D.defineModel()

pose2D.setWebcam(0)

while True:

    ret, frame = pose2D.getWebcamFrame()

    if ret:
        outputs = pose2D.predictFrame(frame)
        image = pose2D.drawSkeleton(frame, outputs)
    
        pose2D.showImage(frame)



