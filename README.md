# Image_Processing
This project is Image processing project for take the count of moving objects.detections done via mobilenet V2 neural network and count taking from Opencv based centroid tracking algorithm.Image input is taken using a camera connected to raspberry pi b3+  and output shown to the user vis android apps developed using java.

model was trained via tensorflow library by loaded via below code piece.

from tensorflow.keras.applications import MobileNetV2

the resulting count could be achived without any issue with the speed of the vehicle as below.

# Vehicle with constant speed and from front view

![image](https://github.com/user-attachments/assets/71144111-3c4f-4e2d-864f-2c6b84474ad1)

# Vehicle with speed dropping and in side view

![image](https://github.com/user-attachments/assets/20f30adf-787d-4403-9e09-dd2ee5686d66)

# Vehicle with speed dropping and in top view

![image](https://github.com/user-attachments/assets/855e922d-7fda-49ca-9a9b-42bc8a48715e)
