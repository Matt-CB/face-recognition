# Face-Recognition model pipeline with blacklist/whitelist

PowerVision Face Recognition Model with Blacklist/Whitelist:  
Preprocessing: RGB + Resize 160x160px + Facenet2018 normalization  
Face Detection Model: YunetOpencv  
Face Embedding Model: Facenet512  
Vector Database: Pinecone  
Indexing Algorithm Type: HNSW (Hierarchical Navigable Small World)  
Similarity Metric: Cosine Similarity/Euclidean Distance  
Search Algorithm: K-Nearest Neighbors (KNN)  



**Facenet512** is a face recognition model that uses an Inception ResNet architecture. It is pretrained on the VGGFace2 dataset and provides a 512-dimensional embedding for each face. This model is known for its high accuracy in face recognition tasks and is often used for tasks such as face verification and clustering

**YuNet** is a lightweight, fast, and accurate face detection model developed by OpenCV. It achieves high performance on the WIDER Face validation set, with scores of 0.834 (AP_easy), 0.824 (AP_medium), and 0.708 (AP_hard). YuNet is designed to detect faces with sizes ranging from 10x10 to 300x300 pixels. It is optimized for real-time applications and can be used with OpenCV’s DNN module.




## How to use and requirments:

Python version required: 3.13> It can be version 3.12, 3.11, 3.10

.Install the requirements
```
pip install -r requirements.txt
```

. [How to install Deepface library](https://github.com/serengil/deepface)

. [Pinecone web for obtain a API Key Free](https://www.pinecone.io/)
