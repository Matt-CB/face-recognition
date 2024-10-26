import cv2
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pinecone import Pinecone, ServerlessSpec
from cv2 import FaceDetectorYN  # YuNet for face detection
import numpy as np
from deepface import DeepFace
import gc  # garbage collection

# Vector DB and index initialization
pc = Pinecone(
    api_key="Your API KEY"
)

index_name = 'facial-recognition'
if 'facial-recognition' not in pc.list_indexes().names():
    pc.create_index(
        name='facial-recognition',
        dimension=512,
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )

index = pc.Index(index_name)

def search_in_namespace(query_embedding, top_k, namespace):
    result = index.query(vector=query_embedding.tolist(), top_k=top_k, namespace=namespace)
    return result

SIMILARITY_THRESHOLD = 0.3  # Less is more similar, adjust according to your case
SIMILARITY_THRESHOLD_ED = 10  # Threshold for Euclidean distance comparison

# Add a counter to track consecutive blacklist detections
consecutive_blacklist_count = 0
BLACKLIST_CONFIRMATION_THRESHOLD = 2  # Number of consecutive frames required to confirm blacklist

def detect_face_in_list(results, threshold):
    for match in results['matches']:
        if match['score'] > threshold and 'values' in match:
            print("Blacklist face?")
            return True, match['id'], np.array(match['values'])  # Return True, the match ID, and the embedding values
    return False, None, None

cap = cv2.VideoCapture("rtsp://admin:TADDEO*1303@190.104.131.254:554/Streaming/channels/101")
frame_count = 0
face_id = 1

# Limit the number of threads to avoid excessive resource usage
executor = ThreadPoolExecutor(max_workers=1)

# Initialize YuNet (FaceDetectorYN) for Face Detection
yunet_model_path = r'models\face_detection_yunet_2023mar.onnx'
face_detector = FaceDetectorYN.create(yunet_model_path, "", (320, 320))

# Function to process face detection asynchronously
def process_faces(frame):
    global face_id, consecutive_blacklist_count
    start_time = time.time()
    # Detect faces using YuNet
    height, width, _ = frame.shape
    face_detector.setInputSize((width, height))
    results = face_detector.detect(frame)
    end_time = time.time()
    print(f"Detection time: {end_time - start_time} seconds")
    #print("Detection results (raw):", results)
    
    if results[1] is not None and len(results[1]) > 0:
        for detection in results[1]:
            x1, y1, w, h = detection[:4].astype(int)
            x2 = x1 + w
            y2 = y1 + h

            # Ensure coordinates are within the bounds of the image
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width, x2)
            y2 = min(height, y2)

            if x2 > x1 and y2 > y1:
                face_img = frame[y1:y2, x1:x2]

                try:
                    # Generate embedding without saving the face image
                    embedding_objs = DeepFace.represent(
                        img_path=face_img,
                        model_name='Facenet512',
                        normalization='Facenet2018',
                        enforce_detection=False
                    )
                    embeddings_array = np.array([embedding_objs[0]['embedding']])
                    embeddings_array = embeddings_array.reshape(1, -1)
                    blacklist_results = search_in_namespace(embeddings_array, top_k=1, namespace="blacklist")
                    is_blacklisted, top_match_id, top_match_embedding = detect_face_in_list(blacklist_results, SIMILARITY_THRESHOLD)
                    if is_blacklisted and top_match_embedding.size == 512:
                        # Compare the current embedding with the top matched embedding using Euclidean distance
                        distance = np.linalg.norm(embeddings_array - top_match_embedding.reshape(1, -1))
                        print(f"Distance to top match: {distance}")
                        if distance < SIMILARITY_THRESHOLD_ED:
                            consecutive_blacklist_count += 1
                            if consecutive_blacklist_count >= BLACKLIST_CONFIRMATION_THRESHOLD:
                                print("\n blacklist?")
                                consecutive_blacklist_count = 0
                        else:
                            consecutive_blacklist_count = 0
                    elif detect_face_in_list(blacklist_results, SIMILARITY_THRESHOLD)[0]:
                        consecutive_blacklist_count += 1
                        if consecutive_blacklist_count >= BLACKLIST_CONFIRMATION_THRESHOLD:
                            print("\n BLACKLIST FACE DETECTED")
                            consecutive_blacklist_count = 0
                    elif detect_face_in_list(search_in_namespace(embeddings_array, top_k=1, namespace="whitelist"), SIMILARITY_THRESHOLD)[0]:
                        consecutive_blacklist_count = 0  # Reset count if not consecutively blacklisted
                        print("\n WHITELIST FACE DETECTED")
                    else:
                        consecutive_blacklist_count = 0  # Reset count if not consecutively blacklisted
                        print("\n UNKNOWN FACE")
                except Exception as e:
                    print(f"Error generating embeddings with DeepFace: {e}")
            
            del face_img
            gc.collect()  # Collect garbage to free memory
    else:
        print("No faces detected in the frame.")

while True:
    cap.grab()
    
    ret, frame = cap.read()
    if not ret:
        break

    # Apply slight increase in contrast and brightness to improve face detection
    frame = cv2.convertScaleAbs(frame, alpha=1.05, beta=10) #OPTIONAL. Just if is dark

    frame_count += 1

    if frame_count % 15 == 0:
        executor.submit(process_faces, frame.copy())

    cv2.imshow('Frame', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    time.sleep(0.1)  # Pause to free up resources

cap.release()
cv2.destroyAllWindows()
executor.shutdown(wait=True)
