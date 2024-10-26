import tensorflow as tf
import matplotlib.image as mpimg
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import cv2
from pinecone import Pinecone, ServerlessSpec

@tf.keras.utils.register_keras_serializable()  
def scaling(x, scale):
    return x * scale

# Load model
model = load_model(r'models\facenet512\facenet512_model.h5')

def preprocess_image(img_bgr): # For facenet512 embedding model
    # Convert to RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Resize to 160x160
    img_resized = cv2.resize(img_rgb, (160, 160))

    # Normalize the image between -1 and 1
    img_normalized = img_resized / 127.5 - 1
    
    # Expand dimensions to make it compatible with the model
    img_result = np.expand_dims(img_normalized, axis=0)
    
    return img_result, img_resized  # Return normalized and RGB image

def display_image(img_rgb):
    # Display image using Matplotlib
    plt.imshow(img_rgb)
    plt.axis('off') 
    plt.title("Processed Image")
    plt.show()

# Initialize Pinecone
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

# Path to the image to be processed
img_path = r'faces\sample_face.jpg'
img = cv2.imread(img_path)

# Preprocess the image
img_result, img_resized = preprocess_image(img)

# Display the preprocessed image
display_image(img_resized)

# Get embeddings from the preprocessed image
embeddings = model.predict(img_result)
print("Generated facial embeddings successfully", embeddings)

def upload_embeddings(embedding, namespace):
    """
    Uploads the embedding to Pinecone in the specified namespace.
    
    Parameters:
    - embedding: The embedding to upload.
    - namespace: Namespace (Blacklist or Whitelist).
    """
    vector = [("vector_1", embedding.flatten().tolist())]
    index.upsert(vectors=vector, namespace=namespace)
    print("\nUploaded correctly to the vectorDB!\n")

# Upload the embedding directly
upload_embeddings(embeddings, 'blacklist')
