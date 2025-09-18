import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
from PIL import Image
import numpy as np
import os

# Define dataset folder and persistent Chroma client
dataset_folder = 'Data'
chroma_client = chromadb.PersistentClient(path="Vector_database")
image_loader = ImageLoader()
CLIP = OpenCLIPEmbeddingFunction()

# Create or get the 'image' collection in Chroma
image_vdb = chroma_client.get_or_create_collection(name="image", embedding_function=CLIP, data_loader=image_loader)

ids = []
uris = []
embeddings = []

# Loop through images in the dataset folder
for i, filename in enumerate(sorted(os.listdir(dataset_folder))):
    if filename.endswith('.png'):
        file_path = os.path.join(dataset_folder, filename)
        
        # Append the image id and URI
        ids.append(str(i))
        uris.append(file_path)
        
        # Open the image
        image = Image.open(file_path)

        # Convert the PIL image to a numpy array before passing it to _encode_image
        image_array = np.array(image)

        # Use _encode_image with the numpy array
        image_embedding = CLIP._encode_image(image_array)  # Pass numpy array
        embeddings.append(image_embedding)

# Add the image embeddings and metadata to the Chroma database
image_vdb.add(
    ids=ids,
    uris=uris,
    embeddings=embeddings
)

print("Images stored to the Vector database.")
