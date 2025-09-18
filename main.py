import streamlit as st
import numpy as np
import PIL.Image
import io
import requests
import google.generativeai as genai
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
import chromadb
from dotenv import load_dotenv
import os
import warnings


# Suppress warnings
warnings.filterwarnings("ignore")

# Load environment variables from .env file
load_dotenv()

# Get API key securely from environment variables
api_key = os.getenv("api_key")

# If API key is not found, show an error message
if not api_key:
    st.error("API key is not set in the environment variables. Please set it in your .env file.")
else:
    # Configure the genai API
    genai.configure(api_key=api_key)

# Function to format image inputs
def format_image_inputs(data):
    image_path = []
    image_path_1 = data['uris'][0][0]
    image_path_2 = data['uris'][0][1]
    image_path.append(image_path_1)
    image_path.append(image_path_2)
    return image_path

# Function to open image from various formats
def open_image(img_data):
    try:
        if isinstance(img_data, str):
            response = requests.get(img_data)
            img = PIL.Image.open(io.BytesIO(response.content))
        elif isinstance(img_data, np.ndarray):
            img = PIL.Image.fromarray(img_data.astype('uint8'))
        elif isinstance(img_data, list):
            img_data = np.array(img_data, dtype='uint8')
            img = PIL.Image.fromarray(img_data)
        else:
            raise ValueError("Unsupported image data format")
    except Exception as e:
        st.error(f"Error opening image: {e}")
        return None
    return img

# Streamlit app UI
st.title("AureaParis AI-Your personal AI Fashionist ")
st.write("Enter your styling query and get image-based recommendations, or upload an image to retrieve similar images.")

uploaded_file = st.file_uploader("Upload an image to retrieve similar images:", type=["jpg", "jpeg", "png"])
query = st.text_input("Or, enter your styling query:")

if st.button("Generate Styling Ideas / Retrieve Images"):
    chroma_client = chromadb.PersistentClient(path="Vector_database")
    image_loader = ImageLoader()
    CLIP = OpenCLIPEmbeddingFunction()
    image_vdb = chroma_client.get_or_create_collection(name="image", embedding_function=CLIP, data_loader=image_loader)

    # If an image is uploaded, process it
    if uploaded_file is not None:
        uploaded_image = np.array(PIL.Image.open(uploaded_file))
        
        # Retrieve similar images from the database
        try:
            retrieved_imgs = image_vdb.query(query_images=[uploaded_image], include=['data'], n_results=3)
            st.subheader("Retrieved Similar Images:")
            for i, img_data in enumerate(retrieved_imgs['data'][0]):
                img = open_image(img_data)
                if img:
                    st.image(img, caption=f"Retrieved Image {i+1}", use_container_width=True)

                    # Generate styling recommendations for the image
                    try:
                        prompt = ("You are a professional fashion and styling assistant with expertise in creating personalized outfit recommendations. "
                                  "Analyze the provided image carefully and give detailed fashion advice, including how to style and complement this item. "
                                  "Offer suggestions for pairing it with accessories, footwear, and other clothing pieces. "
                                  "Focus on the specific design elements, colors, and texture of the clothing item in the image. "
                                  "Based on the image, recommend how best to style this outfit to make a fashion statement.")
                        response = genai.GenerativeModel(model_name="gemini-1.5-pro").generate_content([prompt, img])
                        st.subheader("Styling Recommendations:")
                        st.write(response.text)
                    except Exception as e:
                        st.error(f"Error generating content: {e}")
        except Exception as e:
            st.error(f"Error retrieving similar images: {e}")

    # If a query is entered, retrieve relevant images based on the query
    if query:
        def query_db(query, results):
            try:
                return image_vdb.query(
                    query_texts=[query],
                    n_results=results,
                    include=['uris', 'distances'])
            except Exception as e:
                st.error(f"Error querying the database: {e}")
                return None
        
        results = query_db(query, results=2)
        if results:
            image_paths = format_image_inputs(results)
            try:
                sample_file_1 = PIL.Image.open(image_paths[0])
                sample_file_2 = PIL.Image.open(image_paths[1])

                # Display the retrieved images
                st.image(sample_file_1, caption="Image 1", use_container_width=True)
                st.image(sample_file_2, caption="Image 2", use_container_width=True)

                # Generate styling recommendations based on the query and the images
                try:
                    prompt = ("You are a professional fashion and styling assistant with expertise in creating personalized outfit recommendations. "
                              "Analyze the provided image carefully and give detailed fashion advice, including how to style and complement this item. "
                              "Offer suggestions for pairing it with accessories, footwear, and other clothing pieces. "
                              "Focus on the specific design elements, colors, and texture of the clothing item in the image. "
                              "This is the piece I want to wear: " + query + ". "
                              "Based on the image, recommend how best to style this outfit to make a fashion statement.")
                    response = genai.GenerativeModel(model_name="gemini-1.5-pro").generate_content([prompt, sample_file_1, sample_file_2])
                    st.subheader("Styling Recommendations:")
                    st.write(response.text)
                except Exception as e:
                    st.error(f"Error generating content: {e}")
            except Exception as e:
                st.error(f"Error opening image(s): {e}")
