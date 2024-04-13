import base64
import os
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage
from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI


load_dotenv()


def encode_image(image_path):
    """Getting the base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def image_summarize(img_base64, prompt):
    print("Summarising image")
    """Make image summary"""
    # chat = Ollama(base_url='http://119.74.32.2:11434',
    #               model="llava", temperature=0)
    chat = ChatOpenAI(model="gpt-4-turbo",
                      api_key=os.getenv("OPENAI_API_KEY"), max_tokens=4096)

    msg = chat.invoke([HumanMessage(content=[
        {"type": "text", "text": prompt},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
        },
    ]
    )])

    return msg.content


def generate_img_summaries(path):
    """
    Generate summaries and base64 encoded strings for images
    path: Path to list of .jpg files extracted by Unstructured
    """

    # Store base64 encoded images
    img_base64_list = []

    # Store image summaries
    image_summaries = []

    # Prompt
    prompt = """You are an assistant tasked with summarizing images for retrieval. \
    These summaries will be embedded and used to retrieve the raw image. \
    Give a concise summary of the image that is well optimized for retrieval."""

    # Apply to images
    for img_file in sorted(os.listdir(path)):
        if img_file.endswith(".jpg"):
            img_path = os.path.join(path, img_file)
            base64_image = encode_image(img_path)
            img_base64_list.append(base64_image)
            image_summaries.append(image_summarize(base64_image, prompt))

    return img_base64_list, image_summaries


# Image summaries
img_base64_list, image_summaries = generate_img_summaries(
    "/Users/aaron/Documents")
print(image_summaries)
