import os
from dotenv import load_dotenv
import vertexai

from vertexai.preview.generative_models import (
    GenerationConfig,
    GenerativeModel,
    HarmCategory,
    HarmBlockThreshold,
    Part,
)

load_dotenv()

PROJECT_ID = os.environ["PROJECT_ID"]
REGION = os.environ["REGION"]

vertexai.init(project=PROJECT_ID, location=REGION)

MODEL_ID = "gemini-1.5-pro-preview-0514"

model = GenerativeModel(MODEL_ID)

example_model = GenerativeModel(
    MODEL_ID,
    system_instruction=[
        "You are a helpful language translator.",
        "Your mission is to translate text in English to French.",
    ],
)

# Set model parameters
generation_config = GenerationConfig(
    temperature=0.9,
    top_p=1.0,
    top_k=32,
    candidate_count=1,
    max_output_tokens=8192,
)

# Set safety settings
safety_settings = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
}

prompt = """
  User input: I like bagels.
  Answer:
"""

# Set contents to send to the model
contents = [prompt]

# Counts tokens
print(example_model.count_tokens(contents))

# Prompt the model to generate content
response = example_model.generate_content(
    contents,
    generation_config=generation_config,
    safety_settings=safety_settings,
)

# Print the model response
print(f"\nAnswer:\n{response.text}")
print(f'\nUsage metadata:\n{response.to_dict().get("usage_metadata")}')
print(f"\nFinish reason:\n{response.candidates[0].finish_reason}")
print(f"\nSafety settings:\n{response.candidates[0].safety_ratings}")

# Audio understanding
audio_file_path = "cloud-samples-data/generative-ai/audio/pixel.mp3"
audio_file_uri = f"gs://{audio_file_path}"
audio_file_url = f"https://storage.googleapis.com/{audio_file_path}"

# Example 1: Summarization
prompt = """
  Please provide a summary for the audio.
  Provide chapter titles, be concise and short, no need to provide chapter summaries.
  Do not make up any information that is not part of the audio and do not be verbose.
"""

audio_file = Part.from_uri(audio_file_uri, mime_type="audio/mpeg")
contents = [audio_file, prompt]

response = model.generate_content(contents)
print("audio summary:\n", response.text)

# Example 2: Transcription
prompt = """
    Can you transcribe this interview, in the format of timecode, speaker, caption.
    Use speaker A, speaker B, etc. to identify the speakers.
"""

audio_file = Part.from_uri(audio_file_uri, mime_type="audio/mpeg")
contents = [audio_file, prompt]

responses = model.generate_content(contents, stream=True)

for response in responses:
    print(response.text)

# Video with audio understanding
video_file_path = "cloud-samples-data/generative-ai/video/pixel8.mp4"
video_file_uri = f"gs://{video_file_path}"
video_file_url = f"https://storage.googleapis.com/{video_file_path}"

prompt = """
  Provide a description of the video.
  The description should also contain anything important which people say in the video.
"""

video_file = Part.from_uri(video_file_uri, mime_type="video/mp4")
contents = [video_file, prompt]

response = model.generate_content(contents)
print("video and audio description:\n", response.text)

# PDF document analysis
pdf_file_uri = "gs://cloud-samples-data/generative-ai/pdf/2403.05530.pdf"

prompt = """
  Your are a very professional document summarization specialist.
  Please summarize the given document.
"""

pdf_file = Part.from_uri(pdf_file_uri, mime_type="application/pdf")
contents = [pdf_file, prompt]

response = model.generate_content(contents)
print("pdf details:\n", response.text)

# images
image_file_path = "cloud-samples-data/generative-ai/image/cumulative-average.png"
image_file_url = f"https://storage.googleapis.com/{image_file_path}"
image_file_uri = f"gs://{image_file_path}"

prompt = """
Task: Answer the following questions based on a PDF document and image file provided in the context.

Instructions:
- Look through the image and the PDF document carefully and answer the question.
- Give a short and terse answer to the following question.
- Do not paraphrase or reformat the text you see in the image.
- Cite the source of page number for the PDF document provided as context.

  Questions:
  - What is in the given image?
  - Is there a similar graph in the given document?

Context:
"""

contents = [
    pdf_file,
    image_file_uri,
    prompt,
]

response = model.generate_content(contents)
print(response.text)

# All modalities (images, video, audio, text) at once
video_file_path = "cloud-samples-data/generative-ai/video/behind_the_scenes_pixel.mp4"
video_file_uri = f"gs://{video_file_path}"
video_file_url = f"https://storage.googleapis.com/{video_file_path}"

image_file_path = "cloud-samples-data/generative-ai/image/a-man-and-a-dog.png"
image_file_uri = f"gs://{image_file_path}"
image_file_url = f"https://storage.googleapis.com/{image_file_path}"

video_file = Part.from_uri(video_file_uri, mime_type="video/mp4")
image_file = Part.from_uri(image_file_uri, mime_type="image/png")

prompt = """
  Look through each frame in the video carefully and answer the questions.
  Only base your answers strictly on what information is available in the video attached.
  Do not make up any information that is not part of the video and do not be too
  verbose, be to the point.

  Questions:
  - When is the moment in the image happening in the video? Provide a timestamp.
  - What is the context of the moment and what does the narrator say about it?
"""

contents = [video_file, image_file, prompt]

response = model.generate_content(contents)
print(response.text)

