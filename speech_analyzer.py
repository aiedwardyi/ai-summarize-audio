import torch
import os
import gradio as gr
from dotenv import load_dotenv # Added for local secrets
from transformers import pipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams

# 1. Load your real credentials from the .env file
load_dotenv()

my_credentials = {
    "url"    : "https://jp-tok.ml.cloud.ibm.com",
    "apikey" : os.getenv("WATSONX_APIKEY") # Pulls from .env
}

params = {
    GenParams.MAX_NEW_TOKENS: 800,
    GenParams.TEMPERATURE: 0.1,
}

# 2. Use your personal Project ID from .env
LLAMA3_model = Model(
    model_id= 'meta-llama/llama-3-2-11b-vision-instruct', 
    credentials=my_credentials,
    params=params,
    project_id=os.getenv("WATSONX_PROJECT_ID"), # Pulls from .env
)

llm = WatsonxLLM(LLAMA3_model)  

#######------------- Updated Prompt Template for Llama 3-------------####

# This fixes the "SYS SYS" repetition issue for Llama 3.2
temp = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
List the key points with details from the context:<|eot_id|>
<|start_header_id|>user<|end_header_id|>
The context: {context}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""

pt = PromptTemplate(
    input_variables=["context"],
    template=temp)

prompt_to_LLAMA3 = LLMChain(llm=llm, prompt=pt)

#######------------- Speech2text-------------####

def transcript_audio(audio_file):
    pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-tiny.en",
        chunk_length_s=30,
    )
    transcript_txt = pipe(audio_file, batch_size=8)
    # Using the updated chain name
    result = prompt_to_LLAMA3.run({"context": transcript_txt})
    return result

#######------------- Gradio-------------####

audio_input = gr.Audio(sources=["upload"], type="filepath") # Fixed sources list
output_text = gr.Textbox()

iface = gr.Interface(fn=transcript_audio, 
                    inputs=audio_input, 
                    outputs=output_text, 
                    title="Audio Transcription App",
                    description="Upload the audio file")

# Use 127.0.0.1 for local testing on Windows
iface.launch(server_name="127.0.0.1", server_port=7860)
