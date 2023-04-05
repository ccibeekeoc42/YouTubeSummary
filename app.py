import os
import whisper
from flask import Flask, request, render_template
from pytube import YouTube


import openai
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
# import pandas as pd
# import numpy as np
# from openai.embeddings_utils import get_embedding, cosine_similarity

from config import OPENAI_API_KEY
openai.api_key = OPENAI_API_KEY
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
model = whisper.load_model('tiny') # seleting the base

app = Flask(__name__)

@app.route("/")
def search_form():
    """Function to render the search form"""
    return render_template("search_form.html")


@app.route("/search")
def search():
    """Function to search query"""
    # get search query from the URL query string
    youtube_video_url = request.args.get("query")
    youtube_video = YouTube(youtube_video_url)
    stream = youtube_video.streams.filter(only_audio=True).first()
    stream.download(filename='plugins.mp4') # Saving the audio file
    url = youtube_video.embed_url

    # Using the whisper model to transcribe the audio
    output = model.transcribe("plugins.mp4")

    # Initializing the LLM
    llm = OpenAI(model_name="gpt-3.5-turbo")
    template = """
        {content}
        """
    prompt = PromptTemplate(input_variables=['content'], template=template)
    chain = LLMChain(llm=llm, prompt=prompt)
    results = chain.run(output['text'])
    
    # renders the search results passing the search query and the results
    return render_template("search_results.html", url=url, results=results) 


if __name__ == "__main__":
    app.run(debug=True) 