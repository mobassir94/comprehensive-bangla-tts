# make sure you have FastApi installed
#https://towardsdatascience.com/step-by-step-approach-to-build-your-machine-learning-api-using-fast-api-21bd32f2bbdb

from starlette.responses import StreamingResponse
from multilingual_tts_v2 import BigTextToAudio
from typing import Optional
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
import uvicorn
import base64
import argparse
import json
import time
from argparse import Namespace
import soundfile as sf

app = FastAPI()


class TextJson(BaseModel):
    text: str

# Define the default route 
@app.get("/")
def root():
    return {"message": "Comprehensive Multilingual(Bangla,Arabic code mixed) TTS"}
    
@app.post("/TTS")
async def tts(input: TextJson):
    MLT_TTS=BigTextToAudio()

    text = input.text

    args = Namespace(**input.dict())

    args.wav = './comprehensive_mlt_synthesized.wav'

    if text:
        audio=MLT_TTS(text,resample_audio_to_out_sample_rate=True)
        sf.write(args.wav, audio, 22050)
  
    else:

        raise HTTPException(status_code=400, detail={"error": "No text"})
    

    ## to return outpur as a file
    audio = open(args.wav, mode='rb')
    return StreamingResponse(audio, media_type="audio/wav")


if __name__ == "__main__":
 
    uvicorn.run(
        "api:app", host="0.0.0.0", port=6006, log_level="debug"
    )

