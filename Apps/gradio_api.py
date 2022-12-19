
from multilingual_tts_v2 import BigTextToAudio
import soundfile as sf
 
import gradio as gr
import argparse
import numpy as np
from argparse import Namespace


def run_tts(textbox):
    inputs_to_gradio = {'text' : textbox}

    args = Namespace(**inputs_to_gradio)
    #args.wav = None

    if args.text:
        MLT_TTS=BigTextToAudio()
        audio=MLT_TTS(args.text,resample_audio_to_out_sample_rate=True)
        args.wav = './comprehensive_mlt_synthesized_tts.wav'
        sf.write(args.wav, audio, 22050)
  
        return 'comprehensive_mlt_synthesized_tts.wav' #audio


if __name__ == "__main__":
  
    textbox = gr.inputs.Textbox(placeholder="Enter Code Mixed (Bangla,Arabic) Text to run", default="", label="TTS")
   
    op = gr.outputs.Audio(type="numpy", label=None)

    inputs_to_gradio = [textbox]
    iface = gr.Interface(fn=run_tts, inputs=inputs_to_gradio, outputs=op, title='Multilingual(Bangla,Arabic code mixed) TTS')
    iface.launch(share=True, enable_queue=True)

