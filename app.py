import os
import re
import torch
import whisper
import validators
import gradio as gr

from wordcloud import WordCloud, STOPWORDS

from scipy.io.wavfile import write
from espnet2.bin.tts_inference import Text2Speech

from utils import *

# load whisper model for ASR and BART for summarization
default_model = 'base.en' if torch.cuda.is_available() else 'tiny.en'
asr_model = whisper.load_model(default_model)
summarizer = gr.Interface.load("facebook/bart-large-cnn", src='huggingface')
tts_model = Text2Speech.from_pretrained("espnet/kan-bayashi_ljspeech_joint_finetune_conformer_fastspeech2_hifigan")


def load_model(name: str):
    """

    :param name: model options, tiny or base only, for quick inference
    :return:
    """
    global asr_model
    asr_model = whisper.load_model(f"{name.lower()}")
    return name


def audio_from_url(url, dst_dir='data', name=None, format='wav'):
    """ Download video from url and save the audio from video

    :param url: str, the video url
    :param dst_dir: destination directory for save audio
    :param name: audio file's name, if none, assign the name as the video's title
    :param format: format type for audio file, such as 'wav', 'mp3'. WAV is preferred.
    :return: path of audio
    """

    if not validators.url(url):
        return None

    os.makedirs(dst_dir, exist_ok=True)

    # download audio
    path = os.path.join(dst_dir, f"audio.{format}")
    if os.path.exists(path):
        os.remove(path)
    os.system(f"yt-dlp -f 'ba' -x --audio-format {format} {url}  -o {path} --quiet")

    return path


def speech_to_text(audio, beam_size=5, best_of=5, language='en'):
    """ ASR inference with Whisper

    :param audio: filepath
    :param beam_size: beam search parameter
    :param best_of: number of best results
    :param language: Currently English only
    :return: transcription
    """

    result = asr_model.transcribe(audio, language=language, beam_size=beam_size, best_of=best_of, fp16=False)

    return result['text']


def text_summarization(text):
    return summarizer(text)


def wordcloud_func(text: str, out_path='data/wordcloud_output.png'):
    """ generate wordcloud based on text

    :param text: transcription
    :param out_path: filepath
    :return: filepath
    """

    if len(text) == 0:
        return None

    stopwords = STOPWORDS

    wc = WordCloud(
        background_color='white',
        stopwords=stopwords,
        height=600,
        width=600
    )

    wc.generate(text)
    wc.to_file(out_path)

    return out_path


def normalize_dollars(text):
    """ text normalization for '$'

    :param text:
    :return:
    """

    def expand_dollars(m):
        match = m.group(1)
        parts = match.split(' ')
        parts.append('dollars')
        return ' '.join(parts)

    units = ['hundred', 'thousand', 'million', 'billion', 'trillion']
    _dollars_re = re.compile(fr"\$([0-9\.\,]*[0-9]+ (?:{'|'.join(units)}))")

    return re.sub(_dollars_re, expand_dollars, text)


def text_to_speech(text: str, out_path="data/short_speech.wav"):

    # espnet tts model process '$1.4 trillion' as 'one point four dollar trillion'
    # use this function to fix this issue
    text = normalize_dollars(text)

    output = tts_model(text)
    write(out_path, 22050, output['wav'].numpy())

    return out_path


demo = gr.Blocks(css=demo_css, title="Speech Summarization")

demo.encrypt = False

with demo:
    # demo description
    gr.Markdown("""
    ## Speech Summarization with Whisper
    This space is intended to summarize a speech, a short one or long one, to save us sometime 
    (runs faster with GPU inference). Check the example links provided below:
    [3 mins speech](https://www.youtube.com/watch?v=DuX4K4eeTz8), 
    [13 mins speech](https://www.youtube.com/watch?v=nepOSEGHHCQ)
    
    1. Type in a youtube URL or upload an audio file
    2. Generate transcription with Whisper (English Only)
    3. Summarize the transcribed speech
    4. Generate summary speech with the ESPNet model
    """)

    # data preparation
    with gr.Row():
        with gr.Column():
            url = gr.Textbox(label="URL", placeholder="video url")

            url_btn = gr.Button("clear")
            url_btn.click(lambda x: '', inputs=url, outputs=url)

        speech = gr.Audio(label="Speech", type="filepath")

        url.change(audio_from_url, inputs=url, outputs=speech)

    # ASR
    text = gr.Textbox(label="Transcription", placeholder="transcription")

    with gr.Row():
        model_options = gr.Dropdown(['Tiny.en', 'Base.en'], value=default_model, label="models")
        model_options.change(load_model, inputs=model_options, outputs=model_options)

        beam_size_slider = gr.Slider(1, 10, value=5, step=1, label="param: beam_size")
        best_of_slider = gr.Slider(1, 10, value=5, step=1, label="param: best_of")

    with gr.Row():
        asr_clr_btn = gr.Button("clear")
        asr_clr_btn.click(lambda x: '', inputs=text, outputs=text)
        asr_btn = gr.Button("Recognize Speech")
        asr_btn.click(speech_to_text, inputs=[speech, beam_size_slider, best_of_slider], outputs=text)

    # summarization
    summary = gr.Textbox(label="Summarization")

    with gr.Row():
        sum_clr_btn = gr.Button("clear")
        sum_clr_btn.click(lambda x: '', inputs=summary, outputs=summary)
        sum_btn = gr.Button("Summarize")
        sum_btn.click(text_summarization, inputs=text, outputs=summary)

    with gr.Row():
        # wordcloud
        image = gr.Image(label="wordcloud", show_label=False).style(height=400, width=400)
        with gr.Column():
            tts = gr.Audio(label="Short Speech", type="filepath")
            tts_btn = gr.Button("Read Summary")
            tts_btn.click(text_to_speech, inputs=summary, outputs=tts)

    text.change(wordcloud_func, inputs=text, outputs=image)

    examples = gr.Examples(examples=[
            "https://www.youtube.com/watch?v=DuX4K4eeTz8",
            "https://www.youtube.com/watch?v=nepOSEGHHCQ"
        ],
        inputs=url, outputs=text,
        fn=lambda x: speech_to_text(audio_from_url(x)),
        cache_examples=True
    )

    gr.HTML(footer_html)


if __name__ == '__main__':
    demo.launch()
