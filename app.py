import os
import whisper
import validators
import gradio as gr

from wordcloud import WordCloud, STOPWORDS

# load whisper model for ASR and BART for summarization
asr_model = whisper.load_model('base.en')
summarizer = gr.Interface.load("facebook/bart-large-cnn", src='huggingface')


def load_model(name: str):
    """

    :param name: model options, tiny or base only, for quick inference
    :return:
    """
    asr_model = whisper.load_model(f"{name.lower()}.en")
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
    :param language:
    :return:
    """

    result = asr_model.transcribe(audio, language=language, beam_size=beam_size, best_of=best_of, fp16=False)

    return result['text']


def text_summarization(text):
    return summarizer(text)


def wordcloud_func(text: str, out_path='wordcloud_output.png'):
    """ generate wordcloud based on text

    :param text:
    :param out_path:
    :return:
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


demo = gr.Blocks(title="Speech Summarization")

demo.encrypt = False

with demo:
    # demo description
    gr.Markdown("""
    ## Speech Summarization with Whisper
    This space is intended to summarize a speech, a short one or long one, to save us sometime 
    (runs faster with local GPU inference). 
    
    1. Type in a youtube URL or upload an audio file
    2. Generate transcription with Whisper (Currently English Only)
    3. Summarize the transcribed speech
    4. A little wordcloud for you as well
    """)

    # data preparation
    with gr.Row():
        with gr.Column():
            url = gr.Textbox(label="URL", placeholder="video url")

            url_btn = gr.Button("clear")
            url_btn.click(lambda x: '', inputs=url, outputs=url)

        speech = gr.Audio(label="Speech", type="filepath")

        url.change(audio_from_url, inputs=url, outputs=speech)

    examples = gr.Examples(examples=["https://www.youtube.com/watch?v=DuX4K4eeTz8",
                                     "https://www.youtube.com/watch?v=nepOSEGHHCQ"],
                           inputs=[url])

    # ASR
    text = gr.Textbox(label="Transcription", placeholder="transcription")

    with gr.Row():
        model_options = gr.Dropdown(['Tiny', 'Base'], value='Base', label="models")
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

    # wordcloud
    image = gr.Image(label="wordcloud", show_label=False).style(height=400, width=400)

    text.change(wordcloud_func, inputs=text, outputs=image)


if __name__ == '__main__':
    demo.launch()