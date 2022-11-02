# Video Speech Summarization

This project is to generate a summarization of a youtube video or a given speech. It includes three distinct models:
[Whisper](https://github.com/openai/whisper) for robust automatic speech recognition, 
[BART (large-sized) model](https://github.com/facebookresearch/fairseq/tree/main/examples/bart) for text summarization, 
[Conformer from ESPNet]("https://github.com/espnet/espnet_model_zoo) for LJSpeech TTS. 

This project enables me to get a quick grasp on news or speech in an automatic fashion.

1. Download the audio from YouTube in **.wav** format

2. Perform automatic speech recognition using Whisper model

3. Generate summarization on the speech transcription

4. Generate speech using Conformer TTS model

The graph below shows the summarization of this [Biden’s speech on official Afghanistan withdrawal, in 3 minutes](https://www.youtube.com/watch?v=DuX4K4eeTz8) video.

![biden speech summary](https://github.com/jiedong-yang/Speech-Summarization/blob/main/results/biden-afgan-test.png)

Summary on [President Biden delivers remarks on his economic plan(13 mins)](https://www.youtube.com/watch?v=nepOSEGHHCQA)

![13 mins speech summary](https://github.com/jiedong-yang/Speech-Summarization/blob/main/results/biden-eco-plan-test.png)
