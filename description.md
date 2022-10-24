# Video Speech Summarization

This project is to generate a summarization of a youtube video or a given speech. It includes two distinct models, the first is [Whisper](https://github.com/openai/whisper) for robust automatic speech recognition, the second one is [BART (large-sized) model](https://github.com/facebookresearch/fairseq/tree/main/examples/bart) for text summarization. 

This project enables me to get a quick grasp on news or speech in an automatic fashion.

1. Download the audio from YouTube in **.wav** format

2. Perform automatic speech recognition using Whisper model

3. Generate summarization on the speech transcription.

The graph below shows the summarization of this [Biden’s speech on official Afghanistan withdrawal, in 3 minutes](https://www.youtube.com/watch?v=DuX4K4eeTz8) video.

![biden speech summary](https://github.com/jiedong-yang/Speech-Summarization/blob/main/results/afgan-test.png)

Summary on [President Biden delivers remarks on his economic plan(13 mins)](https://www.youtube.com/watch?v=nepOSEGHHCQA)

![13 mins speech summary](https://github.com/jiedong-yang/Speech-Summarization/blob/main/results/economic-plan-test.png)
