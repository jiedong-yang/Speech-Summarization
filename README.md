# Video Speech Summarization

This little project is to generate a speech summarization on a youtube video. It concatenate two distinct models, one for speech recognition, one for summarization. 

1. Converting a youtube video into aduio clips, then perform speech recognition on each clip and add all transcriptions into one.

2. Feed the transcription to summarization model, generate a summary on whole speech.

3. If running the app encouters a memory crash problem, make the value of variable *sec_per_split* in **_split_audio_** function smaller.

The graph below shows the summarization of this [Biden’s speech on official Afghanistan withdrawal, in 3 minutes](https://www.youtube.com/watch?v=DuX4K4eeTz8) video.

![biden speech summary](https://github.com/RickestYang/Speech-Summarization/blob/main/results/Biden%20speech%20summary.png)

Summary on [President Joe Biden delivers remarks after Afghanistan withdrawal(26 mins)](https://www.youtube.com/watch?v=unBscCtq9xA)

![26mins speech summary](https://github.com/RickestYang/Speech-Summarization/blob/main/results/biden%2026%20mins%20speech%20summary.png)