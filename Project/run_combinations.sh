#!/bin/bash

for text_encoder in "lstm" "cnn_rnn" "transformer" "cnn"
do
for video_encoder in "lstm" "cnn_rnn" "transformer" "cnn"
do
for audio_encoder in "lstm" "cnn_rnn" "transformer" "cnn"
do
python3 run.py --text_encoder $text_encoder --video_encoder $video_encoder --audio_encoder $audio_encoder
done
done
done