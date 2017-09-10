#!/bin/bash

if [ $# -lt 1 ] || [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
    echo "Usage: $0 youtube_url [time_offset(default: 5, in second)]"
    echo "Requirements:"
    echo "FFmpeg, youtube-dl and Python2.7"
    echo "Required Python packages are listed on requirements.txt"
  exit 0
fi

offset=5
if [ $# -gt 1 ]; then
    offset=$2
fi

>&2 echo "Getting audio from Youtube..."
youtube-dl $1 -f 140 --audio-format m4a -o ./tmp.m4a &> /dev/null
>&2 echo "Processing..."
echo "Audio from: $1"
python classifier.py ./tmp.m4a --ffmpeg --offset $offset 2> /dev/null
rm -f ./tmp.m4a
