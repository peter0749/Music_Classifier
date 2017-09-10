#!/bin/bash

if [ $# -lt 1 ] || [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
    echo "Usage: $0 youtube_url [time_offset(default: 5, in second)]"
  exit 0
fi

offset=5
if [ $# -gt 1 ]; then
    offset=$2
fi

youtube-dl $1 -f 140 --audio-format m4a -o ./tmp.m4a
python classifier.py ./tmp.m4a --ffmpeg --offset $offset
rm -f ./tmp.m4a