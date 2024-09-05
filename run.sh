#!/bin/bash
uvicorn subtitles:app &>/dev/null &
sleep 3
curl 'http://localhost:8000/loadModel?requestModel=tiny.en&useGpu=True' -o-
prevfile=0
ffmpeg -i https://screamrouter.netham45.org/stream/192.168.3.152/ -f segment -segment_time 1 -strftime 1 %H-%M-%S.mp3 -v verbose 2>&1 |
grep -Po --line-buffered "Opening '\K[^']*" |
while read file
do
  if ! [[ "$prevfile" == "0" ]]
  then
      output=$(clear;curl -sS "http://localhost:8000/processFile?fileName=$prevfile" -o- | sed 's/^"//g;s/"$//g')
      echo "$output"
  fi
  prevfile=$file
done
