# Image interpolation
```
python3 inference_img.py --img img0.png img1.png --exp=4
python3 inference_img.py --img ../examples/input_0.png ../examples/input_1.png --exp=4
```

(to read video from pngs, like input/0.png ... input/612.png, ensure that the png names are numbers)
```
python3 inference_video.py --exp=2 --video=video.mp4 --fps=60
```

# Gif
```
ffmpeg -i img%d.png output.gif
```