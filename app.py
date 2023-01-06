import os
os.system('pip install gradio --upgrade')
os.system('pip freeze')
import torch
import gradio as gr
model = torch.hub.load("PeterL1n/RobustVideoMatting", "mobilenetv3") # or "resnet50"

convert_video = torch.hub.load("PeterL1n/RobustVideoMatting", "converter")

def inference(video):
  convert_video(
      model,                           # The loaded model, can be on any device (cpu or cuda).
      input_source=video,        # A video file or an image sequence directory.
      input_resize=(400, 400),       # [Optional] Resize the input (also the output).
      downsample_ratio=0.25,           # [Optional] If None, make downsampled max size be 512px.
      output_type='video',             # Choose "video" or "png_sequence"
      output_composition='com.mp4',    # File path if video; directory path if png sequence.
      output_alpha= None,          # [Optional] Output the raw alpha prediction.
      output_foreground= None,     # [Optional] Output the raw foreground prediction.
      output_video_mbps=4,             # Output video mbps. Not needed for png sequence.
      seq_chunk=7,                    # Process n frames at once for better parallelism.
      num_workers=1,                   # Only for image sequence input. Reader threads.
      progress=True                    # Print conversion progress.
  )
  return 'com.mp4'
  
title = "Robust Video Matting"
description = "Gradio demo for Robust Video Matting. To use it, simply upload your video, or click one of the examples to load them. Read more at the links below."

article = "<p style='text-align: center'><a href='https://arxiv.org/abs/2108.11515'>Robust High-Resolution Video Matting with Temporal Guidance</a> | <a href='https://github.com/PeterL1n/RobustVideoMatting'>Github Repo</a></p>"

examples = [['pexels-darina-belonogova-7539228.mp4']]
gr.Interface(
    inference, 
    gr.inputs.Video(label="Input"), 
    gr.outputs.Video(label="Output"),
    title=title,
    description=description,
    article=article,
    enable_queue=True,
    examples=examples
    ).launch(debug=True)