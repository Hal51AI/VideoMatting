import atexit
import multiprocessing
import os
import shutil
import subprocess
import tempfile

import av
import gradio as gr
import torch

from convert import convert_video


def get_video_length_av(video_path):
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        if container.duration is not None:
            duration_in_seconds = float(container.duration) / av.time_base
        else:
            duration_in_seconds = stream.duration * stream.time_base

    return duration_in_seconds


def get_video_dimensions(video_path):
    with av.open(video_path) as container:
        video_stream = container.streams.video[0]
        width = video_stream.width
        height = video_stream.height

    return width, height


def get_free_memory_gb():
    gpu_index = torch.cuda.current_device()
    gpu_properties = torch.cuda.get_device_properties(gpu_index)

    total_memory = gpu_properties.total_memory
    allocated_memory = torch.cuda.memory_allocated(gpu_index)

    free_memory = total_memory - allocated_memory
    return free_memory / 1024**3


def cleanup_temp_directories():
    print("Deleting temporary files")
    for temp_dir in temp_directories:
        try:
            shutil.rmtree(temp_dir)
        except FileNotFoundError:
            print(f"Could not delete directory {temp_dir}")


def ffmpeg_remux_audio(source_video_path, dest_video_path, output_path):
    # Build the ffmpeg command to extract audio and remux into another video
    command = [
        "ffmpeg",
        "-i",
        dest_video_path,  # Input destination video file
        "-i",
        source_video_path,  # Input source video file (for the audio)
        "-c:v",
        "copy",  # Copy the video stream as is
        "-c:a",
        "copy",  # Copy the audio stream as is
        "-map",
        "0:v:0",  # Map the video stream from the destination file
        "-map",
        "1:a:0",  # Map the audio stream from the source file
        output_path,  # Specify the output file path
    ]

    try:
        # Run the ffmpeg command
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        # Handle errors during the subprocess execution
        print(f"An error occurred: {e}")
        return dest_video_path

    return output_path


def inference(video):
    if get_video_length_av(video) > 30:
        raise gr.Error("Length of video cannot be over 30 seconds")
    if get_video_dimensions(video) > (1920, 1920):
        raise gr.Error("Video resolution must not be higher than 1920x1080")

    temp_dir = tempfile.mkdtemp()
    temp_directories.append(temp_dir)

    output_composition = temp_dir + "/matted_video.mp4"
    convert_video(
        model,  # The loaded model, can be on any device (cpu or cuda).
        input_source=video,  # A video file or an image sequence directory.
        downsample_ratio=0.25,  # [Optional] If None, make downsampled max size be 512px.
        output_composition=output_composition,  # File path if video; directory path if png sequence.
        output_alpha=None,  # [Optional] Output the raw alpha prediction.
        output_foreground=None,  # [Optional] Output the raw foreground prediction.
        output_video_mbps=4,  # Output video mbps. Not needed for png sequence.
        seq_chunk=12,  # Process n frames at once for better parallelism.
        num_workers=1,  # Only for image sequence input. Reader threads.
        progress=True,  # Print conversion progress.
    )

    resulting_video = f"{temp_dir}/matted_{os.path.split(video)[1]}"

    return ffmpeg_remux_audio(video, output_composition, resulting_video)


if __name__ == "__main__":
    temp_directories = []
    atexit.register(cleanup_temp_directories)

    model = torch.hub.load(
        "PeterL1n/RobustVideoMatting", "mobilenetv3", trust_repo=True
    )

    if torch.cuda.is_available():
        free_memory = get_free_memory_gb()
        concurrency_count = int(free_memory // 7)
        print(f"Using GPU with concurrency: {concurrency_count}")
        print(f"Available video memory: {free_memory} GB")
        model = model.cuda()
    else:
        print("Using CPU")
        cpu_count = multiprocessing.cpu_count()
        concurrency_count = cpu_count // 8


    with gr.Blocks(title="Robust Video Matting") as block:
        gr.Markdown("# Robust Video Matting")
        gr.Markdown(
            "Gradio demo for Robust Video Matting. To use it, simply upload your video, or click one of the examples to load them. Read more at the links below."
        )
        with gr.Row():
            inp = gr.Video(label="Input Video", sources=["upload"], include_audio=True)
            out = gr.Video(label="Output Video")
        btn = gr.Button("Run")
        btn.click(inference, inputs=inp, outputs=out)

        gr.Examples(
            examples=[["example.mp4"]],
            inputs=[inp],
        )
        gr.HTML(
            "<p style='text-align: center'><a href='https://arxiv.org/abs/2108.11515'>Robust High-Resolution Video Matting with Temporal Guidance</a> | <a href='https://github.com/PeterL1n/RobustVideoMatting'>Github Repo</a></p>"
        )

    block.queue(api_open=False, max_size=5, concurrency_count=concurrency_count).launch(
        share=False
    )
