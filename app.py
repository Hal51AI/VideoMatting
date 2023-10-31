import av
import torch
import tempfile
import shutil
import atexit
import gradio as gr


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


def inference(video):
    if get_video_length_av(video) > 30:
        raise gr.Error("Length of video cannot be over 30 seconds")
    if get_video_dimensions(video) > (1920, 1920):
        raise gr.Error("Video resolution must not be higher than 1920x1080")

    temp_dir = tempfile.mkdtemp()
    temp_directories.append(temp_dir)

    convert_video(
        model,  # The loaded model, can be on any device (cpu or cuda).
        input_source=video,  # A video file or an image sequence directory.
        downsample_ratio=0.25,  # [Optional] If None, make downsampled max size be 512px.
        output_type="video",  # Choose "video" or "png_sequence"
        output_composition=(
            temp_dir + "/matted_video.mp4"
        ),  # File path if video; directory path if png sequence.
        output_alpha=None,  # [Optional] Output the raw alpha prediction.
        output_foreground=None,  # [Optional] Output the raw foreground prediction.
        output_video_mbps=4,  # Output video mbps. Not needed for png sequence.
        seq_chunk=12,  # Process n frames at once for better parallelism.
        num_workers=1,  # Only for image sequence input. Reader threads.
        progress=True,  # Print conversion progress.
    )
    return temp_dir + "/matted_video.mp4"


if __name__ == "__main__":
    temp_directories = []
    atexit.register(cleanup_temp_directories)

    model = torch.hub.load("PeterL1n/RobustVideoMatting", "mobilenetv3")
    convert_video = torch.hub.load("PeterL1n/RobustVideoMatting", "converter")

    if torch.cuda.is_available():
        free_memory = get_free_memory_gb()
        concurrency_count = int(free_memory // 7)
        print(f"Using GPU with concurrency: {concurrency_count}")
        print(f"Available video memory: {free_memory} GB")
        model = model.cuda()
    else:
        print("Using CPU")
        concurrency_count = 1

    with gr.Blocks(title="Robust Video Matting") as block:
        gr.Markdown("# Robust Video Matting")
        gr.Markdown(
            "Gradio demo for Robust Video Matting. To use it, simply upload your video, or click one of the examples to load them. Read more at the links below."
        )
        with gr.Row():
            inp = gr.Video(label="Input Video")
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

    block.queue(
        api_open=False, max_size=5, concurrency_count=concurrency_count
    ).launch()
