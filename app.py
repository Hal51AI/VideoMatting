import torch
import gradio as gr


def get_free_memory_gb():
    gpu_index = torch.cuda.current_device()
    # Get the GPU's properties
    gpu_properties = torch.cuda.get_device_properties(gpu_index)

    # Get the total and allocated memory
    total_memory = gpu_properties.total_memory
    allocated_memory = torch.cuda.memory_allocated(gpu_index)

    # Calculate the free memory
    free_memory = total_memory - allocated_memory
    return free_memory / 1024**3


model = torch.hub.load("PeterL1n/RobustVideoMatting", "mobilenetv3")

if torch.cuda.is_available():
    free_memory = get_free_memory_gb()
    concurrency_count = int(free_memory // 7.4)
    model = model.cuda()
    print(f"Using GPU with concurrency: {concurrency_count}")
else:
    print("Using CPU")
    concurrency_count = 1

convert_video = torch.hub.load("PeterL1n/RobustVideoMatting", "converter")


def inference(video):
    convert_video(
        model,  # The loaded model, can be on any device (cpu or cuda).
        input_source=video,  # A video file or an image sequence directory.
        downsample_ratio=0.25,  # [Optional] If None, make downsampled max size be 512px.
        output_type="video",  # Choose "video" or "png_sequence"
        output_composition="com.mp4",  # File path if video; directory path if png sequence.
        output_alpha=None,  # [Optional] Output the raw alpha prediction.
        output_foreground=None,  # [Optional] Output the raw foreground prediction.
        output_video_mbps=4,  # Output video mbps. Not needed for png sequence.
        seq_chunk=12,  # Process n frames at once for better parallelism.
        num_workers=1,  # Only for image sequence input. Reader threads.
        progress=True,  # Print conversion progress.
    )
    return "com.mp4"


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

block.queue(api_open=False, max_size=5, concurrency_count=concurrency_count).launch()
