import gradio as gr
import soundfile as sf
import os

from s2s import generate


def get_tmp_path(file_name: str):
    temp_dir = "tempor"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    return os.path.join(temp_dir, file_name)


def process_audio(audio_file):
    result_audio_arr, sample_rate = generate(audio_file, lambda msg: gr.Info(msg))

    gr.Info("Writing response")
    target_path = get_tmp_path("processed_audio.wav")
    sf.write(target_path, result_audio_arr, sample_rate)

    return target_path


demo = gr.Interface(
    fn=process_audio,
    inputs=[gr.Audio(label="User Input", type="filepath", format="wav")],
    outputs=[gr.Audio(label="Response")],
    title="Speech to Speech with Mini Omni",
    description='<h2 style="text-align:center;">Ask anything in English and get the audio response!</h2>',
    article="Generation will take a while. Be patient! :)</br>Wait for a few seconds before your recorded audio is fully uploaded.</br>Retry if you encounter any error.",
)

demo.launch()
