import sys
import os
import time
import asyncio
sys.path.append(os.path.join(os.path.dirname(__file__), "OpenVoice-main"))
sys.path.append(os.path.join(os.path.dirname(__file__), "pyvts-main"))
import argparse
import wave
import pyaudio
import torch
import whisper
from openvoice import se_extractor
from openvoice.api import BaseSpeakerTTS, ToneColorConverter
from openai import OpenAI

import asyncio
import pyvts


print('Hello Tama')

# ANSI escape code for colors
PINK = '\033[95m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
NEON_GREEN = '\033[92m'
RESET_COLOR = '\033[0m'

# init vts object
plugin_info = {
    "plugin_name": "Speaking",
    "developer": "Tama",
    "authentication_token_path": "./token.txt",
}
vts = pyvts.vts(plugin_info=plugin_info)

async def main():

    # Connect
    await vts.connect()

    # Authenticate
    await vts.request_authenticate_token()  # get token
    await vts.request_authenticate()  # use token

    # Custom new parameter named "start_parameter"
    new_parameter_name = "Zeta_speaking"
    recv_new_parameter = await vts.request(
        vts.vts_request.requestCustomParameter(new_parameter_name)
    )  # request custom parameter
    print("received msg after adding the parameter:")
    await mouth_stop_speaking()


    # task_mouth = asyncio.create_task(mouth_start_speaking())
    task_chatbot = asyncio.create_task(user_chatbot_conversation())

    # print("Tama user_chatbot_conversation")
    # await user_chatbot_conversation()  # Start the conversation

    # Wait for both tasks to complete
    while True:
        try:
            # await asyncio.gather(task_mouth)
            await asyncio.gather(task_chatbot)
        except asyncio.CancelledError:
            continue

async def mouth_start_speaking():
    while True:
        asyncio.sleep(0.2)
        try:
            print("Tama mouth_start_speaking coroutine was running")
            await vts.request(
                vts.vts_request.requestSetParameterValue('Zeta_speaking', 1)
            )  # set custom tracking parameter
            await asyncio.sleep(0.2)  # simulate some work with non-blocking sleep
            await vts.request(
                vts.vts_request.requestSetParameterValue('Zeta_speaking', 0)
            )  # set custom tracking parameter
            await asyncio.sleep(0.2)  # simulate some work with non-blocking sleep
        except asyncio.CancelledError:
            print("Tama mouth_start_speaking coroutine was cancelled")


async def mouth_stop_speaking():
    set_tracking_data = await vts.request(
        vts.vts_request.requestSetParameterValue('Zeta_speaking', 0)
    )  # set custom tracking parameter


async def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()


# Initialize the OpenAI client with the API
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

# Define the name of the Log file
chat_log_filename = "chatbot_conversation_log.txt"


# function to play audio using PyAudio
async def play_audio(file_path):

    # Open the audio file
    wf = wave.open(file_path, 'rb')

    # Create a PyAudio instance
    p = pyaudio.PyAudio()

    # Open a stream to play audio
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)

    # Read and play audio data
    data = wf.readframes(1024)
    while data:
        stream.write(data)
        data = wf.readframes(1024)

    # Stop and close the stream an PyAudio instance
    stream.stop_stream()
    stream.close()
    p.terminate()


# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--share", action='store_true', default=False, help="make link public")
args = parser.parse_args()

# Model and device setup
en_ckpt_base = 'checkpoints/base_speakers/EN'
ckpt_converter = 'checkpoints/converter'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
output_dir = 'outputs'
os.makedirs(output_dir, exist_ok=True)

# Load models
en_base_speaker_tts = BaseSpeakerTTS(f'OpenVoice-main/{en_ckpt_base}/config.json', device=device)
en_base_speaker_tts.load_ckpt(f'OpenVoice-main/{en_ckpt_base}/checkpoint.pth')
tone_color_converter = ToneColorConverter(f'OpenVoice-main/{ckpt_converter}/config.json', device=device)
tone_color_converter.load_ckpt(f'OpenVoice-main/{ckpt_converter}/checkpoint.pth')

# Load speaker embedding for English
en_source_default_se = torch.load(f'OpenVoice-main/{en_ckpt_base}/en_default_se.pth').to(device)
en_source_style_se = torch.load(f'OpenVoice-main/{en_ckpt_base}/en_style_se.pth').to(device)


# Main processing function
async def process_and_play(prompt, style, audio_file_pth):
    tts_model = en_base_speaker_tts
    source_se = en_source_default_se if style == 'defaul' else en_source_style_se

    speaker_wav = audio_file_pth

    # Process text and generate audio
    try:
        target_se, audio_name = se_extractor.get_se(speaker_wav, tone_color_converter, target_dir='process', vad=True)

        src_path = f'{output_dir}/tmp.wav'
        tts_model.tts(prompt, src_path, speaker=style, language='English')

        save_path = f'{output_dir}/output.wav'
        # Run the tone color converter
        encode_message = "@MyShell"
        tone_color_converter.convert(audio_src_path=src_path, src_se=source_se, tgt_se=target_se, output_path=save_path, message=encode_message)

        print("Audio generated successfully.")
        await play_audio(save_path)
    except Exception as e:
        print(f"Error during audio generation: {e}")


# Send the query to the OpenAI API with streaming enabled
async def chatgpt_streamed(user_input, system_message, conversation_history, bot_name):
    streamed_completion = client.chat.completions.create(
        model="lmstudio-ai/gemma-2b-it-GGUF",
        messages=[
            {"role": "system", "content": system_message}
        ] + conversation_history + [{"role": "user", "content": user_input}],
        stream=True  # Enable streaming
    )

    temperature=1

    # Initialize variables to hold the streamed response and the current line buffer
    full_response = ""
    line_buffer = ""

    # Open the log file in append mode
    with open(chat_log_filename, "a") as log_file:
        # Iterate over the streamed completion chunks
        for chunk in streamed_completion:
            # extract the delta content from each chunk
            delta_content = chunk.choices[0].delta.content

            # If delta content is not None, process it
            if delta_content is not None:
                # Add the delta content ot the line buffer
                line_buffer += delta_content

                # If a new line character is found, print the line in yellow and clear the buffer
                if '\n' in line_buffer:
                    lines = line_buffer.split('\n')
                    for line in lines[:-1]: # Print all but the last line (which might be incomplete)
                        print(NEON_GREEN + line + RESET_COLOR)
                        full_response += line + '\n'
                    line_buffer = lines[-1] # Keep the last line in the buffer

        # Print any remaining content in the buffer in yellow
        if line_buffer:
            print(NEON_GREEN + line_buffer + RESET_COLOR)
            full_response += line_buffer

    # Return the assembled full full_response
    return full_response


async def transcribe_with_whisper(audio_file_path):
    # Load the model
    model = whisper.load_model("base.en") # You can choose different model size like 'tiny', 'base', 'small', 'medium'. 'large'

    options = whisper.DecodingOptions(language='en', fp16=False)

    # Transcribe the audio
    result = model.transcribe(audio_file_path, fp16=False, language='English')
    print("transcribed")
    return result["text"]


# Function to record audio from the microphone and save to a file
async def record_audio(file_path):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
    frames = []
    
    print("Recording...")
    
    try:
        while True:
            data = stream.read(1024)
            frames.append(data)
    except KeyboardInterrupt:
        print('Interrupted')
        pass
    
    print("Recording stopped.")
    
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    wf = wave.open(file_path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(16000)
    wf.writeframes(b''.join(frames))
    wf.close()


#New function to handle a conversation with a user
async def user_chatbot_conversation():
    try:
        conversation_history = []
        system_message = await open_file("chatbot1.txt")
        while True:
            await asyncio.sleep(0.5)
            print("TAMAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
            audio_file = "temp_recording.wav"
            await record_audio(audio_file)
            user_input = await transcribe_with_whisper(audio_file)
            os.remove(audio_file) # Clean up the temporary audio file
            if user_input.lower() == "exit":  # Say 'exit to end the conversation
                print("chatbot exit")
                break

            print(CYAN + "You: ", user_input + RESET_COLOR)
            conversation_history.append({"role": "user", "content": user_input})
            print(PINK + "Zeta:" + RESET_COLOR)
            chatbot_response = await chatgpt_streamed(user_input, system_message, conversation_history, "Chatbot")
            conversation_history.append({"role": "assistant", "content": chatbot_response})

            prompt2 = chatbot_response
            style = "default"
            audio_file_pth2 = "Zeta.mp3"
            await process_and_play(prompt2, style, audio_file_pth2)

            if len(conversation_history) > 20:
                conversation_history = conversation_history[-20:]
    except asyncio.CancelledError:
        print("user_chatbot_conversation coroutine was cancelled")
        # Perform any cleanup if needed


if __name__ == "__main__":
    asyncio.run(main())
