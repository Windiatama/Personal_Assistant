import whisper

model = whisper.load_model("base")
result = model.transcribe("replicate-prediction-1tmwsj9ehhrgj0cfhbft0zthjw.mp3")
print(result["text"])