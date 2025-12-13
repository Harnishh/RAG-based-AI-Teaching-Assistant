import whisper

model =  whisper.load_model("large-v2" , device = "cuda")

result = model.transcribe(audio = "audios/12_exercise-1-pure-html-media-player.mp3" ,  task = "translate")

print(result["text"])