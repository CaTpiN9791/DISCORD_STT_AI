import asyncio
import os
import tempfile
import whisper
import requests
import discord
from discord.ext import commands
import wave
import time
import audioop
import queue
import torch
import torchaudio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from dotenv import load_dotenv
load_dotenv()

BOT_NAME = os.getenv("BOT_NAME")
TOKEN = os.getenv("DISCORD_TOKEN")
WHISPER_MODEL = os.getenv("WHISPER_MODEL")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")
CHANNAL_ID = os.getenv("CHANNAL_ID")
VOICE_DIR = os.getenv("VOICE_DIR")

device = "cuda" if torch.cuda.is_available() else "cpu"
config_path = os.path.join(VOICE_DIR, "config.json")
path = os.path.join(VOICE_DIR, "model.pth")
reference_wav = r"reference.wav"
output_wav = r"output.wav"
torch.load(path, map_location=device, weights_only=True)
config = XttsConfig()
config.load_json(config_path)
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir=VOICE_DIR, checkpoint_path=path)
model.to(device)
language = "en"

intents = discord.Intents.default()
intents.message_content = True
intents.voice_states = True
vad_instance = None

bot = commands.Bot(command_prefix='!', intents=intents)
torch.set_grad_enabled(False)
whisper_model = whisper.load_model(WHISPER_MODEL, device=device)
OLLAMA_API_URL = "http://localhost:11434/api/generate"
temp_dir = tempfile.mkdtemp()
print(f"Temporary directory created: {temp_dir}")

SILENCE_THRESHOLD = 200
SILENCE_DURATION = 0.5
MIN_SPEECH_DURATION = 0.2
FRAME_DURATION = 0.02

voice_processing_queue = queue.Queue()

class VoiceActivityDetector(discord.sinks.WaveSink):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.last_audio_time = {}
        self.is_speaking = {}
        self.speech_start_time = {}
        self.silence_start_time = {}
        self.buffer_data = {}
        self.recent_rms_values = {}
        self.consecutive_silence_frames = {}
        
    def write(self, data, user_id):
        user = bot.fetch_user(user_id)
        super().write(data, user_id)
        current_time = time.time()
        if user_id not in self.buffer_data:
            self.buffer_data[user_id] = bytearray()
            self.is_speaking[user_id] = False
            self.speech_start_time[user_id] = 0
            self.silence_start_time[user_id] = current_time
            self.recent_rms_values[user_id] = []
            self.consecutive_silence_frames[user_id] = 0
        self.buffer_data[user_id].extend(data)

        try:
            rms = audioop.rms(data, 2)
            self.recent_rms_values[user_id].append(rms)
            if len(self.recent_rms_values[user_id]) > 5:
                self.recent_rms_values[user_id].pop(0)
            avg_rms = sum(self.recent_rms_values[user_id]) / len(self.recent_rms_values[user_id])

        except Exception as e:
            print(f"RMS error: {e}")
            rms = 0
            avg_rms = 0
        self.last_audio_time[user_id] = current_time
        if avg_rms > SILENCE_THRESHOLD:
            self.consecutive_silence_frames[user_id] = 0
            if not self.is_speaking[user_id]:
                self.is_speaking[user_id] = True
                self.speech_start_time[user_id] = current_time
                print(f"User {user} voice start detected (RMS: {avg_rms:.2f})")
        else:
            self.consecutive_silence_frames[user_id] += 1
            silence_duration = self.consecutive_silence_frames[user_id] * FRAME_DURATION
            if self.is_speaking[user_id] and silence_duration >= SILENCE_DURATION:
                speech_duration = current_time - self.speech_start_time[user_id]
                print(f"User {user} voice termination detected (Voice duration: {speech_duration:.2f}sec)")
                if speech_duration > MIN_SPEECH_DURATION:
                    buffer_copy = bytes(self.buffer_data[user_id])
                    self.buffer_data[user_id] = bytearray()
                    voice_processing_queue.put((user_id, buffer_copy, self.ctx))
                    self.is_speaking[user_id] = False
                    self.consecutive_silence_frames[user_id] = 0
                    self.reset_user_state(user_id)
    
    def reset_user_state(self, user_id):
        self.buffer_data[user_id] = bytearray()
        self.is_speaking[user_id] = False
        self.speech_start_time[user_id] = 0
        self.silence_start_time[user_id] = time.time()
        self.recent_rms_values[user_id] = []
        self.consecutive_silence_frames[user_id] = 0
        self.last_audio_time[user_id] = time.time()
    
async def process_voice_chunk(user_id, buffer_data, ctx):
    try:
        file_path = os.path.join(temp_dir, f"{user_id}_chunk.wav")
        with wave.open(file_path, 'wb') as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(48000)
            wf.writeframes(buffer_data)
        user = await bot.fetch_user(user_id)
        if user.bot:
            return
        transcript = transcribe_audio(file_path)
        if not transcript.strip():
            return
        print(f"Recognized text: ({user.name}) {transcript}")
        await channel.send(f"{user.name}: {transcript}")
        response = process_with_ollama(transcript)
        if ctx.voice_client and ctx.voice_client.is_connected():
            await play_tts_response(ctx, response)
    except Exception as e:
        print(f"Error Occured while processing voice chunk: {e}")

async def voice_processing_task():
    while True:
        try:
            if not voice_processing_queue.empty():
                user_id, buffer_data, ctx = voice_processing_queue.get_nowait()
                await process_voice_chunk(user_id, buffer_data, ctx)
                voice_processing_queue.task_done()
            await asyncio.sleep(0.1)
        except Exception as e:
            print(f"Error while processing voice information: {e}")
            await asyncio.sleep(1)

@bot.event
async def on_ready():
    global channel
    print(f"{bot.user} is ready!")
    channel = bot.get_channel(CHANNAL_ID)
    bot.loop.create_task(voice_processing_task())
    bot.loop.create_task(vad_timeout_checker())
    await channel.send("Bot Turned On!")

@bot.command()
async def join(ctx):
    if ctx.author.voice is None:
        await ctx.send("Connect to a voice channel first!")
        return

    channel = ctx.author.voice.channel
    try:
        if ctx.voice_client is not None:
            await ctx.voice_client.move_to(channel)
        else:
            await channel.connect()
        await ctx.send(f"Connected to {channel}!")
        await start_vad_recording(ctx)
    except Exception as e:
        await ctx.send(f"Error Occured: {e}")

@bot.command()
async def leave(ctx):
    if ctx.voice_client is not None:
        if ctx.voice_client.recording:
            ctx.voice_client.stop_recording()
        await ctx.voice_client.disconnect()
        await ctx.send("Disconnected from the voice channel.")
    else:
        await ctx.send("Not connected to any voice channel.")

async def start_vad_recording(ctx):
    global vad_instance
    try:
        vad_sink = VoiceActivityDetector(ctx)
        vad_instance = vad_sink
        ctx.voice_client.start_recording(
            vad_sink,
            ctx
        )
        await ctx.send("Starting voice record. If you speak and remain silent for a while, it will be processed automatically.")
        while ctx.voice_client and ctx.voice_client.is_connected():
            await asyncio.sleep(1)
            
    except Exception as e:
        print(f"VAD error: {e}")
        await ctx.send(f"Error occured while starting record: {e}")

def transcribe_audio(file_path):
    try:
        result = whisper_model.transcribe(file_path)
        return result["text"]
    except Exception as e:
        print(f"Error occured while recognizing voice: {e}")
        return

def process_with_ollama(text):
    try:
        payload = {
            "model": f"{OLLAMA_MODEL}",
            "prompt": text,
            "stream": False
        }
        
        response = requests.post(OLLAMA_API_URL, json=payload)
        if response.status_code == 200:
            result = response.json()
            return result.get('response', 'request failed')
        else:
            print(f"Ollama API error: {response.status_code}, {response.text}")
            return "Failed to get response from Ollama."
    except Exception as e:
        print(f"Ollama processing error: {e}")
        return "Error occurred while processing with Ollama."

async def play_tts_response(ctx, text):
    output = model.synthesize(
    config=config,
    text=text,
    speaker_wav=reference_wav,
    language=language,
    temperature=0.75,
    length_penalty=1.0,
    repetition_penalty=5.0,
    top_k=50,
    top_p=0.85,
    speed=1.0,
    )
    torchaudio.save(output_wav, torch.tensor(output["wav"]).unsqueeze(0), sample_rate=config.audio["sample_rate"])
    ctx.voice_client.play(discord.FFmpegPCMAudio(output_wav), after= print(f"{bot}: {text}"))
    print(f"✅ {bot}:", output_wav)
    await channel.send(f"Muffin: {text}")

async def vad_timeout_checker():
    global vad_instance
    TIMEOUT_DURATION = 0.8
    while True:
        await asyncio.sleep(1)
        if vad_instance is None:
            continue
        current_time = time.time()
        for user_id in list(vad_instance.last_audio_time.keys()):
            last_time = vad_instance.last_audio_time.get(user_id, current_time)
            if vad_instance.is_speaking.get(user_id, False):
                if current_time - last_time > TIMEOUT_DURATION:
                    print(f"[Timeout detected] User {user_id} is silent for {TIMEOUT_DURATION}seconds! Processing voice data information.")
                    await channel.send(f"[Timeout detected] User {user_id} is silent for {TIMEOUT_DURATION} seconds! Assume to end the conversation.")
                    buffer_copy = bytes(vad_instance.buffer_data.get(user_id, b""))
                    if len(buffer_copy) > 0:
                        voice_processing_queue.put((user_id, buffer_copy, vad_instance.ctx))
                    vad_instance.reset_user_state(user_id)
        
if __name__ == "__main__":
    bot.run(TOKEN)