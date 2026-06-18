from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import asyncio
import shutil
import os
import tempfile
from mutagen.mp3 import MP3
from mutagen.wave import WAVE
import mutagen
from faster_whisper import WhisperModel
import edge_tts
from pydantic import BaseModel

app = FastAPI(title="DeepLearn AI Audio API")

# Allow requests from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# စည်းမျဉ်းများ (Constraints)
# ---------------------------------------------------------
MAX_DURATION_SECONDS = 10 * 60  # ၁၀ မိနစ်
MAX_CONCURRENT_USERS = 10       # အများဆုံး ပြိုင်တူသုံးခွင့် ၁၀ ယောက်

# 10 ယောက်ပြိုင်တူသုံးရန် ကန့်သတ်ထားသော Queue စနစ် (Semaphore)
process_semaphore = asyncio.Semaphore(MAX_CONCURRENT_USERS)

# ---------------------------------------------------------
# AI Models (Global Load)
# ---------------------------------------------------------
# faster-whisper သည် အပေါ့ပါးဆုံးနှင့် အမြန်ဆုံးဖြစ်သည်။ "tiny" model ကိုသုံးထားသည်။
print("Loading Whisper Model...")
whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")
print("Model Loaded!")

# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
def format_timestamp(seconds: float) -> str:
    """စက္ကန့်များကို SRT format (00:00:00,000) သို့ ပြောင်းပေးရန်"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def check_audio_duration(file_path: str):
    """Audio ဖိုင်၏ အရှည် (၁၀ မိနစ်ကျော်မကျော်) စစ်ဆေးရန်"""
    try:
        audio = mutagen.File(file_path)
        if audio is not None and audio.info is not None:
            if audio.info.length > MAX_DURATION_SECONDS:
                raise ValueError("ဖိုင်အရှည်သည် ၁၀ မိနစ်ထက် မကျော်ရပါ။ (Max 10 minutes exceeded)")
    except Exception as e:
        if "၁၀ မိနစ်" in str(e):
            raise e
        # ဖိုင်အမျိုးအစားစစ်မရပါက ကျော်သွားမည် (Whisper က ဆက်လက်လုပ်ဆောင်ပေးမည်)
        pass

def cleanup_file(path: str):
    """လုပ်ဆောင်ပြီးပါက နေရာမယူအောင် ယာယီဖိုင်များဖျက်ရန်"""
    if os.path.exists(path):
        os.remove(path)

# ---------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------

@app.get("/")
def read_root():
    return {"message": "DeepLearn AI Audio Studio API is running."}

@app.post("/api/transcribe")
async def transcribe_audio(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Audio မှ SRT စာသားထုတ်ပေးသည့် အပိုင်း"""
    
    # ယာယီဖိုင်သိမ်းဆည်းခြင်း
    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, file.filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        # ၁၀ မိနစ် စစ်ဆေးခြင်း
        check_audio_duration(file_path)
        
        # Queue ထဲဝင်ရန်စောင့်ခြင်း (လူ ၁၀ ယောက်ပြည့်နေပါက ဤနေရာတွင် တန်းစီစောင့်ရမည်)
        async with process_semaphore:
            # CPU Intensive ဖြစ်သောကြောင့် asyncio thread ဖြင့် Run ပါသည်
            segments, info = await asyncio.to_thread(whisper_model.transcribe, file_path, beam_size=5)
            
            srt_content = ""
            segment_id = 1
            for segment in segments:
                start_time = format_timestamp(segment.start)
                end_time = format_timestamp(segment.end)
                srt_content += f"{segment_id}\n"
                srt_content += f"{start_time} --> {end_time}\n"
                srt_content += f"{segment.text.strip()}\n\n"
                segment_id += 1
                
        # လုပ်ဆောင်ပြီးပါက ဖိုင်ကို ပြန်ဖျက်ရန်
        background_tasks.add_task(cleanup_file, file_path)
        
        return JSONResponse(content={"status": "success", "srt_content": srt_content})
        
    except ValueError as ve:
        cleanup_file(file_path)
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        cleanup_file(file_path)
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")


class TTSRequest(BaseModel):
    text: str
    voice: str = "my-MM-NilarNeural" # Default Nilar voice

@app.post("/api/tts")
async def generate_voice(request: TTSRequest, background_tasks: BackgroundTasks):
    """Microsoft Edge TTS အသုံးပြု၍ အသံထုတ်ပေးသည့် အပိုင်း"""
    
    if not request.text:
        raise HTTPException(status_code=400, detail="စာသား မပါဝင်ပါ။")
        
    temp_audio_path = tempfile.mktemp(suffix=".mp3")
    
    try:
        # Queue ထဲဝင်ရန်စောင့်ခြင်း
        async with process_semaphore:
            communicate = edge_tts.Communicate(request.text, request.voice)
            await communicate.save(temp_audio_path)
            
        # ဖိုင်ပို့ပြီးပါက ပြန်ဖျက်ရန်
        background_tasks.add_task(cleanup_file, temp_audio_path)
        
        return FileResponse(temp_audio_path, media_type="audio/mpeg", filename="voiceover.mp3")
        
    except Exception as e:
        cleanup_file(temp_audio_path)
        raise HTTPException(status_code=500, detail=f"TTS Error: {str(e)}")

# To run: uvicorn main:app --host 0.0.0.0 --port 8000