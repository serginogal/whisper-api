import os
import ffmpeg
import whisper
from yt_dlp import YoutubeDL
from pydantic import BaseModel
from fastapi import FastAPI, Request
from starlette.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi.exception_handlers import RequestValidationError

app = FastAPI()

model = whisper.load_model("base")

# MODEL_NOT_FOUND_MESSAGE = (
#     "Model not found. \n"
#     "Please provide a model parameter.\n"
#     "Supported models: tiny, base, small, medium, "
#     "large-v1, large-v2"
# )

MODEL_NOT_FOUND_MESSAGE = (
    "Youtube URL not found. \n"
    "Please provide a youtubeURL parameter."
)

INVALID_URL_MESSAGE = (
    "URL not found. \n"
    "Please provide a valid url parameter."
)

INVALID_YOUTUBE_URL_MESSAGE = (
    "Youtube URL not found. \n"
    "Please provide a valid youtubeURL parameter."
)


class UnicornException(Exception):
    def __init__(self, error_type: str):
        self.error_type = error_type


@app.exception_handler(UnicornException)
async def unicorn_exception_handler(request: Request, exc: UnicornException):
    error_type = exc.error_type
    print(error_type)
    if error_type == "MODEL_NOT_FOUND":
        return JSONResponse(
            status_code=400,
            content={
                "status": "ERROR",
                "code": "MODEL_NOT_FOUND",
                "message": MODEL_NOT_FOUND_MESSAGE
            },
        )
    if error_type == "INVALID_YOUTUBE_URL":
        return JSONResponse(
            status_code=400,
            content={
                "status": "ERROR",
                "code": "INVALID_YOUTUBE_URL",
                "message": INVALID_YOUTUBE_URL_MESSAGE
            },
        )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    error_json = jsonable_encoder(exc.errors())
    error_loc = error_json[0]["loc"]
    error_type = error_json[0]["type"]
    print(error_loc[0])
    if (
        len(error_loc) == 1 and
        error_loc[0] == "body" and
        error_type == "type_error.dict"
    ):
        return JSONResponse(
            status_code=400,
            content={
                "status": "ERROR",
                "code": "INVALID_REQUEST_BODY",
                "message": "Invalid request body"
            },
        )
    if (
        len(error_loc) > 1 and
        error_loc[0] == "body" and
        error_loc[1] == "model"
    ):
        return JSONResponse(
            status_code=400,
            content={
                "status": "ERROR",
                "code": "MODEL_NOT_FOUND",
                "message": MODEL_NOT_FOUND_MESSAGE
            },
        )
    if (
        len(error_loc) > 1 and
        error_loc[0] == "body" and
        error_loc[1] == "youtubeUrl"
    ):
        return JSONResponse(
            status_code=400,
            content={
                "status": "ERROR",
                "code": "MODEL_NOT_FOUND",
                "message": MODEL_NOT_FOUND_MESSAGE
            },
        )
    return JSONResponse(
        status_code=422,
        content={
            "status": "ERROR",
            "detail": error_json
        },
    )


@app.get("/")
async def root():
    return {
        "status": "OK",
        "message": "Welcome to Whisper API"
    }


@app.get("/whisper/")
async def whisper_root():
    return {
        "status": "OK",
        "message": "Welcome to Whisper API"
    }


# class TranscribeBody(BaseModel):
#     model: str


# @app.post("/whisper/transcribe/")
# def whisper_transcribe(body: TranscribeBody):
#     # return body
#     selected_model = body.model
#     if (
#         selected_model != "tiny" and
#         selected_model != "base" and
#         selected_model != "small" and
#         selected_model != "medium" and
#         selected_model != "large-v1" and
#         selected_model != "large-v2"
#     ):
#         raise UnicornException(error_type="MODEL_NOT_FOUND")
#     print(selected_model)
#     model = whisper.load_model(selected_model)
#     result = model.transcribe("./sample/sample1.mp3")
#     print(result["text"])
#     # model = whisper_ai.load_model(selected_model)
#     return result

def youtube_download(url: str):
    ydl_opts = {
        "format": "bestaudio/best",
        'outtmpl': 'tmp/%(id)s.%(ext)s',
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }],
    }
    with YoutubeDL(ydl_opts) as ydl:
        # ydl.download([url])
        info = ydl.extract_info(url, download=True)
        return ydl.sanitize_info(info)["requested_downloads"][0]


def parse_time(sec):
    milliseconds = int((sec % 1) * 1000)
    hours, _minutes = divmod(sec, 3600)
    minutes, seconds = divmod(_minutes, 60)
    return '{:02}:{:02}:{:02}.{:03}'.format(
        int(hours), int(minutes), int(seconds), int(milliseconds)
    )


def parse_transcript(segment):
    start_time = parse_time(segment["start"])
    end_time = parse_time(segment["end"])

    return f"[{start_time} - {end_time}] {segment['text']}"


class TranscribeBody(BaseModel):
    youtubeUrl: str = None
    isYoutube: bool = False
    url: str = None


@app.post("/whisper/transcribe/")
def whisper_transcribe(body: TranscribeBody):
    if body.isYoutube is not True and body.url is None:
        raise UnicornException(error_type="INVALID_URL")
    if body.isYoutube and body.youtubeUrl is None:
        raise UnicornException(error_type="INVALID_YOUTUBE_URL")

    # file = youtube_download(body.youtubeUrl)
    # file_path =  file["filepath"]
    file_path = body.url if body.isYoutube is not True else youtube_download(
        body.youtubeUrl)["filepath"]
    result = model.transcribe(
        file_path,
        word_timestamps=False,
        verbose=False
    )
    text = result["text"]
    segments = result["segments"]
    transcript = list(map(parse_transcript, segments))
    print(transcript)
    if body.isYoutube:
        os.remove(file_path)

    return {
        "status": "OK",
        "text": text,
        "transcript": transcript
    }
