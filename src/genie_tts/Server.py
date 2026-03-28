import asyncio
import logging
import os
import threading
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional, Callable, Union, Dict, Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .Audio.ReferenceAudio import ReferenceAudio
from .Core.TTSPlayer import tts_player
from .ModelManager import model_manager
from .Utils.Shared import context
from .Utils.Language import normalize_language

logger = logging.getLogger(__name__)

@dataclass
class ServerRuntimeConfig:
    single_process: bool = False
    max_concurrency: int = 1
    queue_maxsize: int = 0
    active_requests: int = 0
    waiting_requests: int = 0
    condition: threading.Condition = field(default_factory=threading.Condition)


_reference_audios: Dict[str, dict] = {}
SUPPORTED_AUDIO_EXTS = {'.wav', '.flac', '.ogg', '.aiff', '.aif'}
_server_runtime = ServerRuntimeConfig()

app = FastAPI()


class CharacterPayload(BaseModel):
    character_name: str
    onnx_model_dir: str
    language: str
    runtime_config: Optional[Dict[str, Any]] = None


class UnloadCharacterPayload(BaseModel):
    character_name: str


class ReferenceAudioPayload(BaseModel):
    character_name: str
    audio_path: str
    audio_text: str
    language: str


class TTSPayload(BaseModel):
    character_name: str
    text: str
    split_sentence: bool = False
    save_path: Optional[str] = None


@app.post("/load_character")
def load_character_endpoint(payload: CharacterPayload):
    try:
        model_manager.load_character(
            character_name=payload.character_name,
            model_dir=payload.onnx_model_dir,
            language=normalize_language(payload.language),
            runtime_config=payload.runtime_config,
        )
        return {"status": "success", "message": f"Character '{payload.character_name}' loaded."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/unload_character")
def unload_character_endpoint(payload: UnloadCharacterPayload):
    try:
        model_manager.remove_character(character_name=payload.character_name)
        return {"status": "success", "message": f"Character '{payload.character_name}' unloaded."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/set_reference_audio")
def set_reference_audio_endpoint(payload: ReferenceAudioPayload):
    ext = os.path.splitext(payload.audio_path)[1].lower()
    if ext not in SUPPORTED_AUDIO_EXTS:
        raise HTTPException(
            status_code=400,
            detail=f"Audio format '{ext}' is not supported. Supported formats: {SUPPORTED_AUDIO_EXTS}",
        )
    _reference_audios[payload.character_name] = {
        'audio_path': payload.audio_path,
        'audio_text': payload.audio_text,
        'language': normalize_language(payload.language),
    }
    return {"status": "success", "message": f"Reference audio for '{payload.character_name}' set."}


def run_tts_in_background(
        character_name: str,
        text: str,
        split_sentence: bool,
        save_path: Optional[str],
        chunk_callback: Callable[[Optional[bytes]], None]
):
    try:
        context.current_speaker = character_name
        context.current_prompt_audio = ReferenceAudio(
            prompt_wav=_reference_audios[character_name]['audio_path'],
            prompt_text=_reference_audios[character_name]['audio_text'],
            language=_reference_audios[character_name]['language'],
        )
        tts_player.start_session(
            play=False,
            split=split_sentence,
            save_path=save_path,
            chunk_callback=chunk_callback,
        )
        tts_player.feed(text)
        tts_player.end_session()
        tts_player.wait_for_tts_completion()
    except Exception as e:
        logger.error(f"Error in TTS background task: {e}", exc_info=True)


async def audio_stream_generator(queue: asyncio.Queue) -> AsyncIterator[bytes]:
    while True:
        chunk = await queue.get()
        if chunk is None:
            break
        yield chunk


async def _tracked_audio_stream_generator(queue: asyncio.Queue) -> AsyncIterator[bytes]:
    try:
        async for chunk in audio_stream_generator(queue):
            yield chunk
    finally:
        _release_tts_slot()


async def _acquire_tts_slot() -> None:
    if not _server_runtime.single_process:
        return

    while True:
        should_wait = False
        with _server_runtime.condition:
            if _server_runtime.active_requests < _server_runtime.max_concurrency:
                _server_runtime.active_requests += 1
                return

            if _server_runtime.queue_maxsize and _server_runtime.waiting_requests >= _server_runtime.queue_maxsize:
                raise HTTPException(status_code=429, detail="TTS queue is full.")

            _server_runtime.waiting_requests += 1
            should_wait = True

        try:
            if should_wait:
                await asyncio.to_thread(_wait_for_tts_slot)
        finally:
            with _server_runtime.condition:
                if _server_runtime.waiting_requests > 0:
                    _server_runtime.waiting_requests -= 1


def _wait_for_tts_slot() -> None:
    with _server_runtime.condition:
        while _server_runtime.active_requests >= _server_runtime.max_concurrency:
            _server_runtime.condition.wait()


def _release_tts_slot() -> None:
    if _server_runtime.single_process:
        with _server_runtime.condition:
            if _server_runtime.active_requests > 0:
                _server_runtime.active_requests -= 1
            _server_runtime.condition.notify()


@app.post("/tts")
async def tts_endpoint(payload: TTSPayload):
    await _acquire_tts_slot()
    try:
        if payload.character_name not in _reference_audios:
            raise HTTPException(status_code=404, detail="Character not found or reference audio not set.")

        loop = asyncio.get_running_loop()
        stream_queue: asyncio.Queue[Union[bytes, None]] = asyncio.Queue()

        def tts_chunk_callback(chunk: Optional[bytes]):
            loop.call_soon_threadsafe(stream_queue.put_nowait, chunk)

        loop.run_in_executor(
            None,
            run_tts_in_background,
            payload.character_name,
            payload.text,
            payload.split_sentence,
            payload.save_path,
            tts_chunk_callback
        )

        return StreamingResponse(_tracked_audio_stream_generator(stream_queue), media_type="audio/wav")
    except Exception:
        _release_tts_slot()
        raise


@app.post("/stop")
def stop_endpoint():
    try:
        tts_player.stop()
        return {"status": "success", "message": "TTS stopped."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/clear_reference_audio_cache")
def clear_reference_audio_cache_endpoint():
    try:
        ReferenceAudio.clear_cache()
        return {"status": "success", "message": "Reference audio cache cleared."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def start_server(
        host: str = "127.0.0.1",
        port: int = 8000,
        workers: Optional[int] = None,
        max_concurrency: Optional[int] = None,
        queue_maxsize: Optional[int] = None,
):
    effective_workers = workers if workers is not None else int(os.getenv("GENIE_SERVER_WORKERS", "1"))
    effective_max_concurrency = (
        max_concurrency if max_concurrency is not None else int(os.getenv("GENIE_SERVER_MAX_CONCURRENCY", "1"))
    )
    effective_queue_maxsize = (
        queue_maxsize if queue_maxsize is not None else int(os.getenv("GENIE_SERVER_QUEUE_MAXSIZE", "0"))
    )

    _server_runtime.single_process = effective_workers == 1
    _server_runtime.max_concurrency = effective_max_concurrency
    _server_runtime.queue_maxsize = effective_queue_maxsize
    uvicorn.run(app, host=host, port=port, workers=effective_workers)


if __name__ == "__main__":
    start_server(host="0.0.0.0", port=8000, workers=1)
