#!/usr/bin/python3
"""WebSocket server that performs live transcription of audio streams"""
import asyncio
import argparse
import logging
from typing import Dict, Set, Callable

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn

from subtitles import Subtitles, SubtitleStreamProperties, WhisperDevice, WhisperModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="Whisper Transcription WebSocket Server")

# Global variables
active_connections: Dict[str, Set[WebSocket]] = {}
active_transcribers: Dict[str, Subtitles] = {}
main_event_loop = None
stream_base_url = "https://screamrouter.netham45.org/stream/"


# Special transcription class that directly modifies the write_line method to broadcast updates
class TranscribingSubtitles(Subtitles):
    def __init__(self, stream_properties, update_callback):
        self.update_callback = update_callback
        # Store original write_line method
        self.original_write_line = Subtitles._Subtitles__write_line
        # Replace the method with our own version
        Subtitles._Subtitles__write_line = self._modified_write_line
        super().__init__(stream_properties)
    
    def _modified_write_line(self, line, is_start):
        # Call original method
        self.original_write_line(self, line, is_start)
        # Send update via callback
        if self.update_callback:
            self.update_callback(line)


async def broadcast_message(ip_address: str, message: str):
    """Broadcast a message to all clients connected to a specific IP stream"""
    logger.info(f"Broadcasting message for {ip_address}: {message}")
    if ip_address in active_connections:
        disconnected_clients = set()
        
        for websocket in active_connections[ip_address]:
            try:
                await websocket.send_text(message)
                logger.info(f"Successfully sent message to client")
            except Exception as e:
                logger.error(f"Failed to send message: {e}")
                disconnected_clients.add(websocket)
        
        # Remove disconnected clients
        for client in disconnected_clients:
            active_connections[ip_address].remove(client)
            
        # If no clients left, stop transcription
        if not active_connections[ip_address] and ip_address in active_transcribers:
            stop_transcription(ip_address)


def start_transcription(ip_address: str):
    """Start a transcription process for the given IP address"""
    global stream_base_url
    
    if ip_address in active_transcribers:
        logger.info(f"Transcription for {ip_address} already running")
        return
    
    logger.info(f"Starting transcription for {ip_address}")
    
    # Define a callback function that broadcasts transcription updates
    def update_callback(line: str):
        global main_event_loop
        if main_event_loop:
            logger.info(f"Got transcription update: {line}")
            future = asyncio.run_coroutine_threadsafe(
                broadcast_message(ip_address, line),
                main_event_loop
            )
            # Wait for the future to complete
            try:
                future.result(timeout=1.0)
            except Exception as e:
                logger.error(f"Error in broadcast future: {e}")
    
    # Create stream properties
    stream_url = f"{stream_base_url}{ip_address}/"
    stream_properties = SubtitleStreamProperties(
        device_type=WhisperDevice.CUDA,
        whisper_model=WhisperModel.LARGE_V3_TURBO_DISTILL,
        chunk_duration=0.20,
        num_chunks=25,
        source=stream_url,
        ffmpeg_realtime=False,
        dont_clear=True,
        num_lines=5,
        history_size=300
    )
    
    # Create and start the transcriber
    transcriber = TranscribingSubtitles(stream_properties, update_callback)
    active_transcribers[ip_address] = transcriber
    transcriber.start()
    
    logger.info(f"Transcription started for {ip_address}")


def stop_transcription(ip_address: str):
    """Stop the transcription process for the given IP address"""
    if ip_address in active_transcribers:
        logger.info(f"Stopping transcription for {ip_address}")
        transcriber = active_transcribers[ip_address]
        transcriber.stop()
        del active_transcribers[ip_address]
        logger.info(f"Transcription stopped for {ip_address}")


@app.websocket("/transcribe/{ip_address}")
@app.websocket("/transcribe/{ip_address}/")
async def websocket_endpoint(websocket: WebSocket, ip_address: str):
    """WebSocket endpoint for transcription"""
    try:
        logger.info(f"WebSocket connection attempt for IP: {ip_address}")
        await websocket.accept()
        logger.info(f"WebSocket connection accepted for IP: {ip_address}")
        
        # Register the client for broadcast messages
        if ip_address not in active_connections:
            active_connections[ip_address] = set()
        active_connections[ip_address].add(websocket)
        
        # Start transcription if not already running
        if ip_address not in active_transcribers:
            start_transcription(ip_address)
        
        # Keep the connection open until the client disconnects
        while True:
            try:
                # Wait for messages (which we ignore)
                await websocket.receive_text()
            except WebSocketDisconnect:
                logger.info(f"WebSocket disconnect for IP: {ip_address}")
                break
    except Exception as e:
        logger.error(f"Error in WebSocket connection: {e}")
    finally:
        # Unregister the client
        if ip_address in active_connections and websocket in active_connections[ip_address]:
            active_connections[ip_address].remove(websocket)
            
            # If no clients left, stop transcription
            if not active_connections[ip_address]:
                stop_transcription(ip_address)
        logger.info(f"WebSocket connection closed for IP: {ip_address}")


@app.on_event("startup")
async def startup_event():
    """Run when the server starts up"""
    global main_event_loop
    main_event_loop = asyncio.get_event_loop()
    logger.info("WebSocket server started")


@app.on_event("shutdown")
async def shutdown_event():
    """Run when the server shuts down"""
    logger.info("Shutting down server...")
    
    # Stop all transcription processes
    for ip_address in list(active_transcribers.keys()):
        stop_transcription(ip_address)
    
    logger.info("Server shutdown complete")


def run_server(base_url: str, host: str = "0.0.0.0", port: int = 8085):
    """Run the FastAPI server with Uvicorn"""
    global stream_base_url
    stream_base_url = base_url
    logger.info(f"Using stream base URL: {stream_base_url}")
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="WebSocket server for live audio transcription")
    parser.add_argument(
        "--base-url", 
        default="https://screamrouter.netham45.org/stream/",
        help="Base URL for audio streams (default: https://screamrouter.netham45.org/stream/)"
    )
    parser.add_argument(
        "--host", 
        default="0.0.0.0",
        help="Host to bind the server to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=8085,
        help="Port to bind the server to (default: 8085)"
    )
    
    args = parser.parse_args()
    
    try:
        run_server(base_url=args.base_url, host=args.host, port=args.port)
    except KeyboardInterrupt:
        logger.info("Server terminated by keyboard interrupt")
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
