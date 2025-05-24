# create_video.py (v1.20 - Multi-Product + Overlay Randomization)
import os
import argparse
import time
import uuid
import requests
import imageio_ffmpeg # For managing ffmpeg on different envs
import subprocess # For running ffmpeg
import re # For parsing ffmpeg output
import shutil # For copying files
import random
import json
from datetime import timedelta, datetime # Import datetime
from dotenv import load_dotenv
from openai import OpenAI
from elevenlabs.client import ElevenLabs
from elevenlabs import save
from google.cloud import storage
import traceback # For detailed error logging
import math # For ceiling function later
from typing import Callable, Optional, Tuple
import os, re, time, uuid, traceback
from datetime import datetime
from pathlib import Path

from randomizer import randomize_video

# --- Whisper and FFmpeg specific imports ---
try:
    import whisper
except ImportError:
    print("WARNING: Whisper library not found. Run 'pip install -U openai-whisper'. Overlay feature will fail.")
    whisper = None # Set whisper to None if import fails

# ─── Global Working Directory Setup TODO ────────────────────────────────
HOME_DIR       = Path.home() / ".zyra-video-agent"
WORKING_DIR    = HOME_DIR / "working-dir"
OUTPUT_BASE_DIR = HOME_DIR / "output"

# Make sure they exist
for d in (WORKING_DIR, OUTPUT_BASE_DIR):
    d.mkdir(parents=True, exist_ok=True)

# --- Configuration ---
SCRIPT_VERSION = "1.17 (Overlay Feature Added, Corrected Placement, Helpers Included)" # Updated version
OPENAI_MODEL = "gpt-4o"
ELEVENLABS_MODEL = "eleven_multilingual_v2"
TEMP_AUDIO_FILENAME_BASE = "temp_generated_audio" # Base name, will add UUID
DREAMFACE_SUBMIT_URL = "https://api.newportai.com/api/async/talking_face"
DREAMFACE_POLL_URL = "https://api.newportai.com/api/getAsyncResult"

# Polling config
POLLING_INTERVAL_SECONDS = 15
MAX_POLLING_ATTEMPTS = 40
# Silence removal config
SILENCE_THRESHOLD_DB = "-35dB"
SILENCE_MIN_DURATION_S = "0.4"

# --- Overlay Defaults (Temporary fallback if no config file is used) ---
DEFAULT_OVERLAY_POSITIONS = [{"x": "10", "y": "10", "w": "-1", "h": "-1"}]

# --- Helper Functions (Copied from User's v1.16) ---

def run_ffmpeg_command(cmd):
    """Runs an FFmpeg command using subprocess and returns (success, error_message)."""
    try:
        subprocess.run(cmd, check=True)
        return True, None
    except subprocess.CalledProcessError as e:
        return False, str(e)

def generate_script(client: OpenAI, product: str, persona: str, setting: str, emotion: str, hook_guidance: str, example_script: str, language: str, enhance_for_elevenlabs: bool, brand_name: str) -> str | None:
    """Generates script via OpenAI based on refined prompt structure and length request. Returns script text or None."""


def generate_script(client: OpenAI, product: str, persona: str, setting: str, emotion: str, hook_guidance: str, example_script: str, language: str, enhance_for_elevenlabs: bool, brand_name: str) -> str | None:
    """Generates script via OpenAI based on refined prompt structure and length request. Returns script text or None."""
    # Inside the generate_script function in create_video.py
# (Make sure this is within the function definition that now accepts language, enhance_for_elevenlabs, brand_name)

    print(f"Generating script with model: {OPENAI_MODEL} using updated prompt logic...")
    if not example_script or len(example_script.strip()) < 50:
        print("ERROR: Example script provided is missing or too short. Please provide a valid example.")
        return None

    # --- Step 3 START: Modify Prompt Construction ---

    # 3.1: New System Prompt (More specific constraints)
    system_prompt = (
        "You are a professional scriptwriter and persuasive storyteller focused on crafting emotionally resonant, high-converting TikTok Shop video scripts "
        "Your goal is to transform simple prompts into compelling, human-centered narratives that flow naturally in voiceover — using structured storytelling, emotional insight, and subtle persuasion to spark trust and drive action "
        "You are an introspective storyteller crafting grounded, emotionally compelling TikTok Shop scripts. Your scripts are written like personal reflections the kind of quiet honesty someone might share on a podcast or in a heartfelt voiceover. Avoid buzzwords, and sales tactics. Let the narrative unfold naturally, using human struggles like fatigue, burnout, loss of drive to guide the arc. Speak plainly and insightfully. Your goal is not to sell, but to share and in doing so, build quiet trust."
        "Avoid overused supplement marketing clichés. Do not use words like zest, vital, game-changer, enhances. Speak like someone unpacking their personal journey in a quiet moment not performing."
        "Strictly adhere to the output format requested. Do NOT include explanations, introductions, summaries, "
        "or any text other than the script content itself unless specifically asked to use SSML tags. "
        "Do NOT use markers like 'Script:', '[HOOK]', '[INTRO]', stage directions like '[camera pans]', or sound cues."
    )

    # 3.2: Build the User Prompt Dynamically
    prompt_lines = []
    prompt_lines.append(f"Product: {product}")
    prompt_lines.append(f"Creator Persona: {persona}")
    prompt_lines.append(f"Setting: {setting}")
    prompt_lines.append(f"Emotion: {emotion}")
    prompt_lines.append(f"Language: {language}") # Use language variable
    prompt_lines.append(f"Hook Requirement: {hook_guidance}")
    prompt_lines.append(f"Brand Name to include naturally near the end: {brand_name}") # Add brand name context

    # Add conditional SSML instructions
    if enhance_for_elevenlabs:
        prompt_lines.append(
            "\nIMPORTANT FORMATTING REQUIREMENT: "
            "Make this script perfect for eleven labs to make it sound very human-like. " # Your requested instruction
            "Wrap the entire script output in <speak> tags. Use SSML tags like <break time=\"Xs\"/> for pauses (vary duration appropriately, e.g., 0.3s, 0.7s, 1s) "
            "and <emphasis level=\"moderate\"> for moderate emphasis on key words/phrases to ensure a human-like delivery for ElevenLabs text-to-speech. "
            "Focus on natural pauses and tonality. But don't add anything that tries to change the pronunciation of specific words." # Your requested instruction
        )
        prompt_lines.append("The example script below MAY NOT contain SSML, but your output MUST use SSML tags as described above.") # Clarify example relevance
    else:
        prompt_lines.append(
            "\nIMPORTANT FORMATTING REQUIREMENT: "
            "Output ONLY the raw spoken dialogue text, with no extra tags (like SSML) or formatting."
        )

    # --- Start Replace --- (Replace the single length request line with this block)

    target_duration = "75 to 90 seconds" # Define target duration
    if enhance_for_elevenlabs:
        # If SSML is ON, explicitly ask AI to account for break times
        prompt_lines.append(
            f"\nGenerate one unique script based on these details. IMPORTANT: The total duration, including both the spoken dialogue AND the <break> tag pause times, "
            f"should be approximately {target_duration} long. Please factor in the pause durations when determining script length."
        )
    else:
        # If SSML is OFF, the original time request is probably fine
        prompt_lines.append(
            f"\nGenerate one unique script based on these details. The script should be suitable for a video approximately {target_duration} long."
        )

    # --- End Replace ---

    prompt_lines.append(f"\nHere is an example script primarily for structure, tone, and style inspiration, please follow this style closely (ignore its specific formatting if SSML was requested above):")
    prompt_lines.append(f"\n--- BEGIN EXAMPLE SCRIPT ---")
    prompt_lines.append(example_script) # Assumes example_script is the full string content
    prompt_lines.append(f"--- END EXAMPLE SCRIPT ---")

    # New Final Reminder (accounts for conditional SSML)
    if enhance_for_elevenlabs:
        prompt_lines.append(
            "\nFinal Reminder: Output ONLY the script content enclosed in <speak> tags, using appropriate <break> and <emphasis> SSML tags as requested. "
            "Do not add any other commentary, introductions, summaries, bracketed notes, or stage directions."
        )
    else:
        prompt_lines.append(
            "\nFinal Reminder: Output ONLY the raw spoken dialogue text for the script. "
            "Do not add any commentary, introductions, summaries, SSML tags, bracketed notes, or stage directions."
        )

    # Combine lines into the final user prompt string
    user_prompt = "\n".join(prompt_lines)

    # --- Step 3 END: Modify Prompt Construction ---

    # The rest of the function (OpenAI API call, cleanup) remains largely the same...
    print("\n--- Sending Prompt to OpenAI ---")
    # Optional: Print the constructed prompt to see exactly what's being sent
    # print("--- USER PROMPT ---")
    # print(user_prompt)
    # print("-------------------")
    print("-------------------------------\n")

    try:
        response = client.chat.completions.create(
            # ... rest of API call parameters (model, messages, temperature) ...
            # Make sure messages uses the new system_prompt and user_prompt variables
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7 # Keep temperature for now, can tweak later if needed
        )
        script_content = response.choices[0].message.content.strip()

        # --- IMPORTANT: SSML needs careful cleanup ---
        # The existing cleanup might strip SSML tags. We need to adjust it.
        # If SSML is expected, we probably want LESS cleanup, just removing prefix/suffix junk.
        # If SSML is NOT expected, the old cleanup is fine.

        if enhance_for_elevenlabs:
            # For SSML, maybe only strip leading/trailing whitespace and common GPT preamble/postamble,
            # but be careful not to strip the <speak> tags themselves or tags within.
            # This needs careful testing. Let's start minimal:
            print(f"DEBUG: Raw SSML content from OpenAI:\n---\n{script_content}\n---")
            # Minimal cleanup for SSML - remove common non-SSML junk
            unwanted_phrases_ssml = ["```xml", "```", "Response:", "Output:", "Generated Script:"] # Add others if needed
            cleaned_script = script_content
            for phrase in unwanted_phrases_ssml:
                if cleaned_script.startswith(phrase):
                    cleaned_script = cleaned_script[len(phrase):].lstrip()
                if cleaned_script.endswith(phrase):
                    cleaned_script = cleaned_script[:-len(phrase)].rstrip()
            # Ensure it starts/ends with <speak> tags (basic check)
            if not cleaned_script.startswith("<speak>"):
                print("Warning: SSML output doesn't start with <speak>. Might need manual correction.")
            if not cleaned_script.endswith("</speak>"):
                print("Warning: SSML output doesn't end with </speak>. Might need manual correction.")

        else:
            # Use the ORIGINAL more aggressive cleanup for non-SSML text
            print(f"DEBUG: Raw non-SSML content from OpenAI:\n---\n{script_content}\n---")
            unwanted_phrases = [
                "script:", "here's the script:", "script start", "script end",
                "--- begin script ---", "--- end script ---",
                "--- begin example script ---", "--- end example script ---",
                "okay, here is the script:", "sure, here's a script:", "certainly, here is the script:",
                "here is one script:", "one script:", "```markdown", "```",
                "Response:", "Output:", "Generated Script:", "Okay, here's a script...",
                "<speak>", "</speak>" # Also remove speak tags if enhance was FALSE
            ]
            cleaned_script = script_content
            modified = True
            while modified:
                modified = False
                original_content = cleaned_script
                for phrase in unwanted_phrases:
                    if cleaned_script.lower().startswith(phrase.lower()):
                        cleaned_script = cleaned_script[len(phrase):].lstrip(" \n\t:")
                        modified = True
                        break
                    if cleaned_script.lower().endswith(phrase.lower()):
                        cleaned_script = cleaned_script[:-len(phrase)].rstrip(" \n\t:")
                        modified = True
                        break
            # Add the regex cleanup here as discussed (Idea D) for non-SSML
            cleaned_script = re.sub(r'\[.*?\]', '', cleaned_script) # Remove [...]
            # Optional: Remove (...) - use with caution
            # cleaned_script = re.sub(r'\(.*?\)', '', cleaned_script)
            cleaned_script = "\n".join(line for line in cleaned_script.splitlines() if line.strip()) # Remove empty lines

        cleaned_script = cleaned_script.strip()
        if not cleaned_script:
            print("ERROR: Script content became empty after cleanup.")
            return None

        # --- END Adjust Cleanup ---

        print("Script generated successfully by OpenAI (after cleanup).")
        return cleaned_script

    # ... keep the rest of the function (except block) ...

    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        traceback.print_exc() # Add traceback for OpenAI errors
        return None

def generate_audio(client: ElevenLabs, script_text: str, voice_id: str, output_path: str, model: str = ELEVENLABS_MODEL) -> bool:
    """Generates audio via ElevenLabs. Returns True/False."""
    print(f"Generating audio with ElevenLabs Voice ID: {voice_id}...")
    try:
        if not script_text:
            print("Error: Cannot generate audio from empty script.")
            return False
        audio_bytes = client.generate(text=script_text, voice=voice_id, model=model)
        save(audio_bytes, output_path)
        # Verify file creation and size
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
             print(f"Audio successfully generated and saved to: {output_path}")
             return True
        else:
             print(f"Error: Audio file generation appeared successful but file is missing or empty: {output_path}")
             if os.path.exists(output_path): # Clean up empty file if created
                 try: os.remove(output_path)
                 except OSError: pass
             return False
    except Exception as e:
        print(f"Error calling ElevenLabs API or saving audio: {e}")
        traceback.print_exc() # Add traceback
        return False


def upload_to_gcs(bucket_name: str, source_file_name: str, destination_blob_name: str) -> bool:
    """Uploads a file to the GCS bucket."""
    if not os.path.exists(source_file_name):
        print(f"Error: Source file for GCS upload not found: {source_file_name}"); return False
    if os.path.getsize(source_file_name) == 0:
        print(f"Error: Source file for GCS upload is empty: {source_file_name}"); return False
    print(f"Uploading {source_file_name} to gs://{bucket_name}/{destination_blob_name}...")
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        # Consider adding a timeout? Default might be long.
        blob.upload_from_filename(source_file_name)
        print("File uploaded successfully.")
        return True
    except Exception as e:
        print(f"Error uploading to GCS: {e}")
        traceback.print_exc() # Add traceback
        return False

def generate_signed_url(bucket_name: str, blob_name: str, expiration_minutes: int = 20) -> str | None:
    """Generates a v4 signed URL for downloading a blob."""
    print(f"Generating signed URL for gs://{bucket_name}/{blob_name}...")
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        url_expiration = timedelta(minutes=expiration_minutes)
        # Ensure the blob actually exists before generating URL? Optional, but safer.
        # if not blob.exists():
        #     print(f"Error: Blob {blob_name} does not exist in bucket {bucket_name}.")
        #     return None
        url = blob.generate_signed_url(version="v4", expiration=url_expiration, method="GET")
        print(f"Signed URL generated (valid for {url_expiration.total_seconds() / 60.0} mins).")
        return url
    except Exception as e:
        print(f"Error generating signed URL: {e}")
        traceback.print_exc() # Add traceback
        return None

def submit_dreamface_job(api_key: str, video_url: str, audio_url: str) -> str | None:
    """Submits job to DreamFace /talking_face endpoint. Returns taskId or None."""
    print("Submitting job to DreamFace API...")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    # Consider increasing video_enhance if quality is needed, maybe impacts time/cost?
    payload = {"srcVideoUrl": video_url, "audioUrl": audio_url, "videoParams": {"video_width": 0, "video_height": 0, "video_enhance": 1, "fps": "original"}}
    try:
        response = requests.post(DREAMFACE_SUBMIT_URL, headers=headers, json=payload, timeout=30) # Standard 30s timeout
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        result = response.json()
        if result.get("code") == 0 and result.get("data", {}).get("taskId"):
            task_id = result["data"]["taskId"]
            print(f"DreamFace job submitted successfully. Task ID: {task_id}")
            return task_id
        else:
            # Log more details on failure
            error_msg = result.get('message', 'Unknown error')
            print(f"DreamFace job submission failed: {error_msg}")
            print(f"Full API response: {result}")
            return None
    except requests.exceptions.Timeout:
        print("Error calling DreamFace submit API: Request timed out.")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error calling DreamFace submit API: {e}")
        return None
    except Exception as e: # Catch potential JSON parsing errors or others
        print(f"Error processing DreamFace submit response: {e}")
        traceback.print_exc()
        return None

def poll_dreamface_job(api_key: str, task_id: str) -> str | None:
    """Polls DreamFace /getAsyncResult endpoint. Returns final video URL or None."""
    print(f"Polling DreamFace job status for Task ID: {task_id}...")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"taskId": task_id}
    attempts = 0
    while attempts < MAX_POLLING_ATTEMPTS:
        attempts += 1
        print(f"Polling attempt {attempts}/{MAX_POLLING_ATTEMPTS}...")
        try:
            response = requests.post(DREAMFACE_POLL_URL, headers=headers, json=payload, timeout=30) # Poll timeout
            response.raise_for_status()
            result = response.json()

            if result.get("code") != 0:
                error_msg = result.get('message', 'Unknown polling error')
                print(f"Polling failed: {error_msg}")
                print(f"Full API response: {result}")
                return None # Stop polling on API error

            task_data = result.get("data", {}).get("task", {})
            status = task_data.get("status")
            status_map = {1: "Submitted", 2: "Processing", 3: "Success", 4: "Failed"}
            status_str = status_map.get(status, f"Unknown ({status})")
            print(f"  Current task status: {status_str}")

            if status == 3: # Success
                # Try to extract videoUrl robustly
                videos_list = result.get("data", {}).get("videos", [])
                url = None
                if videos_list and isinstance(videos_list, list) and len(videos_list) > 0:
                    video_info = videos_list[0]
                    if isinstance(video_info, dict):
                        url = video_info.get("videoUrl")
                if not url:
                    url = result.get("data", {}).get("videoUrl") # Fallback to top level if needed

                if url:
                    print(f"DreamFace job successful! Final video URL: {url}")
                    return url
                else:
                    print(f"ERROR: Polling status is Success(3), but videoUrl not found.")
                    print(f"Full response data: {result.get('data')}")
                    return None # Treat as failure if URL missing on success
            elif status == 4: # Failure
                reason = task_data.get("reason", "Unknown reason")
                print(f"DreamFace job failed: {reason}")
                return None
            elif status in [1, 2]: # Still processing
                print(f"  Job still processing. Waiting {POLLING_INTERVAL_SECONDS} seconds...")
                time.sleep(POLLING_INTERVAL_SECONDS)
            else: # Unknown status
                print(f"  Unknown status code encountered: {status}. Stopping polling.")
                return None

        except requests.exceptions.Timeout:
            print(f"Timeout during polling attempt {attempts}. Retrying after {POLLING_INTERVAL_SECONDS}s...")
            time.sleep(POLLING_INTERVAL_SECONDS)
        except requests.exceptions.RequestException as e:
            print(f"Error calling DreamFace poll API: {e}. Retrying after {POLLING_INTERVAL_SECONDS}s...")
            time.sleep(POLLING_INTERVAL_SECONDS)
        except Exception as e:
            print(f"Error processing DreamFace poll response: {e}. Stopping polling.")
            traceback.print_exc()
            return None

    print(f"ERROR: Polling timed out after {MAX_POLLING_ATTEMPTS} attempts.")
    return None

def download_video(video_url: str, local_filename: str) -> bool:
    """Downloads a video from a URL to a local file."""
    print(f"Downloading final video from {video_url} to {local_filename}...")
    try:
        # Use stream=True and iterate content to handle potentially large files
        with requests.get(video_url, stream=True, timeout=(10, 300)) as r: # (connect timeout, read timeout)
            r.raise_for_status()
            # Ensure directory exists before opening file
            os.makedirs(os.path.dirname(local_filename) or '.', exist_ok=True)
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        # Verify download
        if os.path.exists(local_filename) and os.path.getsize(local_filename) > 0:
            print("Video downloaded successfully.")
            return True
        else:
            print(f"Error: Downloaded file is missing or empty after download attempt: {local_filename}")
            if os.path.exists(local_filename): # Clean up potentially corrupted file
                 try: os.remove(local_filename)
                 except OSError: pass
            return False
    except requests.exceptions.Timeout:
        print(f"Error downloading video: Request timed out from {video_url}")
        return False
    except requests.exceptions.RequestException as e:
        print(f"Error downloading video: {e}")
        return False
    except Exception as e:
        print(f"Error saving downloaded video file to {local_filename}: {e}")
        traceback.print_exc()
        # Clean up if file exists but saving failed mid-way
        if os.path.exists(local_filename):
             try: os.remove(local_filename)
             except OSError: pass
        return False

def delete_from_gcs(bucket_name: str, blob_name: str):
    """Deletes a blob from the GCS bucket."""
    print(f"Deleting temporary file gs://{bucket_name}/{blob_name} from GCS...")
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        # Check if blob exists before attempting delete to avoid unnecessary warnings on cleanup
        if blob.exists():
            blob.delete()
            print(f"Temporary file gs://{bucket_name}/{blob_name} deleted successfully.")
        else:
             print(f"Temporary file gs://{bucket_name}/{blob_name} not found, skipping delete.")
    except Exception as e:
        # Log as warning, cleanup failure shouldn't stop the whole process usually
        print(f"Warning: Failed to delete temporary file {blob_name} from GCS: {e}")
        # Optionally add traceback here if needed for debugging GCS issues
        # traceback.print_exc()

def remove_silence_from_video(input_path: str, output_path: str) -> bool:
    """Removes silence from video using ffmpeg silencedetect and applies audio fades."""
    # This is a complex function. Adding detailed comments or breaking it down further
    # might improve maintainability, but using the user's provided code directly for now.
    print(f"\n--- DEBUG: Starting Silence Removal ---")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Silence Threshold: {SILENCE_THRESHOLD_DB}, Min Duration: {SILENCE_MIN_DURATION_S}s")
    if not os.path.exists(input_path):
        print(f"DEBUG: Error - Input video for silence removal not found: {input_path}")
        return False
    print("DEBUG: Detecting silence intervals...")

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    if not ffmpeg_exe:
        print("ERROR: ffmpeg not found. Please install ffmpeg or add it to your PATH.")
        return False

    silence_detect_cmd = [
        ffmpeg_exe, '-nostdin', '-i', input_path,
        '-af', f'silencedetect=noise={SILENCE_THRESHOLD_DB}:d={SILENCE_MIN_DURATION_S}',
        '-f', 'null', '-'
    ]
    print(f"DEBUG: Running silencedetect command: {' '.join(silence_detect_cmd)}")
    try:
        # Increased timeout for silence detection, might take longer on some videos
        process = subprocess.run(silence_detect_cmd, capture_output=True, text=True, check=False, encoding='utf-8', errors='ignore', timeout=120)
        print("DEBUG: silencedetect command finished.")
    except subprocess.TimeoutExpired:
        print(f"DEBUG: Error - FFmpeg silencedetect command timed out for {input_path}")
        return False
    except FileNotFoundError:
        print("DEBUG: Error - ffmpeg command not found. Make sure FFmpeg is installed and in your system's PATH.")
        return False
    except Exception as e:
        print(f"DEBUG: Error running ffmpeg silencedetect: {e}")
        traceback.print_exc()
        return False

    stderr_output = process.stderr
    # print("\n----- DEBUG: FFmpeg silencedetect stderr START -----\n") # Reduce noise
    # print(stderr_output)
    # print("\n----- DEBUG: FFmpeg silencedetect stderr END -----\n")

    # Use try-except for float conversion, more robust
    try:
        silence_starts = [float(t) for t in re.findall(r"silence_start:\s*([\d\.]+)", stderr_output)]
        silence_ends = [float(t) for t in re.findall(r"silence_end:\s*([\d\.]+)", stderr_output)]
    except ValueError as e:
        print(f"DEBUG: Error parsing silence timestamps from ffmpeg output: {e}")
        print(f"FFmpeg stderr was:\n{stderr_output}")
        return False # Cannot proceed if timestamps are invalid

    duration_match = re.search(r"Duration:\s*(\d{2}):(\d{2}):(\d{2})\.(\d+)", stderr_output) # Match one or more decimal digits
    print(f"DEBUG: Detected silence starts: {silence_starts}")
    print(f"DEBUG: Detected silence ends: {silence_ends}")

    if not silence_starts or not silence_ends:
        if "error" in stderr_output.lower() or "invalid" in stderr_output.lower() or process.returncode != 0:
            print("DEBUG: Error detected in FFmpeg silencedetect output. Cannot proceed.")
            print(f"FFmpeg stderr:\n{stderr_output}")
            return False
        print("DEBUG: No silence detected or silence detection output not parsed.")
        try:
            shutil.copy(input_path, output_path)
            print(f"DEBUG: Copied input to output as no silence was processed: {output_path}")
            return True
        except Exception as e:
            print(f"DEBUG: Error copying file when no silence detected: {e}")
            return False

    if len(silence_starts) != len(silence_ends):
        print(f"DEBUG: Warning - Mismatched silence start ({len(silence_starts)}) and end ({len(silence_ends)}) counts. Attempting simple fix.")
        # Simple fix: assume last start or first end is cut off
        if len(silence_starts) > len(silence_ends):
             silence_starts.pop() # Remove last start
        elif len(silence_ends) > len(silence_starts):
             silence_ends.pop(0) # Remove first end
        # If still mismatched or empty after fix, bail out by copying
        if len(silence_starts) != len(silence_ends) or not silence_starts:
            print("DEBUG: Reconciliation failed or resulted in no pairs. Copying original.")
            try: shutil.copy(input_path, output_path); return True
            except Exception as copy_e: print(f"DEBUG: Error copying file after reconciliation failed: {copy_e}"); return False

    print(f"DEBUG: Processing {len(silence_starts)} silence interval(s).")
    video_duration = 0
    if duration_match:
        try:
            h, m, s, ms_str = duration_match.groups()
            # Handle potentially short ms_str like '7' -> 0.07 or '71' -> 0.71
            ms = float(f"0.{ms_str}")
            video_duration = int(h) * 3600 + int(m) * 60 + int(s) + ms
            print(f"DEBUG: Parsed video duration: {video_duration:.3f}s")
        except ValueError as e:
            print(f"DEBUG: Error parsing video duration: {e}. Attempting estimation.")
            video_duration = 0 # Reset duration if parsing failed
    else:
         print("DEBUG: Warning - Could not parse video duration from FFmpeg output.")

    # Estimate duration if parsing failed or wasn't found, but we have silence ends
    if video_duration <= 0 and silence_ends:
        # Estimate based on the maximum end time found
        max_end_time = max(silence_ends) if silence_ends else 0
        if max_end_time > 0:
             # Add a small buffer (e.g., 1 second) to estimated duration
             estimated_duration = max_end_time + 1.0
             print(f"DEBUG: Estimating duration from max silence end time: {estimated_duration:.3f}s")
             video_duration = estimated_duration
        else:
             print(f"DEBUG: Error - Cannot reliably determine video duration.")
             return False # Cannot proceed without duration

    if video_duration <= 0:
        print(f"DEBUG: Error - Invalid or undetermined video duration ({video_duration:.3f}s).")
        return False

    # --- Build segments to keep ---
    segments = []
    min_segment_len = 0.1 # Minimum length of a non-silent segment to keep

    last_end_time = 0.0
    for i in range(len(silence_starts)):
         start_segment = last_end_time
         end_segment = silence_starts[i]
         # Ensure times are valid and segment has minimum length
         if start_segment >= 0 and end_segment > start_segment and (end_segment - start_segment) >= min_segment_len:
              segments.append((start_segment, end_segment))
         # Update last_end_time for the next iteration
         last_end_time = silence_ends[i]
         # Handle potential negative end times from detection? Clamp to 0?
         if last_end_time < 0: last_end_time = 0

    # Add the final segment after the last silence
    if last_end_time < video_duration and (video_duration - last_end_time) >= min_segment_len:
         segments.append((last_end_time, video_duration))

    if not segments:
        print("DEBUG: No non-silent segments found after processing silence intervals.")
        try: shutil.copy(input_path, output_path); return True
        except Exception as e: print(f"DEBUG: Error copying file when no segments found: {e}"); return False

    print(f"DEBUG: Identified {len(segments)} non-silent segments to keep:")
    total_kept_duration = 0.0
    for i, (start, end) in enumerate(segments):
        # Clamp segment times to ensure they are within video duration
        clamped_start = max(0.0, start)
        clamped_end = min(video_duration, end)
        duration = clamped_end - clamped_start
        if duration < min_segment_len / 2: # Skip tiny segments after clamping
             print(f"  Segment {i}: Skipping tiny segment after clamping ({clamped_start:.3f}s -> {clamped_end:.3f}s)")
             continue
        print(f"  Segment {i}: {clamped_start:.3f}s -> {clamped_end:.3f}s (Duration: {duration:.3f}s)")
        total_kept_duration += duration

    if total_kept_duration == 0:
         print("DEBUG: Error - Total duration of segments to keep is zero.")
         return False

    # --- Build complex filtergraph ---
    fade_duration = 0.05 # Shorter fade
    video_select_parts = []
    audio_filter_chains = []
    valid_segment_count = 0

    current_offset = 0.0 # Track time offset for concatenation
    for i, (start, end) in enumerate(segments):
        clamped_start = max(0.0, start)
        clamped_end = min(video_duration, end)
        segment_duration = clamped_end - clamped_start

        if segment_duration < min_segment_len / 2.0: continue # Skip tiny segments

        v_select = f"between(t,{clamped_start},{clamped_end})"
        video_select_parts.append(v_select)

        # Audio chain for this segment
        in_label = f"[0:a]" # Original audio input
        trim_label = f"[a_trimmed_{valid_segment_count}]"
        fade_label = f"[a_faded_{valid_segment_count}]"

        # Trim audio segment and reset its timestamp
        trim_filter = f"{in_label}atrim={clamped_start}:{clamped_end},asetpts=PTS-STARTPTS{trim_label}"

        # Apply fades (in for all except first, out for all except last)
        fade_filters = []
        effective_fade_duration = min(fade_duration, segment_duration / 2.0) # Ensure fade isn't longer than half segment
        if valid_segment_count > 0: # Fade in if not the first segment
             fade_filters.append(f"afade=t=in:st=0:d={effective_fade_duration}")
        if i < len(segments) -1: # Check original index to see if it's the last *potential* segment
             # Check if the *next* valid segment exists to determine if fade out is needed
             next_valid_exists = False
             for k in range(i + 1, len(segments)):
                  next_start, next_end = segments[k]
                  next_clamped_start = max(0.0, next_start)
                  next_clamped_end = min(video_duration, next_end)
                  if (next_clamped_end - next_clamped_start) >= min_segment_len / 2.0:
                       next_valid_exists = True
                       break
             if next_valid_exists: # Fade out if not the last valid segment
                 fade_out_start = max(0.0, segment_duration - effective_fade_duration)
                 fade_filters.append(f"afade=t=out:st={fade_out_start:.3f}:d={effective_fade_duration}")

        if fade_filters:
            fade_chain = f"{trim_label}{','.join(fade_filters)}{fade_label}"
        else: # No fades needed (single segment case)
             fade_chain = f"{trim_label}anull{fade_label}" # Use anull filter as placeholder

        audio_filter_chains.append(trim_filter + ";" + fade_chain)
        valid_segment_count += 1

    if valid_segment_count == 0:
        print("DEBUG: Error - No valid segments left after processing for filtergraph.")
        try: shutil.copy(input_path, output_path); print("DEBUG: Copying original file as fallback."); return True
        except Exception as copy_e: print(f"DEBUG: Error copying fallback file: {copy_e}"); return False

    # --- Combine filters ---
    video_filtergraph = f"select='{'+'.join(video_select_parts)}',setpts=N/(FRAME_RATE*TB)[outv]" # Correct PTS adjustment for select

    if valid_segment_count == 1: # Simpler audio graph if only one segment
         # The single audio chain already ends in [a_faded_0]
         full_audio_filtergraph = audio_filter_chains[0].replace('[a_faded_0]', '[outa]') # Rename final output label
    else: # Concatenate multiple audio segments
        concat_inputs = "".join([f"[a_faded_{j}]" for j in range(valid_segment_count)])
        audio_concat_filter = f"{concat_inputs}concat=n={valid_segment_count}:v=0:a=1[outa]"
        full_audio_filtergraph = ";".join(audio_filter_chains) + ";" + audio_concat_filter

    filter_complex_string = f"{full_audio_filtergraph};{video_filtergraph}"

    print("\n----- DEBUG: Generated Filter Complex String START -----\n")
    print(filter_complex_string)
    print("\n----- DEBUG: Generated Filter Complex String END -----\n")
    print(f"DEBUG: Estimated final duration: {total_kept_duration:.3f}s")

    print(f"DEBUG: Generating final trimmed video with audio fades: {output_path}...")
    final_cmd = [
        ffmpeg_exe, '-hide_banner', '-loglevel', 'warning', # Less verbose output
        '-i', input_path,
        '-filter_complex', filter_complex_string,
        '-map', '[outv]', # Map final video stream
        '-map', '[outa]', # Map final audio stream
        # Encoding parameters (consider adjusting preset/crf for speed/quality)
        '-c:v', 'libx264', '-preset', 'medium', '-crf', '23',
        '-c:a', 'aac', '-b:a', '128k',
        '-movflags', '+faststart', # Good for web/streaming
        '-y', # Overwrite output without asking
        output_path
    ]
    # print(f"DEBUG: Running final ffmpeg command: {' '.join(final_cmd)}") # Less verbose now

    try:
        # Increased timeout for final encoding
        process = subprocess.run(final_cmd, capture_output=True, text=True, check=True, encoding='utf-8', errors='ignore', timeout=600)
        print("DEBUG: Final ffmpeg command finished successfully.")
        if process.stderr: # Log warnings/info even on success
             print("--- FFmpeg Info/Warnings ---")
             print(process.stderr)
             print("--------------------------")
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            print("DEBUG: Error - FFmpeg reported success, but output file is missing or empty.")
            try: shutil.copy(input_path, output_path); print("DEBUG: Copying original file as fallback."); return True
            except Exception as copy_e: print(f"DEBUG: Error copying fallback file: {copy_e}"); return False
        print("DEBUG: Silence removal with fades successful.")
        return True
    except subprocess.TimeoutExpired:
        print("DEBUG: Error - Final ffmpeg command timed out.")
        try: shutil.copy(input_path, output_path); print("DEBUG: Copying original file as fallback."); return True
        except Exception as copy_e: print(f"DEBUG: Error copying fallback file: {copy_e}"); return False
    except subprocess.CalledProcessError as e:
        print(f"DEBUG: Error running final ffmpeg command (Return Code: {e.returncode})")
        print("\n----- DEBUG: Final FFmpeg stderr START -----\n")
        print(e.stderr)
        print("\n----- DEBUG: Final FFmpeg stderr END -----\n")
        try: shutil.copy(input_path, output_path); print("DEBUG: Copying original file as fallback."); return True
        except Exception as copy_e: print(f"DEBUG: Error copying fallback file: {copy_e}"); return False
    except FileNotFoundError:
        print("DEBUG: Error - ffmpeg not found during final command execution.")
        return False
    except Exception as e:
        print(f"DEBUG: An unexpected error occurred during final ffmpeg execution: {e}")
        traceback.print_exc()
        try: shutil.copy(input_path, output_path); print("DEBUG: Copying original file as fallback."); return True
        except Exception as copy_e: print(f"DEBUG: Error copying fallback file: {copy_e}"); return False
    finally:
         print(f"--- DEBUG: Ending Silence Removal ---")


# --- Whisper Timestamp Function (Simplified Logic v2) ---
transcription_cache = {}
whisper_model = None

# --- Whisper Timestamp Function (v3 - First Trigger + Fixed Duration) ---
transcription_cache = {}
whisper_model = None


# === Helper Function for Phase 2: Calculate Overlay Geometry (Revised for Constraints) ===
def calculate_overlay_geometry(placement_str, relative_size, main_w, main_h, overlay_aspect_ratio,
                               margin_percent=7):  # Increased default margin to 7%
    """
    Calculates overlay X, Y, Width, Height based on placement string, relative size,
    main video dimensions, overlay aspect ratio, and margin preference.
    Supports multiple placement positions around the video frame.

    Args:
        placement_str (str): Position code like "top_left", "middle_right", "bottom_center", etc.
        relative_size (float): Desired overlay width relative to main video width (e.g., 0.4).
        main_w (int): Width of the main video in pixels.
        main_h (int): Height of the main video in pixels.
        overlay_aspect_ratio (float): Width / Height of the overlay clip itself.
        margin_percent (int): Percentage margin from edges (default: 7).

    Returns:
        dict | None: Dictionary {'x': int, 'y': int, 'w': int, 'h': int} or None if inputs are invalid.
    """
    if not all([placement_str, relative_size, main_w, main_h, overlay_aspect_ratio]):
        print("ERROR: Invalid inputs to calculate_overlay_geometry.")
        return None
    if main_w <= 0 or main_h <= 0 or overlay_aspect_ratio <= 0 or relative_size <= 0:
        print("ERROR: Non-positive dimension or size input to calculate_overlay_geometry.")
        return None

    print(
        f"Calculating geometry for: placement='{placement_str}', rel_size={relative_size:.2f}, main={main_w}x{main_h}, AR={overlay_aspect_ratio:.2f}")

    # Calculate pixel margins
    margin_x = int(main_w * margin_percent / 100)
    margin_y = int(main_h * margin_percent / 100)

    # Calculate overlay dimensions
    overlay_w = int(relative_size * main_w)
    overlay_h = int(overlay_w / overlay_aspect_ratio)

    # Ensure overlay fits within the frame (considering potential margins)
    max_allowable_w = main_w - 2 * margin_x
    max_allowable_h = main_h - 2 * margin_y  # Max height also constrained by margins now
    if overlay_w > max_allowable_w:
        overlay_w = max_allowable_w
        overlay_h = int(overlay_w / overlay_aspect_ratio)  # Recalculate height if width clamped
    if overlay_h > max_allowable_h:
        overlay_h = max_allowable_h
        overlay_w = int(overlay_h * overlay_aspect_ratio)  # Recalculate width if height clamped

    if overlay_w <= 0 or overlay_h <= 0:
        print("ERROR: Calculated overlay dimension is zero or negative after margin/size checks.")
        return None

    # --- Calculate X, Y based on placement string ---
    x, y = 0, 0  # Default initialization

    # Handle different placement options with proper edge alignment
    if placement_str == "top_left":
        x, y = 0, margin_y  # Align to left edge
    elif placement_str == "top_center":
        x, y = (main_w - overlay_w) // 2, margin_y
    elif placement_str == "top_right":
        x, y = main_w - overlay_w, margin_y  # Align to right edge
    elif placement_str == "middle_left":
        x, y = 0, (main_h - overlay_h) // 2  # Align to left edge
    elif placement_str == "middle_center" or placement_str == "center":
        x, y = (main_w - overlay_w) // 2, (main_h - overlay_h) // 2
    elif placement_str == "middle_right":
        x, y = main_w - overlay_w, (main_h - overlay_h) // 2  # Align to right edge
    elif placement_str == "bottom_left":
        x, y = 0, main_h - overlay_h - margin_y  # Align to left edge
    elif placement_str == "bottom_center":
        x, y = (main_w - overlay_w) // 2, main_h - overlay_h - margin_y
    elif placement_str == "bottom_right":
        x, y = main_w - overlay_w, main_h - overlay_h - margin_y  # Align to right edge
    else:
        # Fallback if an unexpected placement string is passed
        print(f"WARNING: Unsupported placement string '{placement_str}'. Defaulting to middle_left calculation.")
        x, y = 0, (main_h - overlay_h) // 2  # Default to left edge

    # Final check to prevent going off-screen (shouldn't happen with clamping above, but safe)
    if x + overlay_w > main_w: x = main_w - overlay_w
    if y + overlay_h > main_h: y = main_h - overlay_h

    print(f"Calculated Geometry: X={x}, Y={y}, W={overlay_w}, H={overlay_h}")
    return {'x': x, 'y': y, 'w': overlay_w, 'h': overlay_h}
# === End Helper Function ===

def get_product_mention_times(audio_path: str, trigger_keywords: list[str], language: str, job_name: str = "Job", desired_duration: float = 5.0) -> tuple[float | None, float | None]: # Added language parameter
    """
    Analyzes audio using Whisper to find the FIRST occurrence of any trigger keyword
    and returns a fixed duration window starting from that point.

    Args:
        audio_path: Path to the audio file.
        trigger_keywords: List of keywords (case-insensitive) to trigger the overlay.
        job_name: Identifier for logging.
        desired_duration: How long the overlay should last in seconds (default: 5.0).
                           Adjust this value between 3.0 and 8.0 as needed.

    Returns:
        A tuple (start_time, end_time) in seconds if a trigger keyword is found,
        otherwise (None, None).
    """
    global whisper_model
    if whisper is None:
        print(f"ERROR [{job_name}]: Whisper library not installed. Cannot analyze.")
        return None, None

    print(f"[{job_name}] Analyzing audio for trigger keywords: {trigger_keywords}")

    # Use cached result if available (optional, clear cache if script logic changes significantly)
    if audio_path in transcription_cache:
        print(f"[{job_name}] Using cached transcription for {audio_path}")
        result = transcription_cache[audio_path]
    else:
        try:
            if whisper_model is None:
                # --- Ensure correct model is specified here ---
                target_model = "small.en" # Or "medium.en" etc.
                print(f"[{job_name}] Loading Whisper model ({target_model})...")
                whisper_model = whisper.load_model(target_model)
                print(f"[{job_name}] Whisper model loaded.")

            print(f"[{job_name}] Transcribing audio file: {audio_path} with word timestamps...")
            language_code = {"english": "en", "spanish": "es"}.get(language.lower(), "en")
            result = whisper_model.transcribe(audio_path, word_timestamps=True, fp16=False, language=language_code)
            transcription_cache[audio_path] = result
            print(f"[{job_name}] Transcription complete.")
            # Optional: Log full transcript for debugging
            print(f"DEBUG [{job_name}]: Whisper Transcript Text:\n{result.get('text', 'N/A')}\n-----")

        except Exception as e:
            print(f"ERROR [{job_name}]: Whisper transcription failed for {audio_path}: {e}")
            traceback.print_exc()
            if audio_path in transcription_cache: del transcription_cache[audio_path]
            return None, None

    # --- Search Logic: Find FIRST trigger keyword ---
    if not result or 'segments' not in result:
        print(f"Warning [{job_name}]: Whisper result invalid or missing segments.")
        return None, None

    # --- Start Paste --- (Paste this block where you just deleted)

    # Prepare initial keywords from the input list
    base_trigger_keywords = {keyword.lower().strip() for keyword in trigger_keywords if keyword}

    # --- Add conditional keywords based on language ---
    final_trigger_keywords = set(base_trigger_keywords) # Start with a copy of base keywords
    # Check language case-insensitively
    if language and language.lower() == 'spanish':
        print(f"[{job_name}] Language is Spanish, adding 'gomitas' to trigger keywords.")
        final_trigger_keywords.add("gomitas") # Add the Spanish word
    # --- End conditional keywords ---

    # Check if there are any keywords to search for *after* potential additions
    if not final_trigger_keywords:
        print(f"Warning [{job_name}]: No valid trigger keywords to search for.")
        return None, None

    # Use the potentially expanded set of keywords for searching
    print(f"[{job_name}] Searching for FIRST occurrence of any keyword in {final_trigger_keywords}...")

    # --- End Paste ---

    for segment in result.get('segments', []):
        for word_info in segment.get('words', []):
            if not isinstance(word_info, dict) or 'word' not in word_info or 'start' not in word_info:
                continue # Skip invalid word data

            word_text = word_info['word'].lower().strip(".,!?;:").strip() # Strip punctuation AND whitespace

            if word_text in final_trigger_keywords:
                # Found the FIRST match!
                found_start_time = word_info['start']
                print(f"[{job_name}] Found FIRST trigger keyword '{word_text}' (from search set {final_trigger_keywords}) at {found_start_time:.2f}s")
                # Calculate end time based on desired duration
                found_end_time = found_start_time + desired_duration
                print(f"[{job_name}] Setting overlay end time to {found_end_time:.2f}s ({desired_duration}s duration)")
                # --- IMPORTANT: Return immediately after finding the first match ---
                return found_start_time, found_end_time

    # If loop finishes without finding any trigger keyword
    print(f"[{job_name}] Trigger keywords {final_trigger_keywords} not found in audio.")
    return None, None

# --- Core Video Generation Function (Modified Signature) ---
def create_video_job(
    # --- Existing parameters ---
    product: str, persona: str, setting: str, emotion: str, hook: str,
    elevenlabs_voice_id: str, avatar_video_path: str, example_script_content: str,
    remove_silence: bool, language: str, enhance_for_elevenlabs: bool, brand_name: str,
    # --- API keys / Config ---
    openai_api_key: str, elevenlabs_api_key: str, dreamface_api_key: str, gcs_bucket_name: str,
    output_path: str, use_randomization: bool, randomization_intensity: str = "medium",
    # --- Job Info ---
    job_name: str = "Unnamed Job",
    # --- Progress callback ---
    progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> tuple[bool, str]:
    """
    Performs the complete video generation process for a single job.
    Returns (True, output_path) on success,
    or (False, last_error_message) on failure.
    """
    # === Optional: Add Repeatable Randomness Seed ===
    # If you want the *same* job_name to always pick the same random values (useful for debugging)
    # uncomment the next line. Otherwise, randomness will be different each run.
    # random.seed(job_name)
    # === End Optional Seed ===

    steps = [
        "Initialization...",
        "Generating script",
        "Synthesizing audio",
        "Uploading to GCS",
        "Signing uploaded files",
        "Running lip-sync",
        "Checking generated video",
        "Removing silence",
        "Preparing video base",
        "Randomizing video",
        "Uploading video result"
    ]
    total_steps = len(steps)
    step = 0
    last_error_message = ""

    print(f"\n--- Starting Job: {job_name} [{datetime.now().isoformat()}] ---")
    job_start_time = time.time()
    # Step 1: Initialization
    if progress_callback:
        progress_callback(step, total_steps, steps[step])
        step += 1

    # --- Validate Inputs ---
    if not all([product, persona, setting, emotion, hook, elevenlabs_voice_id, avatar_video_path, example_script_content]):
        print(f"ERROR [{job_name}]: Missing one or more required text parameters.")
        last_error_message = "Missing one or more required text parameters."
        return False, last_error_message
    if not all([openai_api_key, elevenlabs_api_key, dreamface_api_key, gcs_bucket_name]):
        print(f"ERROR [{job_name}]: Missing one or more required API keys or GCS bucket name.")
        last_error_message = "Missing one or more required API keys or GCS bucket name"
        return False, last_error_message
    if not os.path.exists(avatar_video_path):
        print(f"ERROR [{job_name}]: Avatar video file not found: {avatar_video_path}")
        last_error_message = f"Avatar video file not found: {avatar_video_path}"
        return False, last_error_message
    if len(example_script_content.strip()) < 50:
        print(f"Warning [{job_name}]: Example script content seems very short.")

    # --- API Client Initialization ---
    try:
        openai_client = OpenAI(api_key=openai_api_key)
        print(f"[{job_name}] OpenAI client initialized.")
    except Exception as e:
        print(f"ERROR [{job_name}] initializing OpenAI client: {e}")
        traceback.print_exc()    # Add traceback and return
        last_error_message = f"Failed initializing OpenAI client: {e}"
        return False, last_error_message
    try:
        elevenlabs_client = ElevenLabs(api_key=elevenlabs_api_key)
        print(f"[{job_name}] ElevenLabs client initialized.")
    except Exception as e:
        print(f"ERROR [{job_name}] Failed initializing ElevenLabs client: {e}")
        traceback.print_exc()    # Add traceback and return
        last_error_message = f"Failed initializing ElevenLabs client: {e}"
        return False, last_error_message

    # --- Unique Identifiers and Paths for this Job ---
    run_uuid = uuid.uuid4().hex[:8]
    timestamp_uuid = f"{int(time.time())}_{run_uuid}"
    temp_audio_filename = str(WORKING_DIR / f"{TEMP_AUDIO_FILENAME_BASE}_{run_uuid}.mp3")
    raw_downloaded_video_path = str(WORKING_DIR / f"temp_video_raw_{timestamp_uuid}.mp4")
    # Define GCS names early for use in finally block, even if upload fails
    gcs_audio_blob_name = f"audio_uploads/{timestamp_uuid}_audio.mp3"
    gcs_video_blob_name = f"video_uploads/{timestamp_uuid}_avatar.mp4"

    # --- Sanitize Names for Filename/Paths ---
    sanitized_product_name = re.sub(r'[^\w\-]+', '_', product).strip('_')
    if not sanitized_product_name: sanitized_product_name = "unknown_product"
    try:
        avatar_filename = os.path.basename(avatar_video_path)
        avatar_name_base, _ = os.path.splitext(avatar_filename)
        sanitized_avatar_name = re.sub(r'[^\w\-]+', '_', avatar_name_base).strip('_')
        if not sanitized_avatar_name: sanitized_avatar_name = "unknown_avatar"
    except Exception as e:
        print(f"Warning [{job_name}]: Could not extract avatar name from path: {e}")
        sanitized_avatar_name = "unknown_avatar"

    # --- Define Output Paths ---
    today_date_str = datetime.now().strftime('%Y-%m-%d')
    print(f"DEBUG: today_date_str = {today_date_str}") # DEBUG PRINT
    print(f"DEBUG: OUTPUT_BASE_DIR = {output_path}") # DEBUG PRINT
    print(f"DEBUG: sanitized_product_name = {sanitized_product_name}") # DEBUG PRINT

    # Define the output_product_folder path string first
    # Wrap this in its own try-except in case os.path.join fails
    output_product_folder = None # Initialize to None
    path = None
    if not os.path.exists(output_path):
        path = OUTPUT_BASE_DIR
    else:
        path = output_path

    try:
        output_product_folder_path_string = os.path.join(path, today_date_str, sanitized_product_name)
        print(f"DEBUG: Attempting to define output_product_folder as string: '{output_product_folder_path_string}'") # DEBUG PRINT
        output_product_folder = output_product_folder_path_string # Assign the string path
    except Exception as join_e:
        print(f"ERROR [{job_name}]: Failed during os.path.join for output_product_folder: {join_e}")
        traceback.print_exc()
        output_product_folder = "." # Fallback on any join error
        print(f"DEBUG: Set output_product_folder to '.' due to os.path.join Exception") # DEBUG PRINT

    # --- Define output_file_base AFTER output_product_folder is set --- <<< MOVED HERE
    output_file_base = os.path.join(output_product_folder, f"{job_name}_{run_uuid}")

    # Now try creating the directory using the determined path string
    try:
        # Ensure output_product_folder is a usable path before os.makedirs
        if not output_product_folder: output_product_folder = "." # Ensure it's at least "."

        os.makedirs(output_product_folder, exist_ok=True)
        print(f"[{job_name}] Ensured output directory exists: {output_product_folder}")
        print(f"DEBUG: Successfully created/found output_product_folder: '{output_product_folder}'") # DEBUG PRINT
    except OSError as e:
        print(f"ERROR [{job_name}]: Could not create output directory '{output_product_folder}': {e}. Saving locally.")
        output_product_folder = "." # Fallback assignment only on OSError
        print(f"DEBUG: Set output_product_folder to '.' due to OSError") # DEBUG PRINT
    except Exception as e_generic: # Catch any other potential error during makedirs
         print(f"ERROR [{job_name}]: Unexpected error creating directory '{output_product_folder}': {e_generic}")
         traceback.print_exc()
         output_product_folder = "." # Fallback just in case
         print(f"DEBUG: Set output_product_folder to '.' due to generic Exception during makedirs") # DEBUG PRINT

    # --- Define output_file_base AFTER output_product_folder should be set ---
    # Add prints right before the line that failed
    print(f"DEBUG: About to define output_file_base.") # DEBUG PRINT
    # Use locals().get() for safer access in debug print just in case it's still None or undefined
    print(f"DEBUG: Value of output_product_folder just before use: '{locals().get('output_product_folder', 'Not Defined!')}'") # DEBUG PRINT
    print(f"DEBUG: Value of job_name just before use: '{locals().get('job_name', 'Not Defined!')}'") # DEBUG PRINT
    print(f"DEBUG: Value of run_uuid just before use: '{locals().get('run_uuid', 'Not Defined!')}'") # DEBUG PRINT

    # The line that previously failed, wrapped in try-except for more info
    output_file_base = None # Initialize
    try:
        # Ensure output_product_folder has a usable value before joining
        if not output_product_folder:
             print(f"ERROR [{job_name}]: output_product_folder is missing before defining output_file_base!")
             raise ValueError("output_product_folder was not set correctly.")

        output_file_base = os.path.join(output_product_folder, f"{job_name}_{run_uuid}")
        print(f"DEBUG: Successfully defined output_file_base: {output_file_base}") # DEBUG PRINT
    except Exception as base_e:
         print(f"ERROR [{job_name}]: Failed during os.path.join for output_file_base: {base_e}")
         traceback.print_exc()
         # Cannot proceed without output_file_base, maybe raise or return False?
         print(f"ERROR [{job_name}]: Cannot define output_file_base, exiting job.")
         last_error_message = "Cannot define output_file_base, exiting job" # Exit the job cleanly if this fails
         return False, last_error_message

    # --- Define other specific output filenames that depend on output_file_base ---
    # Make sure these come AFTER output_file_base is successfully defined
    silence_removed_path = f"{output_file_base}_edited.mp4"
    # final_raw_video_path is defined using output_product_folder earlier, which is okay
    final_output_with_overlay_path = f"{output_file_base}_final_overlay.mp4"


    # KEEP these definitions using output_product_folder as they define specific target paths for earlier steps
    edited_filename = f"edited_{sanitized_product_name}_{sanitized_avatar_name}_{run_uuid}.mp4"
    raw_filename = f"raw_{sanitized_product_name}_{sanitized_avatar_name}_{run_uuid}.mp4"
    edited_video_path = os.path.abspath(os.path.join(output_product_folder, edited_filename))
    final_raw_video_path = os.path.abspath(os.path.join(output_product_folder, raw_filename))

    # --- Variable Placeholders ---
    final_output_path = None # Will hold the path *before* overlay attempt
    video_with_overlay_path = None # Will hold the path *after* overlay attempt

    # --- Main Process Steps ---
    step_start_time = time.time()
    try:
        # Step 2: Generate Script
        if progress_callback:
            progress_callback(step, total_steps, steps[step])
            step += 1
        print(f"\n--- [{job_name}] Step 2: Generate Script ---")
        generated_script = generate_script(
            openai_client, product, persona, setting, emotion, hook, example_script_content, language=language, enhance_for_elevenlabs=enhance_for_elevenlabs, brand_name=brand_name
        )
        if not generated_script:
            print(f"ERROR [{job_name}]: Script generation failed.")
            last_error_message = "Script generation failed"
            return False, last_error_message
        print(f"[{job_name}] Generated Script Preview:\n---\n{generated_script[:200]}...\n---")
        print(f"[{job_name}] Step 2 completed in {time.time() - step_start_time:.2f}s"); step_start_time = time.time()

        # Step 3: Generate Audio
        if progress_callback:
            progress_callback(step, total_steps, steps[step])
            step += 1
        print(f"\n--- [{job_name}] Step 3: Generate Audio ---")
        audio_success = generate_audio(
            elevenlabs_client, generated_script, elevenlabs_voice_id, temp_audio_filename
        )
        if not audio_success:
            print(f"ERROR [{job_name}]: Audio generation failed.")
            last_error_message = "Audio generation failed"
            return False, last_error_message
        print(f"[{job_name}] Step 3 completed in {time.time() - step_start_time:.2f}s"); step_start_time = time.time()

        # Step 4: Upload & Get URLs
        if progress_callback:
            progress_callback(step, total_steps, steps[step])
            step += 1
        print(f"\n--- [{job_name}] Step 4: Upload & Get URLs ---")
        audio_upload_success = upload_to_gcs(gcs_bucket_name, temp_audio_filename, gcs_audio_blob_name)
        # Only upload avatar video if audio succeeded (save API call if not needed)
        video_upload_success = False
        if audio_upload_success:
            video_upload_success = upload_to_gcs(gcs_bucket_name, avatar_video_path, gcs_video_blob_name)
        if not (audio_upload_success and video_upload_success): # Cleanup in finally
            print(f"ERROR [{job_name}]: GCS upload failed.")
            last_error_message = "GCS upload failed"
            return False, last_error_message

        # Step 5: Signing uploaded files
        if progress_callback:
            progress_callback(step, total_steps, steps[step])
            step += 1
        audio_signed_url = generate_signed_url(gcs_bucket_name, gcs_audio_blob_name)
        video_signed_url = generate_signed_url(gcs_bucket_name, gcs_video_blob_name)
        if not (audio_signed_url and video_signed_url): # Cleanup in finally
            print(f"ERROR [{job_name}]: Signed URL generation failed.")
            last_error_message = "Signed URL generation failed"
            return False, last_error_message
        print(f"[{job_name}] Step 4 completed in {time.time() - step_start_time:.2f}s"); step_start_time = time.time()
        # Delete local temp audio *after* GCS upload confirmed successful
        try: os.remove(temp_audio_filename); print(f"[{job_name}] Deleted local temp audio: {temp_audio_filename}")
        except OSError as e: print(f"Warning [{job_name}]: Failed to delete {temp_audio_filename}: {e}")

        # Step 6: DreamFace Lip-Sync
        if progress_callback:
            progress_callback(step, total_steps, steps[step])
            step += 1
        print(f"\n--- [{job_name}] Step 5: DreamFace Lip-Sync ---")
        task_id = submit_dreamface_job(dreamface_api_key, video_signed_url, audio_signed_url)
        if not task_id:         # Cleanup in finally
            print(f"ERROR [{job_name}]: DreamFace job submission failed.")
            last_error_message = "DreamFace job submission failed"
            return False, last_error_message
        final_video_url = poll_dreamface_job(dreamface_api_key, task_id)
        if not final_video_url:  # Cleanup in finally
            print(f"ERROR [{job_name}]: Failed to get final video URL from DreamFace.")
            last_error_message = "Failed to get final video URL from DreamFace"
            return False, last_error_message
        # Step 7: Checking generated video
        if progress_callback:
            progress_callback(step, total_steps, steps[step])
            if remove_silence:
                step += 1
            else:
                step += 2
        # Ensure download path exists
        os.makedirs(os.path.dirname(raw_downloaded_video_path) or '.', exist_ok=True)
        download_success = download_video(final_video_url, raw_downloaded_video_path)
        if not download_success or not os.path.exists(raw_downloaded_video_path) or os.path.getsize(raw_downloaded_video_path) == 0:
            print(f"ERROR [{job_name}]: Failed to download, verify, or got empty file from DreamFace: {raw_downloaded_video_path}.")
            last_error_message = f"Failed to download, verify, or got empty file from DreamFace: {raw_downloaded_video_path}"
            return False, last_error_message
        print(f"[{job_name}] Raw lip-synced video saved locally temporarily as: {raw_downloaded_video_path}")
        print(f"[{job_name}] Step 5 completed in {time.time() - step_start_time:.2f}s"); step_start_time = time.time()


        print(f"\n--- [{job_name}] Step 6: Finalize Base Video (Silence Removal / Rename) ---")
        current_video_path = raw_downloaded_video_path # Start with the downloaded path
        intended_final_path = None # Path *before* overlay

        if remove_silence:
            # Step 8: Silence Removal / Rename
            if progress_callback:
                progress_callback(step, total_steps, steps[step])
                step += 1
            print(f"[{job_name}] Attempting silence removal from {current_video_path} to {edited_video_path}...")
            # Ensure output dir for edited video exists
            os.makedirs(os.path.dirname(edited_video_path) or '.', exist_ok=True)
            edit_success = remove_silence_from_video(current_video_path, edited_video_path)
            if edit_success and os.path.exists(edited_video_path) and os.path.getsize(edited_video_path) > 0 :
                print(f"[{job_name}] Silence removal successful. Using edited video.")
                intended_final_path = edited_video_path
                # Delete the raw downloaded file now that edited version is confirmed good
                try:
                    if current_video_path and os.path.exists(current_video_path): # Check if path is valid
                         os.remove(current_video_path); print(f"[{job_name}] Removed raw downloaded file: {current_video_path}")
                         raw_downloaded_video_path = None # Clear variable since file is gone
                except OSError as e: print(f"Warning [{job_name}]: Failed to remove raw downloaded file {current_video_path}: {e}")
            else:
                print(f"[{job_name}] Silence removal failed or produced empty file. Using original raw video.")
                # Try to move the original raw video to its final raw path name
                try:
                    # Ensure output dir for raw video exists
                    os.makedirs(os.path.dirname(final_raw_video_path) or '.', exist_ok=True)
                    os.rename(current_video_path, final_raw_video_path)
                    print(f"[{job_name}] Renamed raw video to final raw path: {final_raw_video_path}")
                    intended_final_path = final_raw_video_path
                    raw_downloaded_video_path = None # Clear variable since file is moved/renamed
                except OSError as e:
                     print(f"Warning [{job_name}]: Failed to rename raw video: {e}. Using temp path: {current_video_path}")
                     intended_final_path = current_video_path
                     # Keep raw_downloaded_video_path set, cleanup will handle later if it wasn't moved
        else:
            print(f"[{job_name}] Skipping silence removal step. Using raw video.")
            # Try to move the original raw video to its final raw path name
            try:
                 # Ensure output dir for raw video exists
                os.makedirs(os.path.dirname(final_raw_video_path) or '.', exist_ok=True)
                os.rename(current_video_path, final_raw_video_path)
                print(f"[{job_name}] Renamed raw video to final raw path: {final_raw_video_path}")
                intended_final_path = final_raw_video_path
                raw_downloaded_video_path = None # Clear variable since file is moved/renamed
            except OSError as e:
                 print(f"Warning [{job_name}]: Failed to rename raw video: {e}. Using temp path: {current_video_path}")
                 intended_final_path = current_video_path
                 # Keep raw_downloaded_video_path set, cleanup will handle later if it wasn't moved

        # === Crucial Check ===
        if not intended_final_path or not os.path.exists(intended_final_path) or os.path.getsize(intended_final_path) == 0:
            print(f"ERROR [{job_name}]: Failed to determine valid base video path after finalization.")
            print(f"  Intended path: '{intended_final_path}'")
            print(f"  Check logs for download/rename/silence removal errors.")
            # If original download still exists, maybe keep it? Check raw_downloaded_video_path
            if raw_downloaded_video_path and os.path.exists(raw_downloaded_video_path):
                 print(f"  Original downloaded file still exists at: {raw_downloaded_video_path}")
            last_error_message = "Failed to determine valid base video path after finalization"
            return False, last_error_message

        print(f"[{job_name}] Base video path set to: {intended_final_path}")
        print(f"[{job_name}] Step 6 completed in {time.time() - step_start_time:.2f}s"); step_start_time = time.time()
        final_output_path = intended_final_path

        # --- Step 7: Randomization (Optional) --- << NEW STEP POSITION
        print(f"\n--- [{job_name}] Step 7: Randomization (Optional) ---")
        progress_callback(step, total_steps, steps[step])
        if use_randomization:
            step += 1
        else:
            step += 2
        # The input to this step is the path determined by Step 6
        path_before_randomization = intended_final_path
        path_after_randomization = path_before_randomization  # Default to previous path if randomization skipped/fails
        applied_randomization_settings = None  # Initialize log variable

        # Check the 'use_randomization' flag passed into this function
        if use_randomization:
            print(
                f"[{job_name}] Randomization enabled. Attempting (Intensity: {randomization_intensity}) on: {path_before_randomization}")
            if progress_callback:
                progress_callback(step, total_steps, steps[step])
                step += 1
            # Ensure the base path for output (directory part) exists
            # 'output_file_base' should have been defined earlier based on output dir and job name
            os.makedirs(os.path.dirname(output_file_base), exist_ok=True)

            randomization_log_path = str(WORKING_DIR)
            # --- Call the imported randomize_video function ---
            # It expects input path, base for output names, and intensity.
            # It returns the path to the new video and a dictionary of applied settings.
            randomized_path_output, applied_randomization_settings = randomize_video(
                input_path=path_before_randomization,
                output_base_path=output_file_base,
                working_dir=WORKING_DIR,
                intensity=randomization_intensity,
                randomization_log_path=randomization_log_path,
            )

            # Already assigned above from return value, so this line is now unnecessary
            # You can just delete that old assignment

            # --- Check if randomization was successful ---
            if randomized_path_output and os.path.exists(randomized_path_output) and os.path.getsize(
                    randomized_path_output) > 0:
                print(f"[{job_name}] Randomization successful. Path is now: {randomized_path_output}")
                path_after_randomization = randomized_path_output  # Update the path for the next step

                # If successful, delete the input file to randomization (the pre-randomized version)
                if path_before_randomization != path_after_randomization:  # Safety check
                    try:
                        print(f"[{job_name}] Removing pre-randomization file: {path_before_randomization}")
                        os.remove(path_before_randomization)
                    except OSError as e:
                        print(
                            f"Warning [{job_name}]: Failed to remove pre-randomization file {path_before_randomization}: {e}")
            else:
                # Randomization function failed or produced an empty file
                print(f"Warning [{job_name}]: Randomization failed or produced empty file.")
                # The path variable 'path_after_randomization' still holds the previous path (fallback)
                print(f"[{job_name}] Using non-randomized video path for subsequent steps: {path_after_randomization}")
                # Try to clean up the failed/empty randomized file if it exists
                if randomized_path_output and os.path.exists(randomized_path_output):
                    try:
                        os.remove(randomized_path_output)
                    except OSError:
                        pass
                # Update log status if we got a log dictionary back
                if applied_randomization_settings:
                    applied_randomization_settings["status"] = "failed_fallback"

        else:
            # Randomization was disabled by the 'use_randomization' flag for this job
            print(f"[{job_name}] Skipping randomization (use_randomization is False).")
            # 'path_after_randomization' correctly holds the input path already

        # --- Sanity Check (Make sure we have a valid video file before proceeding) ---
        if not path_after_randomization or not os.path.exists(path_after_randomization) or os.path.getsize(
                path_after_randomization) == 0:
            print(
                f"ERROR [{job_name}]: Video path is invalid after Step 7 (Randomization): '{path_after_randomization}'")
            last_error_message = f"Video path is invalid after Step 7 (Randomization): '{path_after_randomization}'"
            # Log failure details if possible before returning
            json_output_folder = os.path.join(OUTPUT_BASE_DIR, "json")
            os.makedirs(json_output_folder, exist_ok=True)
            randomization_log_path = os.path.join(json_output_folder, f"{job_name}_{run_uuid}_randomizations.json")

            if applied_randomization_settings and randomization_log_path:
                try:
                    applied_randomization_settings["status"] = "job_failed"
                    applied_randomization_settings["job_error"] = "Video missing or empty after randomization step"
                    # Ensure directory exists before writing log
                    os.makedirs(os.path.dirname(randomization_log_path), exist_ok=True)
                    with open(randomization_log_path, 'w') as f:
                        json.dump(applied_randomization_settings, f, indent=4)
                except Exception as log_e:
                    print(f"Warning: Failed to write job failure log: {log_e}")
                    last_error_message = f"Video path is invalid after Step 7 (Randomization): '{path_after_randomization}'"
            return False, last_error_message # Fail the entire job

        # --- Log completion of this step ---
        current_step_time = time.time() - step_start_time  # Calculate time for this step
        print(f"[{job_name}] Step 7 completed in {current_step_time:.2f}s");
        step_start_time = time.time()  # Reset timer for the next step
        final_output_path = path_after_randomization

        # --- Step 8: Product Overlay (Optional) ---
        print(f"\n--- [{job_name}] Step 8: Product Overlay (Optional) ---")
        # Input path for this step is the result of Step 7 (Randomization)
        path_before_overlay = path_after_randomization  # Or whatever variable holds the correct path now

        # This variable will track the final path *resulting* from this step.
        final_output_path = path_before_overlay

        # === V1.20 Initialize variable BEFORE potential use ===
        product_clip_path = None  # Initialize here to ensure it always exists
        # === End Initialization ===

        # === V1.20/Phase 2 START: Get Main Video Dimensions ===
        main_video_width = None
        main_video_height = None
        print(f"[{job_name}] Attempting to get dimensions for main video: {path_before_overlay}")
        if path_before_overlay and os.path.exists(path_before_overlay):
            try:
                ffprobe_cmd = [
                    'ffprobe', '-v', 'error', '-select_streams', 'v:0',
                    '-show_entries', 'stream=width,height', '-of', 'json', path_before_overlay
                ]
                result = subprocess.run(ffprobe_cmd, capture_output=True, text=True, check=True, encoding='utf-8')
                output_data = json.loads(result.stdout)
                if output_data and 'streams' in output_data and len(output_data['streams']) > 0:
                    stream_data = output_data['streams'][0]
                    main_video_width = stream_data.get('width')
                    main_video_height = stream_data.get('height')
                    if isinstance(main_video_width, int) and isinstance(main_video_height, int):
                        print(f"[{job_name}] Found main video dimensions: {main_video_width}x{main_video_height}")
                    else:
                        print(f"WARNING [{job_name}]: ffprobe output missing width/height or they aren't integers.")
                        main_video_width, main_video_height = None, None  # Reset
                else:
                    print(f"WARNING [{job_name}]: ffprobe output does not contain expected streams data.")
            except FileNotFoundError:
                print(
                    f"ERROR [{job_name}]: ffprobe command not found. Ensure FFmpeg (which includes ffprobe) is installed and in your system's PATH.")
            except subprocess.CalledProcessError as e:
                print(
                    f"ERROR [{job_name}]: ffprobe command failed for '{path_before_overlay}'. Return Code: {e.returncode}")
                print(f"ffprobe stderr: {e.stderr}")
            except json.JSONDecodeError as e:
                print(f"ERROR [{job_name}]: Failed to parse ffprobe JSON output. Error: {e}")
                print(f"ffprobe stdout: {result.stdout if 'result' in locals() else 'N/A'}")
            except Exception as e:
                print(f"ERROR [{job_name}]: An unexpected error occurred getting video dimensions: {e}")
                traceback.print_exc()
                main_video_width, main_video_height = None, None  # Ensure None on error
        else:
            print(
                f"WARNING [{job_name}]: Cannot get dimensions, main video path is invalid or missing: {path_before_overlay}")
        # === V1.20/Phase 2 END: Get Main Video Dimensions ===

        # === V1.20 CHANGE START: Dynamic Product Clip Selection ===
        # This block now assumes product_clip_path exists (as None) and tries to assign a real path
        print(
            f"[{job_name}] Attempting to find product clips for product: '{product}' in base directory: '{product_clips_base_dir}'")
        if isinstance(product, str) and product and isinstance(product_clips_base_dir, str) and product_clips_base_dir:
            try:
                product_folder_path = os.path.join(product_clips_base_dir, product)
                print(f"[{job_name}] Constructed product folder path: {product_folder_path}")
                if os.path.isdir(product_folder_path):
                    print(f"[{job_name}] Searching for .mov files in: {product_folder_path}")
                    possible_clips = []
                    try:
                        for filename in os.listdir(product_folder_path):
                            if filename.lower().endswith(".mov"):
                                full_path = os.path.join(product_folder_path, filename)
                                possible_clips.append(full_path)
                    except OSError as list_err:
                        print(f"WARNING [{job_name}]: Error listing files in {product_folder_path}: {list_err}")

                    if possible_clips:
                        # Assign value to the pre-initialized product_clip_path
                        product_clip_path = random.choice(possible_clips)
                        print(f"[{job_name}] Randomly selected product clip: {product_clip_path}")
                    else:
                        print(
                            f"WARNING [{job_name}]: Product folder '{product_folder_path}' found, but no .mov files exist inside. Cannot apply overlay.")
                        # product_clip_path remains None
                else:
                    print(
                        f"WARNING [{job_name}]: Product folder not found: {product_folder_path}. Cannot apply overlay.")
                    # product_clip_path remains None
            except Exception as path_err:
                print(f"ERROR [{job_name}]: Failed during product clip path processing: {path_err}")
                traceback.print_exc()
                product_clip_path = None  # Reset to None on error
        else:
            print(
                f"WARNING [{job_name}]: Invalid 'product' or 'product_clips_base_dir'. Cannot determine product clip path.")
            product_clip_path = None  # Reset to None
        # === V1.20 CHANGE END ===

        # === Check if overlay is possible ===
        # Now this check can safely use product_clip_path because it was initialized earlier
        should_overlay = (
                path_before_overlay and os.path.exists(path_before_overlay)
                and product_clip_path and os.path.exists(
            product_clip_path)  # product_clip_path guaranteed to exist (as None or path)
                and main_video_width and main_video_height  # Check we got main dimensions too
        )

        # Initialize geometry variable
        calculated_geometry = None
        overlay_ready = False  # Flag to track if we have geometry needed for overlay

        if should_overlay:
            print(f"[{job_name}] Overlay possible. Proceeding with geometry calculation.")
            # --- Get Overlay Clip Aspect Ratio ---
            overlay_aspect_ratio = None
            try:
                print(f"[{job_name}] Getting dimensions for overlay clip: {product_clip_path}")
                ffprobe_cmd_clip = [
                    'ffprobe', '-v', 'error', '-select_streams', 'v:0',
                    '-show_entries', 'stream=width,height', '-of', 'json', product_clip_path
                ]
                result_clip = subprocess.run(ffprobe_cmd_clip, capture_output=True, text=True, check=True,
                                             encoding='utf-8')
                output_data_clip = json.loads(result_clip.stdout)
                if output_data_clip and 'streams' in output_data_clip and len(output_data_clip['streams']) > 0:
                    stream_data_clip = output_data_clip['streams'][0]
                    overlay_w_orig = stream_data_clip.get('width')
                    overlay_h_orig = stream_data_clip.get('height')
                    if isinstance(overlay_w_orig, int) and isinstance(overlay_h_orig, int) and overlay_h_orig > 0:
                        overlay_aspect_ratio = overlay_w_orig / overlay_h_orig
                        print(
                            f"[{job_name}] Found overlay clip dimensions: {overlay_w_orig}x{overlay_h_orig}, Aspect Ratio: {overlay_aspect_ratio:.3f}")
                    else:
                        print(
                            f"WARNING [{job_name}]: ffprobe output for overlay clip missing width/height or height is zero.")
                else:
                    print(f"WARNING [{job_name}]: ffprobe output for overlay clip missing stream data.")
            except Exception as ff_err:
                print(f"ERROR [{job_name}]: Failed to get overlay clip dimensions or aspect ratio: {ff_err}")
                # overlay_aspect_ratio remains None

            # --- Determine Placement and Size ---
            if overlay_aspect_ratio:  # Only proceed if we got aspect ratio
                # Defaults and supported placements based on constraints
                selected_placement = "middle_left"
                relative_size = 0.4
                # Expand the list of supported placements to allow for more positioning options
                supported_placements = [
                    "top_left", "top_center", "top_right",
                    "middle_left", "middle_center", "middle_right",
                    "bottom_left", "bottom_center", "bottom_right"
                ]

                # Check overlay_settings from YAML
                if overlay_settings and isinstance(overlay_settings, dict):
                    print(f"[{job_name}] Using overlay_settings from job config.")
                    placements_list = overlay_settings.get('placements', [])
                    valid_user_placements = [p for p in placements_list if
                                             isinstance(p, str) and p in supported_placements]
                    size_range_list = overlay_settings.get('size_range', [])

                    if valid_user_placements:
                        # Use the first placement from the list (not random choice) to ensure consistent placement
                        selected_placement = valid_user_placements[0]
                        print(f"[{job_name}] Using specified placement from config: '{selected_placement}'")
                    else:
                        print(
                            f"WARNING [{job_name}]: No supported placements ({supported_placements}) found in overlay_settings: {placements_list}. Using default '{selected_placement}'.")

                    if isinstance(size_range_list, list) and len(size_range_list) == 2 and \
                            isinstance(size_range_list[0], (int, float)) and isinstance(size_range_list[1],
                                                                                        (int, float)) and \
                            0.05 < size_range_list[0] <= size_range_list[1] < 0.95:
                        # Use the maximum size in the range for larger overlays
                        relative_size = size_range_list[1]
                        print(
                            f"[{job_name}] Using size from YAML: placement='{selected_placement}', size={relative_size:.2f}")
                    else:
                        print(
                            f"WARNING [{job_name}]: Invalid 'size_range' in overlay_settings: {size_range_list}. Using default size {relative_size:.2f}.")
                else:
                    print(
                        f"[{job_name}] No valid 'overlay_settings' found in job config. Using defaults: placement='{selected_placement}', size={relative_size:.2f}")

                # --- Calculate Final Geometry using Helper Function ---
                calculated_geometry = calculate_overlay_geometry(
                    placement_str=selected_placement,
                    relative_size=relative_size,
                    main_w=main_video_width,
                    main_h=main_video_height,
                    overlay_aspect_ratio=overlay_aspect_ratio,
                    margin_percent=7  # Using 7% margin as discussed
                )

                if calculated_geometry:
                    overlay_ready = True
                    print(f"[{job_name}] Geometry calculated. Ready for overlay.")
                else:
                    print(f"ERROR [{job_name}]: Failed to calculate overlay geometry. Skipping overlay.")
            else:
                print(
                    f"WARNING [{job_name}]: Missing overlay aspect ratio. Cannot calculate geometry. Skipping overlay.")

            # --- Proceed ONLY if geometry was successfully calculated ---
            if overlay_ready:
                overlay_step_start_time = time.time()
                video_with_overlay_path = f"{output_file_base}_final_overlay.mp4"
                temp_audio_for_asr_filename = f"temp_asr_audio_{run_uuid}.aac"
                temp_audio_extracted = False
                start_time_asr = None
                end_time_asr = None
                overlay_success = False

                try:
                    # 1. Extract audio
                    print(f"[{job_name}] Extracting audio for timestamp analysis from: {path_before_overlay}")
                    extract_cmd = ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'warning', '-i', path_before_overlay,
                                   '-vn', '-acodec', 'copy', temp_audio_for_asr_filename]
                    extract_success, extract_err = run_ffmpeg_command(extract_cmd)
                    if not extract_success or not os.path.exists(temp_audio_for_asr_filename) or os.path.getsize(
                            temp_audio_for_asr_filename) == 0:
                        raise RuntimeError(f"Audio extraction failed: {extract_err}")
                    temp_audio_extracted = True
                    print(f"[{job_name}] Audio extracted to {temp_audio_for_asr_filename}")

                    # 2. Get timestamps using Whisper
                    keywords_to_use = trigger_keywords if trigger_keywords is not None else []
                    print(f"DEBUG [{job_name}]: Using trigger keywords for ASR from job config: {keywords_to_use}")
                    start_time_asr, end_time_asr = get_product_mention_times(
                        audio_path=temp_audio_for_asr_filename, trigger_keywords=keywords_to_use,
                        language=language, job_name=job_name
                    )

                    # 3. Perform overlay if times were found
                    if start_time_asr is not None and end_time_asr is not None:
                        os.makedirs(os.path.dirname(video_with_overlay_path), exist_ok=True)
                        print(f"[{job_name}] Attempting FFmpeg overlay using calculated geometry...")

                        # === V1.20/Phase 2 CHANGE D: Update call to overlay_product_video ===
                        print(f"[{job_name}] Calling overlay function with geometry: {calculated_geometry}")
                        overlay_success = overlay_product_video(
                            main_video_path=path_before_overlay, product_clip_path=product_clip_path,
                            start_time=start_time_asr, end_time=end_time_asr,
                            output_path=video_with_overlay_path,
                            overlay_x=calculated_geometry['x'], overlay_y=calculated_geometry['y'],
                            overlay_w=calculated_geometry['w'], overlay_h=calculated_geometry['h'],
                            job_name=job_name
                        )
                        # === End Update Call ===

                        if overlay_success and (not os.path.exists(video_with_overlay_path) or os.path.getsize(
                                video_with_overlay_path) == 0):
                            print(
                                f"ERROR [{job_name}]: overlay_product_video reported success, but output file missing or empty: {video_with_overlay_path}")
                            overlay_success = False
                    else:
                        print(
                            f"[{job_name}] Product keywords not found or timing invalid via ASR. Skipping FFmpeg overlay.")
                        overlay_success = False

                except Exception as e:
                    print(f"ERROR [{job_name}]: Failed during overlay processing step (audio/ASR/ffmpeg): {e}")
                    traceback.print_exc()
                    overlay_success = False
                finally:
                    if temp_audio_extracted and os.path.exists(temp_audio_for_asr_filename):
                        try:
                            os.remove(temp_audio_for_asr_filename); print(
                                f"[{job_name}] Cleaned up temp ASR audio file: {temp_audio_for_asr_filename}")
                        except OSError as e:
                            print(
                                f"Warning [{job_name}]: Failed to delete temp ASR audio {temp_audio_for_asr_filename}: {e}")

                # 4. Update final path variable based on overlay success
                if overlay_success:
                    print(f"[{job_name}] Overlay successful. Final video path updated to: {video_with_overlay_path}")
                    final_output_path = video_with_overlay_path
                    try:
                        print(f"[{job_name}] Removing intermediate video (pre-overlay): {path_before_overlay}")
                        os.remove(path_before_overlay)
                    except OSError as e:
                        print(
                            f"Warning [{job_name}]: Failed to remove intermediate video {path_before_overlay}: {e}. Both versions may exist.")
                else:
                    print(
                        f"Warning/Info [{job_name}]: Overlay failed or skipped. Final video path remains: {final_output_path}")

                print(
                    f"[{job_name}] Step 8 Sub-Process (Audio/ASR/FFmpeg) completed in {time.time() - overlay_step_start_time:.2f}s")
                # --- End of block that runs only if overlay_ready is True ---

        else:  # should_overlay was False initially
            # Logging for skipping overlay
            if not path_before_overlay or not os.path.exists(path_before_overlay):
                print(
                    f"[{job_name}] Skipping overlay because base video path is invalid or missing: {path_before_overlay}")
            elif not product_clip_path:  # Covers folder not found, no .mov files, errors, etc.
                print(f"[{job_name}] Skipping overlay because no valid product clip could be selected.")
            elif not os.path.exists(product_clip_path):
                print(
                    f"[{job_name}] Skipping overlay because selected product clip file does not exist: {product_clip_path}")
            elif not main_video_width or not main_video_height:
                print(f"[{job_name}] Skipping overlay because main video dimensions could not be determined.")
            else:  # Generic fallback if none of the specific reasons matched
                print(f"[{job_name}] Skipping overlay for an undetermined reason (should_overlay is False).")
            # General skip message was here - removed for more specific logging above

        # --- Step 8 Block Ends --- The rest of the function continues...

        # --- Final Check (uses final_output_path which might now be the _overlay path) ---
        print(f"[{job_name}] Performing final check on path: {final_output_path}")
        if not final_output_path or not os.path.exists(final_output_path) or os.path.getsize(final_output_path) == 0: # Fail the job
            print(f"ERROR [{job_name}]: Final video output path ('{final_output_path}') is missing or empty after all steps.")
            last_error_message = f"Final video output path ('{final_output_path}') is missing or empty after all steps"
            return False, last_error_message

        # Step 9: Upload to Drive (Placeholder - uses the potentially updated final_output_path)
        # Finalizing result
        if progress_callback:
            progress_callback(step, total_steps, steps[step])
            # step += 1
        print(f"\n--- [{job_name}] Step 9: Upload to Google Drive ---")
        print(f"[{job_name}] (Placeholder) Upload Final Video ({final_output_path}) to Drive.")

        job_duration = time.time() - job_start_time
        print(f"\n--- Job '{job_name}' SUCCESS in {job_duration:.2f} seconds ---")
        print(f"    Final video: {final_output_path}")
        return True, final_output_path

    except Exception as e:
        # Catch errors from steps 2-7 (before overlay attempt)
        print(f"\n--- Job '{job_name}' FAILED during main processing ---")
        print(f"Error: {e}")
        traceback.print_exc()
        last_error_message = f"Job '{job_name}' FAILED during main processing {e}"
        return False, last_error_message

    finally:
        # --- Cleanup ---
        print(f"--- [{job_name}] Final Cleanup ---")
        # Delete temporary local files if they still exist
    #    if 'temp_audio_filename' in locals() and temp_audio_filename and os.path.exists(temp_audio_filename):
    #         try: os.remove(temp_audio_filename); print(f"[{job_name}] Cleaned up: {temp_audio_filename}")
    #         except OSError as e: print(f"Warning [{job_name}]: Failed cleanup {temp_audio_filename}: {e}")
        # Check if raw_downloaded_video_path still exists (it might have been renamed/deleted in Step 6)
        if 'raw_downloaded_video_path' in locals() and raw_downloaded_video_path and os.path.exists(raw_downloaded_video_path):
             try: os.remove(raw_downloaded_video_path); print(f"[{job_name}] Cleaned up: {raw_downloaded_video_path}")
             except OSError as e: print(f"Warning [{job_name}]: Failed cleanup {raw_downloaded_video_path}: {e}")
        # Note: temp_audio_for_asr_filename is cleaned up within the overlay block

        # Always attempt to delete GCS files (if names were generated)
        # Check if variables exist before trying to use them in cleanup
        gcs_audio_blob_to_delete = locals().get('gcs_audio_blob_name')
        gcs_video_blob_to_delete = locals().get('gcs_video_blob_name')
        if gcs_audio_blob_to_delete:
            delete_from_gcs(gcs_bucket_name, gcs_audio_blob_to_delete)
        if gcs_video_blob_to_delete:
            delete_from_gcs(gcs_bucket_name, gcs_video_blob_to_delete)
        print(f"--- [{job_name}] Cleanup Complete ---")


# --- Main Execution Logic (for command-line usage) ---
def main():
    print(f"--- Starting AI Video Creator Script --- Version {SCRIPT_VERSION} ---")
    start_time = time.time()

    print("Loading environment variables from .env file...")
    if not load_dotenv():
        print("Warning: .env file not found or empty.")
    else:
        print("Environment variables loaded.")

    # --- Argument Parsing (Remains the same for single runs) ---
    parser = argparse.ArgumentParser(description="Generate AI Avatar Videos (Single Run).")
    parser.add_argument("--product", required=True, help="Name of the product")
    parser.add_argument("--persona", required=True, help="Description of the creator")
    parser.add_argument("--setting", required=True, help="Setting of the video")
    parser.add_argument("--emotion", required=True, help="Desired emotion")
    parser.add_argument("--hook", required=True, help="Guidance for the hook")
    parser.add_argument("--elevenlabs_voice_id", required=True, help="Voice ID from your ElevenLabs account")
    parser.add_argument("--avatar_video_path", required=True, help="Full path to the base avatar MP4 file")
    parser.add_argument("--example_script_file", required=True, help="Path to a text file containing the example script")
    parser.add_argument("--remove_silence", action='store_true', help="Enable silence removal editing using ffmpeg.")
    args = parser.parse_args()

    print("Input arguments parsed:"); print(f"  Product: {args.product}"); # ... (rest of prints) ...
    print(f"  Silence removal option: {'ENABLED' if args.remove_silence else 'DISABLED'}")

    # --- Get Config/Keys & Validate ---
    openai_api_key=os.getenv('OPENAI_API_KEY'); elevenlabs_api_key=os.getenv('ELEVENLABS_API_KEY'); dreamface_api_key=os.getenv('DREAMFACE_API_KEY'); gcs_bucket_name=os.getenv('GCS_BUCKET_NAME')
    if not all([openai_api_key, elevenlabs_api_key, dreamface_api_key, gcs_bucket_name]):
        print("Error: Required API keys/bucket name missing in environment variables or .env file."); return
    if not os.path.exists(args.avatar_video_path):
        print(f"Error: Avatar video file not found: {args.avatar_video_path}"); return
    if not os.path.exists(args.example_script_file):
        print(f"Error: Example script file not found: {args.example_script_file}"); return

    try:
        with open(args.example_script_file, 'r', encoding='utf-8') as f:
            example_script_content = f.read()
        print(f"Successfully read example script from: {args.example_script_file}")
        if len(example_script_content.strip()) < 50:
             print(f"Warning: Example script in {args.example_script_file} seems very short.")
    except Exception as e:
        print(f"Error reading example script file {args.example_script_file}: {e}")
        return

    # --- Call the Refactored Job Function ---
    # NOTE: When running via command line, product_clips_map defaults to None,
    # so the overlay step inside create_video_job will be skipped.
    success, final_path = create_video_job(
        product=args.product,
        persona=args.persona,
        setting=args.setting,
        emotion=args.emotion,
        hook=args.hook,
        elevenlabs_voice_id=args.elevenlabs_voice_id,
        avatar_video_path=args.avatar_video_path,
        example_script_content=example_script_content, # Pass content
        remove_silence=args.remove_silence,
        openai_api_key=openai_api_key,
        elevenlabs_api_key=elevenlabs_api_key,
        dreamface_api_key=dreamface_api_key,
        gcs_bucket_name=gcs_bucket_name,
        # product_clips_map=None, # Default is None, no need to pass explicitly
        job_name=f"SingleRun_{args.product}"
    )

    # --- Report Result ---
    total_time = time.time() - start_time
    print(f"\n--- Single Run Process finished in {total_time:.2f} seconds ---")
    if success and final_path:
        print(f"Final video is available at: {final_path}")
    else:
        print("Single run failed. Check logs above for details.")


if __name__ == "__main__":
    main()