# run_batch.py
import yaml
import os
import sys
import time
import argparse
import traceback
from datetime import datetime
from dotenv import load_dotenv

# Import the core function from our refactored script
# Ensure create_video.py is in the same directory or Python path
try:
    from create_video import create_video_job
    # You might want to also import SCRIPT_VERSION if needed for logging
    # from create_video import SCRIPT_VERSION as CV_SCRIPT_VERSION
except ImportError:
    print("ERROR: Could not import 'create_video_job' from create_video.py.")
    print("Ensure create_video.py is in the same directory or accessible in PYTHONPATH.")
    sys.exit(1)

# --- Product Greenscreen Clip Mapping ---
# Maps product names (must match 'product' in campaigns.yaml) to their video file paths

# === TEMPORARILY DISABLE OVERLAYS BY MAKING THE MAP EMPTY ===
PRODUCT_CLIPS_MAP = {}  # <-- Make sure this line is ACTIVE (not commented out)
# print("INFO: Product overlay feature temporarily DISABLED for this batch run.")

# === Your original map (now ACTIVE) ===
#PRODUCT_CLIPS_MAP = { # REMOVED '#'
#    "Shilajit Gummies": '/Users/jonnybrower/ai_video_creator/Product Clips/#Shilajit_Transparent_AE_8bit_3.mov', # REMOVED '#'
#    "Black Seed Oil": "/Users/jonnybrower/ai_video_creator/product_clips/#black_seed_oil_green.mp4", # REMOVED '#' (Only if you still need this)
#    # Add entries for ALL products you might want to overlay
#} # REMOVED '#'
#print(f"Loaded Product Clip Mappings for: {list(PRODUCT_CLIPS_MAP.keys())}") # REMOVED '#' (Optional: uncomment to see confirmation)

# --- Main Batch Function (Modified to handle new config fields) ---
def run_batch(config_path: str):
    """
    Reads a YAML config file and runs video creation jobs in batch.
    Handles new configuration fields for randomization and overlay control.
    """
    print(f"--- Starting Batch Video Creation [{datetime.now().isoformat()}] ---")
    print(f"Using configuration file: {config_path}")

    # --- Load Environment Variables for API Keys ---
    # ... (API key loading remains the same) ...
    print("Loading environment variables...")
    if not load_dotenv():
        print("Warning: .env file not found or empty. Relying on environment variables.")
    else:
        print("Environment variables loaded from .env file.")

    openai_api_key = os.getenv('OPENAI_API_KEY')
    elevenlabs_api_key = os.getenv('ELEVENLABS_API_KEY')
    dreamface_api_key = os.getenv('DREAMFACE_API_KEY')
    gcs_bucket_name = os.getenv('GCS_BUCKET_NAME')

    if not all([openai_api_key, elevenlabs_api_key, dreamface_api_key, gcs_bucket_name]):
        print("\nERROR: Required API keys/bucket name missing.")
        sys.exit(1)
    print("API keys and GCS bucket name loaded.")

    # --- Load YAML Configuration ---
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        if config is None or 'jobs' not in config or not isinstance(config['jobs'], list):
             print(f"\nERROR: YAML file '{config_path}' is empty, invalid, or missing 'jobs' list.")
             sys.exit(1)
        jobs = config.get('jobs', []) # Get jobs list safely
        # --- Load Global Config (Defaults) ---
        global_config = config.get('global_config', {}) # Get global config safely, default to empty dict
        default_overlay_positions_path = global_config.get('default_overlay_positions_path', "configs/overlay_positions_default.yaml") # Default path if not in global
        # Note: product_clips_base_dir is now per-job

        print(f"Loaded {len(jobs)} job configurations from {config_path}")
        print(f"Default overlay positions path: {default_overlay_positions_path}")

    # ... (Error handling for YAML loading remains the same) ...
    except FileNotFoundError:
       print(f"\nERROR: Configuration file not found: {config_path}")
       sys.exit(1)
    except yaml.YAMLError as e:
       print(f"\nERROR: Failed to parse YAML file {config_path}: {e}")
       sys.exit(1)
    except Exception as e:
       print(f"\nERROR: An unexpected error occurred while loading config: {e}")
       sys.exit(1)


    # --- Process Jobs ---
    # ... (Summary counters remain the same) ...
    total_jobs = len(jobs)
    processed_count = 0
    success_count = 0
    failed_count = 0
    skipped_count = 0
    batch_start_time = time.time()

    for i, job_config in enumerate(jobs):
        job_index = i + 1
        job_name = job_config.get('job_name', f'Unnamed Job {job_index}')
        is_enabled = job_config.get('enabled', False)

        print(f"\n===== Processing Job {job_index}/{total_jobs}: '{job_name}' =====")

        if not is_enabled:
            print(f"SKIPPED: Job '{job_name}' is disabled.")
            skipped_count += 1
            continue

        processed_count += 1
        job_start_time = time.time()

        # --- Validate required parameters (excluding new optional ones for now) ---
        required_params = [
            'product', 'persona', 'setting', 'emotion', 'hook',
            'elevenlabs_voice_id', 'avatar_video_path', 'example_script_file',
            'language', 'brand_name' # remove_silence, enhance_for_elevenlabs checked below
        ]
        missing_params = [p for p in required_params if p not in job_config or job_config[p] is None]
        if missing_params:
            print(f"FAILED: Job '{job_name}' - Missing required parameters: {', '.join(missing_params)}")
            failed_count += 1
            continue

        # --- Extract existing parameters ---
        product = job_config['product']
        persona = job_config['persona']
        setting = job_config['setting']
        emotion = job_config['emotion']
        hook = job_config['hook']
        elevenlabs_voice_id = job_config['elevenlabs_voice_id']
        avatar_video_path = job_config['avatar_video_path']
        example_script_file = job_config['example_script_file']
        remove_silence = job_config.get('remove_silence', False) # Default false
        language = job_config['language']
        enhance_for_elevenlabs = job_config.get('enhance_for_elevenlabs', False) # Default false
        brand_name = job_config['brand_name']


        # --- Read Example Script Content ---
        # ... (Script reading logic remains the same) ...
        example_script_content = None
        try:
            print(f"Reading example script: {example_script_file}")
            with open(example_script_file, 'r', encoding='utf-8') as f:
                 example_script_content = f.read()
            if not example_script_content or example_script_content.isspace():
                 print(f"Warning: Example script file '{example_script_file}' is empty or whitespace only.")
            elif len(example_script_content) < 50:
                 print(f"Warning: Example script file '{example_script_file}' content seems very short.")
        except FileNotFoundError:
            print(f"FAILED: Job '{job_name}' - Example script file not found: {example_script_file}")
            failed_count += 1
            continue
        except Exception as e:
            print(f"FAILED: Job '{job_name}' - Error reading example script file {example_script_file}: {e}")
            failed_count += 1
            continue


        # --- Execute the Video Creation Job ---
        job_success = False
        final_path = None
        try:
            # Call the imported function with all necessary parameters,
            # including the NEW ones.
            job_success, final_path = create_video_job(
                # Existing parameters
                product=product,
                persona=persona,
                setting=setting,
                emotion=emotion,
                hook=hook,
                elevenlabs_voice_id=elevenlabs_voice_id,
                avatar_video_path=avatar_video_path,
                example_script_content=example_script_content,
                remove_silence=remove_silence,
                language=language,
                enhance_for_elevenlabs=enhance_for_elevenlabs,
                brand_name=brand_name,
                # API Keys / Config
                openai_api_key=openai_api_key,
                elevenlabs_api_key=elevenlabs_api_key,
                dreamface_api_key=dreamface_api_key,
                gcs_bucket_name=gcs_bucket_name,
                # product_clips_map=PRODUCT_CLIPS_MAP, # COMMENTED OUT - Superseded by new logic
                # Job Info
                job_name=job_name,
            ) # Keep the closing parenthesis

            if job_success:
                success_count += 1
                print(f"SUCCESS: Job '{job_name}' completed.")
                if final_path: print(f"  Output: {final_path}")
            else:
                failed_count += 1
                print(f"FAILED: Job '{job_name}' did not complete successfully (check logs above).")

        except Exception as e:
            # Catch unexpected errors during the call itself
            failed_count += 1
            print(f"\n--- CRITICAL FAILURE DURING JOB: '{job_name}' ---")
            print(f"An unexpected error occurred: {e}")
            traceback.print_exc()
            print("--------------------------------------------------")

        job_duration = time.time() - job_start_time
        print(f"===== Job '{job_name}' finished in {job_duration:.2f} seconds =====")


    # --- Batch Summary ---
    # ... (Summary printing remains the same) ...
    batch_duration = time.time() - batch_start_time
    print("\n\n--- Batch Processing Summary ---")
    print(f"Total job configurations found: {total_jobs}")
    print(f"Jobs processed: {processed_count}")
    print(f"Jobs skipped (disabled): {skipped_count}")
    print(f"Successful jobs: {success_count}")
    print(f"Failed jobs: {failed_count}")
    print(f"Total batch duration: {batch_duration:.2f} seconds")
    print("--------------------------------")


# --- Main execution block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AI Video Creator in batch mode using a YAML configuration file.")
    parser.add_argument(
        "-c", "--config",
        default="campaigns.yaml",
        help="Path to the YAML configuration file (default: campaigns.yaml)"
    )
    args = parser.parse_args()

    run_batch(args.config)