import os
import sys
import shutil
import yaml
import traceback
import uuid, threading, queue, json
from pathlib import Path
from flask import Flask, request, flash, abort
from flask import Response, jsonify
from datetime import datetime
from flask_cors import CORS
from dotenv import load_dotenv, set_key
from create_video import create_video_job

# ─── 1) Determine bundle vs. source mode ─────────────────────────────────────
if getattr(sys, "frozen", False):
    # PyInstaller bundle
    BASE_DIR = Path(sys._MEIPASS)
else:
    # Running from source
    BASE_DIR = Path(__file__).resolve().parent

# ─── 2) Prepare a user‐writable config directory ─────────────────────────────
CONFIG_DIR = Path.home() / ".zyra-video-agent"
CONFIG_DIR.mkdir(exist_ok=True)

# Paths for user config files
ENV_PATH       = CONFIG_DIR / ".env"
CAMPAIGNS_PATH = CONFIG_DIR / "campaigns.yaml"

# ─── Avatars config ────────────────────────────────────────────
AVATARS_PATH = CONFIG_DIR / "avatars.yaml"
AVATARS_DIR  = CONFIG_DIR / "uploads" / "avatars"
# ─── Scripts config ──────────────────────────────────────────────
SCRIPTS_PATH = CONFIG_DIR / "scripts.yaml"
SCRIPTS_DIR  = CONFIG_DIR / "uploads" / "scripts"

# ─── 3) Copy templates on first run ──────────────────────────────────────────
bundle_env       = BASE_DIR / ".env"
bundle_campaigns = BASE_DIR / "campaigns.yaml"

# Ensure files exist

if not ENV_PATH.exists():
    if bundle_env.exists():
        shutil.copy(bundle_env, ENV_PATH)
    else:
        ENV_PATH.write_text("")                    # create blank .env

if not CAMPAIGNS_PATH.exists():
    if bundle_campaigns.exists():
        shutil.copy(bundle_campaigns, CAMPAIGNS_PATH)
    else:
        CAMPAIGNS_PATH.write_text(yaml.safe_dump({"jobs": []})) # create empty campaigns.yaml

if not AVATARS_PATH.exists():
    AVATARS_PATH.write_text(yaml.safe_dump({"avatars": []}))

AVATARS_DIR.mkdir(parents=True, exist_ok=True)

if not SCRIPTS_PATH.exists():
    SCRIPTS_PATH.write_text(yaml.safe_dump({"scripts": []}))

SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

# ─── 4) Load environment variables from user .env ────────────────────────────
load_dotenv(dotenv_path=str(ENV_PATH))

# ─── 5) Initialize Flask ─────────────────────────────────────────────────────
app = Flask(__name__)
# TODO Allow your UI origins (you can also use "*" in dev)
CORS(app, resources={
  r"/*":     {"origins": "*"}
  # or simply r"/*": {"origins": "*"} for dev
})

app.secret_key = "secure-temporary-key"

app.job_queues  = {}   # job_id → queue.Queue()
app.job_results = {}   # job_id → {"success":…, "output_path":…}

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status":"ok"}), 200


# ─── Helpers to load & save data ─────────────────────────────────────────
def load_jobs():
    with open(CAMPAIGNS_PATH, "r") as f:
        data = yaml.safe_load(f) or {}
    return data.get("jobs", [])

def save_jobs(jobs):
    with open(CAMPAIGNS_PATH, "w") as f:
        yaml.safe_dump({"jobs": jobs}, f)

def load_avatars():
    with open(AVATARS_PATH, "r") as f:
        data = yaml.safe_load(f) or {}
    return data.get("avatars", [])

def save_avatars(lst):
    with open(AVATARS_PATH, "w") as f:
        yaml.safe_dump({"avatars": lst}, f)

def load_scripts():
    with open(SCRIPTS_PATH, "r") as f:
        data = yaml.safe_load(f) or {}
    return data.get("scripts", [])

def save_scripts(lst):
    with open(SCRIPTS_PATH, "w") as f:
        yaml.safe_dump({"scripts": lst}, f)

# ─── Campaigns Management ────────────────────────────────────
@app.route("/campaigns", methods=["GET"])
def get_campaigns():
    """
    Returns all campaigns as JSON:
    {
      "jobs": [ { job_name, product, persona, … }, … ]
    }
    """
    return jsonify({"jobs": load_jobs()})

@app.route("/campaigns", methods=["POST"])
def add_campaign():
    """
    Create a new campaign. Expects a JSON body, for example:
    {
      "job_name": "My Campaign",
      "product": "Gadget X",
      "persona": "Tech Reviewer",
      "setting": "Studio",
      "emotion": "Enthusiastic",
      "hook": "Unboxing!",
      "elevenlabs_voice_id": "voice-id-123",
      "language": "en",
      "avatar_video_path": "/full/path/to/avatar.mp4",
      "example_script_file": "/full/path/to/script.txt",
      "brand_name": "MyBrand",                 # optional
      "remove_silence": true,                  # optional
      "enhance_for_elevenlabs": false,         # optional
      "output_path": "/full/path/to/output.mp4" # optional
    }
    """

    # Detect JSON vs form-data
    if request.is_json:
        data = request.get_json()
    else:
        data = request.form

    # 1) Required fields
    required = [
        "job_name", "product", "persona", "setting",
        "emotion", "hook", "elevenlabs_voice_id",
        "language", "avatar_video_path", "avatar_id",
        "example_script_file", "script_id"
    ]
    missing = [f for f in required if not data.get(f)]
    if missing:
        return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

    # 2) Build job dict
    job = {k: data[k] for k in required}
    job["brand_name"] = data.get("brand_name", "")
    job["remove_silence"] = bool(data.get("remove_silence"))
    job["enhance_for_elevenlabs"] = bool(data.get("enhance_for_elevenlabs"))
    job["enabled"] = True

    # 3) Metadata
    job["id"] = uuid.uuid4().hex
    job["created_at"] = datetime.now().isoformat()
    job["output_path"] = data.get("output_path")

    # 4) Persist
    jobs = load_jobs()
    jobs.append(job)
    save_jobs(jobs)

    # 5) Return the new object
    return jsonify(job), 201

@app.route("/campaigns/<campaign_id>", methods=["PUT"])
def edit_campaign(campaign_id):
    """
    Update fields of the campaign with the given id.
    Accepts JSON body with any of:
      - job_name
      - product
      - persona
      - setting
      - emotion
      - hook
      - elevenlabs_voice_id
      - language
      - brand_name
      - remove_silence (boolean)
      - enhance_for_elevenlabs (boolean)
      - output_path
      - enabled (boolean)
    """
    # 1) Load existing campaigns
    jobs = load_jobs()

    # 2) Find the one to update
    for i, job in enumerate(jobs):
        if job.get("id") == campaign_id:
            data = request.get_json(force=True)
            if not data:
                return jsonify({"error": "Invalid JSON payload"}), 400

            # 3) Apply allowed updates
            for field in [
                "job_name", "product", "persona", "setting", "emotion", "hook",
                "elevenlabs_voice_id", "language", "brand_name",
                "remove_silence", "enhance_for_elevenlabs",
                "avatar_video_path", "avatar_id",
                "example_script_file", "script_id",
                "output_path", "enabled"
            ]:
                if field in data:
                    job[field] = data[field]

            # 4) Save back to campaigns.yaml
            jobs[i] = job
            save_jobs(jobs)

            # 5) Return the updated object
            return jsonify(job), 200

    # 6) If not found
    abort(404, description=f"Campaign ID '{campaign_id}' not found")

@app.route("/campaigns/<campaign_id>", methods=["DELETE"])
def delete_campaign(campaign_id):
    """
    Delete the campaign with the given id from campaigns.yaml.
    """
    # 1) Load all jobs
    jobs = load_jobs()

    # 2) Filter out the one to delete by id
    new_jobs = [j for j in jobs if j.get("id") != campaign_id]
    if len(new_jobs) == len(jobs):
        # No campaign had that id
        abort(404, description=f"Campaign ID '{campaign_id}' not found")

    # 3) Persist updated list
    save_jobs(new_jobs)

    # 4) Return HTTP 204 No Content
    return "", 204


# ─── Route: Application Settings (API keys, bucket, secret) ──────────────────
@app.route("/api/settings", methods=["GET"])
def get_settings():
    keys = [
        "OPENAI_API_KEY",
        "ELEVENLABS_API_KEY",
        "DREAMFACE_API_KEY",
        "GCS_BUCKET_NAME",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "OUTPUT_PATH"
    ]
    current = {k: os.getenv(k, "") for k in keys}
    return jsonify(current)

@app.route("/api/settings", methods=["POST"])
def save_settings():
    keys = [
        "OPENAI_API_KEY",
        "ELEVENLABS_API_KEY",
        "DREAMFACE_API_KEY",
        "GCS_BUCKET_NAME",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "OUTPUT_PATH"
    ]
    for k in keys:
        val = request.form.get(k, "").strip()
        set_key(str(ENV_PATH), k, val)
    # Reload updated vars immediately
    load_dotenv(dotenv_path=str(ENV_PATH), override=True)
    flash("Settings saved.", "success")
    return jsonify({"success": True})

# ─── Route: Run Campaign ───────────────────────────────────────────
@app.route("/run-job", methods=["POST"])
def run_job():
    # 1) Read the campaign's ID
    campaign_id = request.form.get("campaign_id") or request.form.get("id")
    if not campaign_id:
        return jsonify({"error": "campaign_id is required"}), 400

    # 2) Lookup job
    job = next((j for j in load_jobs() if j["id"] == campaign_id), None)
    if not job:
        return jsonify({"error": "Campaign not found"}), 404

    # 3) Create an SSE queue
    run_id = str(uuid.uuid4())
    q = queue.Queue()
    app.job_queues[run_id] = q

    # 4) Launch background thread
    def _runner():
        try:
            # a) Read script
            script_path = Path(job["example_script_file"])
            example_script = script_path.read_text(encoding="utf-8")

            # b) Call the core function
            success, output_path = create_video_job(
                product                = job["product"],
                persona                = job["persona"],
                setting                = job["setting"],
                emotion                = job["emotion"],
                hook                   = job["hook"],
                elevenlabs_voice_id    = job["elevenlabs_voice_id"],
                avatar_video_path      = job["avatar_video_path"],
                example_script_content = example_script,
                remove_silence         = job.get("remove_silence", False),
                language               = job.get("language", "English"),
                enhance_for_elevenlabs = job.get("enhance_for_elevenlabs", False),
                brand_name             = job.get("brand_name", ""),
                openai_api_key         = os.getenv("OPENAI_API_KEY"),
                elevenlabs_api_key     = os.getenv("ELEVENLABS_API_KEY"),
                dreamface_api_key      = os.getenv("DREAMFACE_API_KEY"),
                gcs_bucket_name        = os.getenv("GCS_BUCKET_NAME"),
                job_name               = job["job_name"],
                output_path            = os.getenv("OUTPUT_PATH"),
                progress_callback      = lambda step, total, msg: q.put({
                    "type": "progress", "step": step, "total": total, "message": msg
                })
            )

            # c) If the function returned failure without exception
            if not success:
                # In case of error using the same string to path the error message
                error_message = output_path
                q.put({
                    "type": "error",
                    "message": f"Job failed without exception. Last error message: {error_message}"
                })
                app.job_results[run_id] = {"success": False, "error": "create_video_job returned False"}
                return

            # d) Signal success
            q.put({"type": "done", "success": True, "output_path": output_path})
            app.job_results[run_id] = {"success": True, "output_path": output_path}

        except Exception as e:
            # e) Catch and emit any unexpected exception
            err = str(e)
            q.put({"type": "error", "Job failed with exception, message": err})
            app.job_results[run_id] = {"success": False, "error": err}
            # No re-raise: we want the thread to exit gracefully

    threading.Thread(target=_runner, daemon=True).start()

    # 5) Return the run ID for the client to open /progress/<run_id>
    return jsonify({"run_id": run_id}), 200

@app.route("/progress/<run_id>")
def progress(run_id):
    def event_stream():
        q = app.job_queues.get(run_id)
        if not q:
            yield 'event: error\ndata: {"message":"Unknown run"}\n\n'
            return

        while True:
            msg = q.get()
            # Normalize event type
            et = msg.get("type", "progress")
            data = {k: v for k, v in msg.items() if k != "type"}
            yield f"event: {et}\ndata: {json.dumps(data)}\n\n"
            if et in ("done", "error"):
                break

    return Response(event_stream(), mimetype="text/event-stream")

# ─── GET /avatars ───────────────────────────────────────────────
@app.route("/avatars", methods=["GET"])
def get_avatars():
    return jsonify({"avatars": load_avatars()})


# ─── POST /avatars ──────────────────────────────────────────────
@app.route("/avatars", methods=["POST"])
def add_avatar():
    # Required fields
    name               = request.form.get("name", "").strip()
    gender             = request.form.get("gender", "").strip()
    eleven_id          = request.form.get("elevenlabs_voice_id", "").strip()
    origin_language    = request.form.get("origin_language", "").strip()

    if not name or not gender:
        return jsonify({"error": "Both name and gender are required"}), 400

    # Optional file upload
    avatar_file = request.files.get("avatar_file")
    if not avatar_file or not avatar_file.filename:
        return jsonify({"error": "Please upload an avatar file"}), 400

    # Save file
    dest = AVATARS_DIR / avatar_file.filename
    avatar_file.save(dest)

    avatar = {
        "id":                  uuid.uuid4().hex,
        "name":                name,
        "gender":              gender,
        "file_path":           str(dest),
        "elevenlabs_voice_id": eleven_id or None,
        "origin_language":     origin_language or None
    }

    lst = load_avatars()
    lst.append(avatar)
    save_avatars(lst)
    return jsonify(avatar), 201

# ─── DELETE /avatars/<id> ────────────────────────────────────────
@app.route("/avatars/<avatar_id>", methods=["DELETE"])
def delete_avatar(avatar_id):
    avatars = load_avatars()
    # 1) Find the avatar to delete
    avatar_to_delete = next((av for av in avatars if av["id"] == avatar_id), None)
    if not avatar_to_delete:
        abort(404, description=f"Avatar ID '{avatar_id}' not found")

    # 2) Delete the file from disk
    file_path = Path(avatar_to_delete.get("file_path", ""))
    try:
        if file_path.exists():
            file_path.unlink()
    except Exception as e:
        # Log or flash a warning, but continue with deletion
        app.logger.warning(f"Failed to delete avatar file '{file_path}': {e}")

    # 3) Remove from the list and persist
    remaining = [av for av in avatars if av["id"] != avatar_id]
    save_avatars(remaining)

    return "", 204

# ─── PUT /avatars/<id> ───────────────────────────────────────────
@app.route("/avatars/<avatar_id>", methods=["PUT"])
def edit_avatar(avatar_id):
    avatars = load_avatars()
    for i, av in enumerate(avatars):
        if av["id"] == avatar_id:
            data = request.form.to_dict()  # to get form fields
            # Update text fields if present
            for fld in ["name", "gender", "elevenlabs_voice_id", "origin_language"]:
                if fld in data:
                    av[fld] = data[fld].strip()

            # Handle optional new file
            new_file = request.files.get("avatar_file")
            if new_file and new_file.filename:
                dest = AVATARS_DIR / new_file.filename
                new_file.save(dest)
                av["file_path"] = str(dest)

            avatars[i] = av
            save_avatars(avatars)
            return jsonify(av), 200

    abort(404, description=f"Avatar ID '{avatar_id}' not found")

# ─── GET /scripts ─────────────────────────────────────────────────
@app.route("/scripts", methods=["GET"])
def get_scripts():
    """Return all scripts as JSON."""
    return jsonify({"scripts": load_scripts()})


# ─── POST /scripts ────────────────────────────────────────────────
@app.route("/scripts", methods=["POST"])
def add_script():
    """
    Upload a new script file and register its metadata:
      - name            (string form field)
      - script_file     (uploaded file)
    Auto-generates:
      - id              (uuid4 hex)
      - created_at      (ISO timestamp)
      - file_path       (absolute path on disk)
    """
    name = request.form.get("name", "").strip()
    if not name:
        return jsonify({"error": "Field 'name' is required"}), 400

    script_file = request.files.get("script_file")
    if not script_file or not script_file.filename:
        return jsonify({"error": "Please upload a script file"}), 400

    dest = SCRIPTS_DIR / script_file.filename
    script_file.save(dest)

    record = {
        "id":         uuid.uuid4().hex,
        "name":       name,
        "created_at": datetime.now().isoformat(),
        "file_path":  str(dest)
    }
    lst = load_scripts()
    lst.append(record)
    save_scripts(lst)
    return jsonify(record), 201


# ─── DELETE /scripts/<script_id> ───────────────────────────────────
@app.route("/scripts/<script_id>", methods=["DELETE"])
def delete_script(script_id):
    """Remove the script record and delete its file from disk."""
    scripts = load_scripts()
    rec = next((s for s in scripts if s["id"] == script_id), None)
    if not rec:
        abort(404, description=f"Script ID '{script_id}' not found")

    # Delete file on disk
    fp = Path(rec["file_path"])
    if fp.exists():
        try: fp.unlink()
        except Exception: app.logger.warning(f"Failed to delete script file {fp}")

    # Remove from list and persist
    remaining = [s for s in scripts if s["id"] != script_id]
    save_scripts(remaining)
    return "", 204


# ─── PUT /scripts/<script_id> ─────────────────────────────────────
@app.route("/scripts/<script_id>", methods=["PUT"])
def edit_script(script_id):
    """
    Update a script’s name or replace its file.
    Accepts multipart/form-data:
      - name (optional)
      - script_file (optional)
    """
    scripts = load_scripts()
    for i, rec in enumerate(scripts):
        if rec["id"] == script_id:
            # Update name if provided
            if "name" in request.form:
                rec["name"] = request.form["name"].strip()

            # Replace file if a new one is uploaded
            new_file = request.files.get("script_file")
            if new_file and new_file.filename:
                # delete old file
                old = Path(rec["file_path"])
                if old.exists():
                    try: old.unlink()
                    except Exception: pass
                # save new file
                dest = SCRIPTS_DIR / new_file.filename
                new_file.save(dest)
                rec["file_path"] = str(dest)

            scripts[i] = rec
            save_scripts(scripts)
            return jsonify(rec), 200

    abort(404, description=f"Script ID '{script_id}' not found")

# ─── Optional: allow direct `python app.py` for debugging ────────────────────
if __name__ == "__main__":
    port = int(os.getenv("VIDEO_AGENT_PORT", 2026))
    app.run(host="localhost", port=port, debug=True)
