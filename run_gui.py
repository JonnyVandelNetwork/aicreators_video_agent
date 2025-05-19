import os
import sys
import threading
import webbrowser
from app import app

# Read port from ENV if set, else default to 5056
PORT = int(os.getenv("VIDEO_AGENT_PORT", 5056))

def serve():
    # Turn off debug in the bundled exe
    app.run(host="localhost", port=PORT, debug=False)

if __name__ == "__main__":
    # Start the web server in a background thread
    thread = threading.Thread(target=serve, daemon=True)
    thread.start()

    # Open the user’s default browser to the UI
    webbrowser.open(f"http://localhost:{PORT}")

    # Keep the main thread alive until the server thread exits
    thread.join()
