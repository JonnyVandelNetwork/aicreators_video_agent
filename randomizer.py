# === START OF RANDOMIZER CODE BLOCK ===

import cv2
import numpy as np
import librosa
import soundfile as sf
import random
import os
import imageio_ffmpeg # For managing ffmpeg on different envs
import subprocess
import json
import traceback
import time
import shutil
import glob  # Added for clip variants later (good to have imports ready)
import yaml  # Added for loading overlay positions later
import math

# --- Helper Functions ---


def random_float(min_val, max_val):
    """Generate a random float between min_val and max_val."""
    return random.uniform(min_val, max_val)


def random_int(min_val, max_val):
    """Generate a random integer between min_val and max_val."""
    return random.randint(min_val, max_val)


# Removed should_apply as randomness is now handled inside specific effect checks based on prob


def generate_random_string(length=10):
    """Generates a random alphanumeric string."""
    import string

    return "".join(random.choices(string.ascii_letters + string.digits, k=length))


def run_ffmpeg_command(cmd_list):
    """Executes an FFmpeg command using subprocess."""
    try:
        print(f"Running FFmpeg command: {' '.join(cmd_list)}")
        process = subprocess.Popen(
            cmd_list,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="ignore",
        )
        stdout, stderr = process.communicate(timeout=300)  # Add timeout
        if process.returncode != 0:
            print("FFmpeg Error Output:")
            print(stderr)
            return False, stderr
        else:
            # print("FFmpeg Output:") # Optional: print stdout for debugging
            # print(stdout)
            return True, None
    except subprocess.TimeoutExpired:
        print("FFmpeg command timed out.")
        # Attempt to kill the process if it's still running
        if process and process.poll() is None:
            try:
                process.kill()
                print("Killed hung FFmpeg process.")
            except Exception as kill_e:
                print(f"Error trying to kill FFmpeg process: {kill_e}")
        return False, "FFmpeg command timed out"
    except FileNotFoundError:
        print(
            "ERROR: ffmpeg command not found. Make sure FFmpeg is installed and in your PATH."
        )
        return False, "ffmpeg not found"
    except Exception as e:
        print(f"Failed to run FFmpeg command: {e}")
        return False, str(e)


def apply_color_jitter(frame, hue=0.1, saturation=0.1):
    """Applies color jittering to a frame.
    
    Args:
        frame: BGR frame to jitter
        hue: Hue jitter intensity (0-1)
        saturation: Saturation jitter intensity (0-1)
        
    Returns:
        Jittered BGR frame
    """
    try:
        # Convert to HSV for easier color manipulation
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        # Saturation
        s_jitter = 1.0 + random_float(-saturation, saturation)  # Multiplier
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * s_jitter, 0, 255)
        
        # Hue (scale hue jitter relative to 180, max value in OpenCV HSV)
        h_jitter = random_float(-hue, hue) * 180.0  # Additive shift in degrees (0-180)
        hsv[:, :, 0] = (hsv[:, :, 0] + h_jitter) % 180  # Hue wraps around
        
        jittered_frame = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return jittered_frame
    except Exception as e:
        print(f"Error applying color jitter: {e}")
        return frame


# Replace the existing apply_random_lut function
def apply_random_lut(frame, gamma_value):
    """Applies a specific gamma value (simulates a consistent LUT)."""
    if gamma_value is None or gamma_value == 1.0:  # Skip if no gamma or gamma is 1
        return frame
    try:
        invGamma = 1.0 / gamma_value
        table = np.array(
            [((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]
        ).astype("uint8")
        return cv2.LUT(frame, table)
    except Exception as e:
        print(
            f"ERROR: Exception occurred applying gamma LUT (gamma={gamma_value}): {e}"
        )  # Changed message slightly
        traceback.print_exc()  # ADD THIS LINE
        return frame

        # Placeholder action: random gamma
        gamma = random_float(0.75, 1.25)
        invGamma = 1.0 / gamma
        table = np.array(
            [((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]
        ).astype("uint8")
        # print(f"Applying simulated LUT (gamma: {gamma:.2f})") # Optional debug
        return cv2.LUT(frame, table)
    except Exception as e:
        print(f"Error applying simulated LUT: {e}")
        return frame


# Replace the existing apply_chromatic_aberration function
def apply_chromatic_aberration(frame, r_shift_x, r_shift_y, b_shift_x, b_shift_y):
    """Applies pre-calculated chromatic aberration shifts."""
    if r_shift_x == 0 and r_shift_y == 0 and b_shift_x == 0 and b_shift_y == 0:
        return frame  # Skip if no shift
    try:
        rows, cols, _ = frame.shape
        aberrated_frame = frame.astype(np.float32)
        Mr = np.float32([[1, 0, r_shift_x], [0, 1, r_shift_y]])
        Mb = np.float32([[1, 0, b_shift_x], [0, 1, b_shift_y]])
        b, g, r = cv2.split(aberrated_frame)
        r_warped = cv2.warpAffine(r, Mr, (cols, rows), borderMode=cv2.BORDER_REPLICATE)
        b_warped = cv2.warpAffine(b, Mb, (cols, rows), borderMode=cv2.BORDER_REPLICATE)
        merged = cv2.merge((b_warped, g, r_warped))
        merged = np.clip(merged, 0, 255)
        return merged.astype(np.uint8)
    except Exception as e:
        print(f"Error applying chromatic aberration: {e}")
        return frame


# --- Visual Randomization Modules ---


# Function for consistent color shift per video
def apply_consistent_color_shift(frame, b_shift, c_mult, s_mult, h_shift_deg):
    """Applies a pre-calculated color shift consistently."""
    try:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        # Apply Contrast (multiplicative) then Brightness (additive)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * c_mult + b_shift * 255.0, 0, 255)
        # Apply Saturation (multiplicative)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * s_mult, 0, 255)
        # Apply Hue (additive, with wrap)
        hsv[:, :, 0] = (hsv[:, :, 0] + h_shift_deg) % 180
        shifted_frame = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return shifted_frame
    except Exception as e:
        print(f"Warning: Error during consistent color shift: {e}")
        return frame  # Return original frame on error


def apply_sharpen(frame, amount=1.0, kernel_size=5):
    """Applies unsharp masking."""
    if amount <= 0:
        return frame
    try:
        # Ensure kernel is odd and positive
        kernel_size = max(1, kernel_size if kernel_size % 2 != 0 else kernel_size + 1)
        blurred = cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)
        sharpened = cv2.addWeighted(frame, 1.0 + amount, blurred, -amount, 0)
        return sharpened
    except Exception as e:
        print(
            f"ERROR: Exception occurred during sharpen: {e}"
        )  # Changed Warning to ERROR
        traceback.print_exc()  # ADD THIS LINE to print detailed error info
        return frame  # Still return original frame for now


def apply_noise_overlay(frame, alpha, std_dev):
    """Overlays subtle Gaussian noise with consistent parameters."""
    if alpha <= 0 or std_dev <= 0:
        return frame
    try:
        noise = np.random.normal(0, std_dev, frame.shape).astype(np.float32)
        noisy_frame = frame.astype(np.float32) + noise * alpha
        noisy_frame = np.clip(noisy_frame, 0, 255)
        return noisy_frame.astype(np.uint8)
    except Exception as e:
        print(f"Warning: Error applying noise overlay: {e}")
        return frame


def apply_analog_grain(frame, alpha, scale):
    """Adds noise resembling grain with consistent parameters."""
    if alpha <= 0 or scale <= 0:
        return frame
    try:
        noise = np.random.normal(0, 15 * scale, frame.shape).astype(np.float32)
        grained_frame = np.clip(frame.astype(np.float32) + noise * alpha, 0, 255)
        return grained_frame.astype(np.uint8)
    except Exception as e:
        print(f"Warning: Error applying analog grain: {e}")
        return frame


# Function to apply pre-calculated smooth shake values
def apply_camera_shake_smooth(frame, dx, dy, angle, scale):
    """Applies a pre-calculated smooth transformation."""
    try:
        rows, cols, _ = frame.shape
        center = (cols // 2, rows // 2)
        M = cv2.getRotationMatrix2D(center, angle, scale)
        M[0, 2] += dx
        M[1, 2] += dy
        # Use BORDER_REFLECT_101 for potentially better edge handling on large transforms
        return cv2.warpAffine(
            frame, M, (cols, rows), borderMode=cv2.BORDER_REFLECT_101
        )  # <<< CHANGED BORDER MODE
    except Exception as e:
        print(
            f"ERROR: Exception occurred during smooth camera shake: {e}"
        )  # Changed Warning to ERROR
        traceback.print_exc()  # ADD THIS LINE
        return frame


# --- Visual Randomization Modules ---
# ... (other apply_ functions) ...


# Revised LAB shift function using uint8 path
def apply_lab_color_shift(frame, a_shift, b_shift):
    """
    Applies color shifts in the LAB color space working primarily with uint8.
    L channel (Lightness) is kept unchanged.
    a_shift affects the green-red axis.
    b_shift affects the blue-yellow axis.
    """
    # Check if shifts are large enough to warrant processing
    # (Using int comparison now as shifts will be rounded)
    if abs(round(a_shift)) < 1 and abs(round(b_shift)) < 1:
        return frame
    try:
        # Convert BGR uint8 -> LAB uint8
        # L=[0, 255], A=[0, 255], B=[0, 255] representing full range
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

        l_channel, a_channel, b_channel = cv2.split(lab)  # Channels are uint8

        # Apply shifts: Convert channel to int16 for safe addition/subtraction,
        # add rounded shift, clip result to uint8 range [0, 255], convert back to uint8
        a_shifted = np.clip(
            a_channel.astype(np.int16) + int(round(a_shift)), 0, 255
        ).astype(np.uint8)
        b_shifted = np.clip(
            b_channel.astype(np.int16) + int(round(b_shift)), 0, 255
        ).astype(np.uint8)

        # Merge uint8 channels back
        merged_lab = cv2.merge(
            (l_channel, a_shifted, b_shifted)
        )  # All channels are uint8

        # Convert LAB uint8 -> BGR uint8
        bgr_shifted = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)

        return bgr_shifted

    except Exception as e:
        print(
            f"Error applying LAB color shift [v2] (a_shift={a_shift:.1f}, b_shift={b_shift:.1f}): {e}"
        )
        # traceback.print_exc() # Uncomment for debugging
        return frame  # Return original frame on error


# --- Visual Randomization Modules ---
# ... (other apply_ functions like apply_lab_color_shift, apply_glow, etc.) ...

# --- Visual Randomization Modules ---
# ... (other apply_ functions like apply_lab_color_shift etc.) ...

# --- Visual Randomization Modules ---
# ... (other apply_ functions like apply_lab_color_shift etc.) ...

# --- Visual Randomization Modules ---
# ... (other apply_ functions like apply_lab_color_shift etc.) ...

# --- Visual Randomization Modules ---
# ... (other apply_ functions) ...


def apply_edge_glow(
    frame,
    alpha,
    grad_threshold,
    gain=1.0,
    frame_num=-1,
    pre_blur_ksize=7,
    sobel_ksize=3,
    falloff_power=0.75,
    max_distance=30,
):
    """
    Applies significantly improved edge glow using LAB color space and limited distance transform.
    - max_distance: Maximum pixel distance for glow effect (lower = tighter glow)
    """
    if alpha <= 0.01 or gain <= 0.01:
        return frame

    try:
        # Convert to LAB for better edge detection
        frame_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel = frame_lab[:, :, 0]  # L channel has better edge contrast

        # Pre-blur with specified kernel size
        l_blur = cv2.GaussianBlur(l_channel, (pre_blur_ksize, pre_blur_ksize), 0)

        # Calculate Gradient Magnitude using Sobel
        sobelx = cv2.Sobel(l_blur, cv2.CV_64F, 1, 0, ksize=sobel_ksize)
        sobely = cv2.Sobel(l_blur, cv2.CV_64F, 0, 1, ksize=sobel_ksize)
        gradient_magnitude = cv2.magnitude(sobelx, sobely)

        # Normalize and threshold
        edge_signal_norm = cv2.normalize(
            gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )
        _, edge_map = cv2.threshold(
            edge_signal_norm, grad_threshold, 255, cv2.THRESH_BINARY
        )

        # Clean up edges slightly with morphology
        kernel = np.ones((2, 2), np.uint8)
        edge_map = cv2.morphologyEx(edge_map, cv2.MORPH_CLOSE, kernel)

        edge_pixels = np.sum(edge_map > 0)
        if edge_pixels == 0:
            if frame_num % 100 == 0:
                print(
                    f" SobelGlow frame {frame_num}: No edges found (T={grad_threshold}). Skipping."
                )
            return frame

        # Create Distance Transform with strict maximum distance limit
        dist_transform = cv2.distanceTransform(255 - edge_map, cv2.DIST_L2, 5)
        # This line is critical - limit max distance for tighter glow
        if max_distance > 0:
            dist_transform = np.minimum(dist_transform, max_distance)

        cv2.normalize(dist_transform, dist_transform, 0.0, 1.0, cv2.NORM_MINMAX)
        falloff_map = 1.0 - dist_transform

        # Apply falloff power for intensity control
        if falloff_power != 1.0:
            epsilon = 1e-7
            falloff_map = np.power(falloff_map + epsilon, falloff_power)

        # Amplify with gain
        amplified_falloff = np.clip(falloff_map * gain, 0.0, 1.0)

        # Create slightly warm-tinted glow (subtle color tint)
        glow_layer = np.zeros_like(frame, dtype=np.float32)
        glow_layer[:, :, 0] = amplified_falloff * 0.9  # B - slightly reduced
        glow_layer[:, :, 1] = amplified_falloff * 1.0  # G
        glow_layer[:, :, 2] = amplified_falloff * 1.1  # R - slightly increased

        glow_layer_bgr = np.clip(glow_layer * 255.0, 0, 255).astype(np.uint8)

        # Debug print
        if frame_num % 30 == 0:
            max_falloff_val = np.max(amplified_falloff)
            print(
                f" SobelGlow frame {frame_num}: Edges={edge_pixels}, MaxFalloff={max_falloff_val:.2f}, Alpha={alpha:.2f}, Gain={gain:.1f}, Thresh={grad_threshold}"
            )

        # Apply Screen Blend
        frame_uint8 = frame.astype(np.uint8)
        inv_base = 255 - frame_uint8
        inv_glow = 255 - glow_layer_bgr
        screen_blend_float = (
            inv_base.astype(np.float32) * inv_glow.astype(np.float32)
        ) / 255.0
        screen_blend_clipped = np.clip(screen_blend_float, 0, 255)
        screen_result = (255 - screen_blend_clipped).astype(np.uint8)

        # Final Alpha Blend
        final_frame = cv2.addWeighted(
            frame_uint8, 1.0 - alpha, screen_result, alpha, 0.0
        )

        return final_frame

    except Exception as e:
        print(
            f"Error applying edge glow [v8 SobelScreen] (frame={frame_num}, alpha={alpha:.2f}, gain={gain:.1f}): {e}"
        )
        traceback.print_exc()
        return frame


# --- Visual Randomization Modules ---
# ... (other apply_ functions like apply_lab_color_shift, apply_edge_glow etc.) ...

# --- Visual Randomization Modules ---
# ... (other apply_ functions) ...


# REPLACE the previous apply_simulated_bloom function with this one:
def apply_difference_glow(
    frame,
    threshold_value=160,
    large_blur_ksize=101,
    alpha=0.5,
    glow_intensity=2.0,
    glow_color=(0.95, 1.0, 1.05),
    frame_num=-1,
):
    """
    Generates glow by blurring a bright mask and SUBTRACTING the original
    mask area, aiming to isolate the outward spread ('haze').

    Args:
        frame: BGR uint8 frame.
        threshold_value: Pixels brighter than this define bright areas (Try 140-170).
        large_blur_ksize: Gaussian blur kernel size for the main glow spread (odd, positive, large).
        alpha: Final opacity of the haze layer (0-1). Might need higher values now.
        glow_intensity: Multiplier for haze brightness before coloring.
        glow_color: Color tint for the haze (BGR float tuple).
        frame_num: Optional frame number for debugging.

    Returns:
        Frame with haze/glow applied (uint8).
    """
    if alpha <= 0.01 or glow_intensity <= 0.01:
        return frame
    try:
        # 1. Isolate bright areas
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, bright_mask = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)

        # Check if any bright pixels were found
        if np.sum(bright_mask) == 0:
            # if frame_num != -1 and frame_num % 100 == 0:
            #    print(f" DifferenceGlow frame {frame_num}: No pixels above threshold {threshold_value}. Skipping.")
            return frame

        # 2. Blur the bright mask significantly to get the full spread
        large_blur_ksize = max(1, large_blur_ksize // 2 * 2 + 1)
        glow_map = cv2.GaussianBlur(
            bright_mask, (large_blur_ksize, large_blur_ksize), 0
        )

        # 3. Isolate the 'haze' (spread) by subtracting the original bright core
        # Convert masks to float32 for subtraction to handle potential negative results before clipping
        glow_map_float = glow_map.astype(np.float32)
        bright_mask_float = bright_mask.astype(np.float32)

        # Subtract the original bright mask from the blurred version
        haze_map_float = glow_map_float - bright_mask_float

        # Clip negative results to zero. We only want the positive difference (the spread).
        haze_map_float = np.clip(
            haze_map_float, 0, 255
        )  # Result is still effectively in 0-255 range conceptually

        # Check if any significant haze remains after subtraction
        if np.sum(haze_map_float) < 255:  # Check if there's more than just noise left
            # if frame_num != -1 and frame_num % 100 == 0:
            #    print(f" DifferenceGlow frame {frame_num}: No significant haze left after subtraction. Skipping.")
            return frame

        # 4. Create Colored Haze Layer from the difference map
        # Normalize the haze map [0, 1] based on its potential max value (255)
        haze_map_normalized = haze_map_float / 255.0
        # Apply intensity multiplier
        haze_map_normalized = np.clip(haze_map_normalized * glow_intensity, 0.0, 1.0)

        # Create BGR layer and apply color tint
        haze_layer_colored = np.zeros_like(frame, dtype=np.float32)
        for i in range(3):
            haze_layer_colored[:, :, i] = haze_map_normalized * glow_color[i]
        haze_layer_colored = np.clip(haze_layer_colored, 0.0, 1.0)

        # 5. Composite Haze Layer onto Original using Alpha Blending
        frame_float = frame.astype(np.float32) / 255.0
        output_frame_float = cv2.addWeighted(
            frame_float, 1.0 - alpha, haze_layer_colored, alpha, 0.0
        )

        # Clip final result and convert back to uint8
        output_frame_float = np.clip(output_frame_float, 0.0, 1.0)
        final_frame_uint8 = (output_frame_float * 255).astype(np.uint8)

        # --- Optional Debug Saving ---
        if frame_num == 50:  # Keep debug frame number consistent for comparison
            cv2.imwrite(f"debug_frame_{frame_num}_diff_input.png", frame)
            cv2.imwrite(
                f"debug_frame_{frame_num}_diff_bright_mask.png", bright_mask
            )  # Original threshold
            cv2.imwrite(
                f"debug_frame_{frame_num}_diff_glow_map.png", glow_map
            )  # Blurred version
            haze_map_uint8 = haze_map_float.astype(
                np.uint8
            )  # Visualize the difference map
            cv2.imwrite(
                f"debug_frame_{frame_num}_diff_haze_map.png", haze_map_uint8
            )  # <<< IMPORTANT DEBUG IMAGE
            haze_layer_uint8 = (haze_layer_colored * 255).astype(
                np.uint8
            )  # Visualize colored haze layer
            cv2.imwrite(
                f"debug_frame_{frame_num}_diff_haze_layer.png", haze_layer_uint8
            )
            cv2.imwrite(f"debug_frame_{frame_num}_diff_final.png", final_frame_uint8)
            print(
                f"DEBUG DifferenceGlow: Saved intermediate images for frame {frame_num}"
            )
        # --- End Debug ---

        return final_frame_uint8

    except Exception as e:
        print(
            f"Error applying difference glow (frame={frame_num}, alpha={alpha:.2f}, thresh={threshold_value}): {e}"
        )
        traceback.print_exc()
        return frame
    

def apply_white_speckles(frame, density=0.001, intensity=255):
    h, w, _ = frame.shape
    num_speckles = int(h * w * density)

    # Choose random pixel positions
    xs = np.random.randint(0, w, num_speckles)
    ys = np.random.randint(0, h, num_speckles)

    # Draw speckles on a copy
    speckle_layer = frame.copy()
    for x, y in zip(xs, ys):
        speckle_layer[y, x] = [intensity, intensity, intensity]  # White pixel

    return speckle_layer


# Remember to:
# 1. Replace the old glow function with this apply_simulated_bloom.
# 2. Update the function call in process_video_frame_by_frame_randomized.
# 3. Update the config section key (e.g., to "simulated_bloom") and parameters (threshold, blur_ksize, alpha).

# --- Default Config and Profiles ---
# Base default config (corresponds to 'medium' intensity)
DEFAULT_CONFIG = {
    "visual": {
        "apply": True,
        # --- Other Effects (Keep defaults as you had them or adjust) ---
        "camera_shake": {"prob": 0.7, "max_dx": 2, "max_dy": 2},  # Example default
        "noise_overlay": {"prob": 0.8, "max_alpha": 0.05, "max_std_dev": 10},
        "color_jitter": {  # Using this key based on your previous DEFAULT_CONFIG structure
            "prob": 0.9,
            "brightness": 0.05,
            "contrast": 0.05,
            "saturation": 0.05,
            "hue": 0.01,
        },
        "lut_application": {  # Assuming gamma-based for default example
            "prob": 0.5,
            "min_gamma": 0.90,
            "max_gamma": 1.10,
            # Or uncomment lut_dir if using that primarily:
            # "lut_dir": "./luts",
        },
        "blur": {"prob": 0.3, "max_kernel_size": 3},  # Subtle blur default
        "analog_grain": {"prob": 0.6, "max_alpha": 0.04, "max_scale": 1.0},
        "chromatic_aberration": {"prob": 0.4, "max_shift": 2},
        # --- Remove old glow keys ---
        # "simulated_bloom": {...}, # REMOVED
        # --- ADD New Difference Glow Effect ---
        "difference_glow": {
            "prob": 0.6,  # Default probability
            "min_threshold": 160,  # Default threshold range
            "max_threshold": 190,
            "min_large_blur_ksize": 81,  # Default blur range for spread
            "max_large_blur_ksize": 131,
            "min_alpha": 0.4,  # Default alpha range for haze layer
            "max_alpha": 0.6,
            "min_intensity": 1.5,  # Default intensity for haze map
            "max_intensity": 2.5,
            # glow_color uses function default, not needed here unless overriding
        },
        # ... other visual effects defaults ...
    },
    "audio": {
        "apply": True,
        "eq_shift": {"prob": 0.8, "num_bands": 3, "max_gain_db": 1.5},
        # "reverb": { "prob": 0.4, "max_reverberance": 30, "max_room_scale": 20,}, # Keep commented if not default
        "pitch_shift": {"prob": 0.5, "max_semitones": 0.15},
        "time_stretch": {"prob": 0.5, "min_rate": 0.99, "max_rate": 1.01},
        "volume_change": {"prob": 0.9, "min_db": -1.0, "max_db": 0.5},
    },
    "metadata": {
        "apply": True,
        "strip_all": True,
        "add_random": {"prob": 0.5, "keys": ["title", "artist", "comment"]},
    },
    "encoding": {
        "apply": True,
        "crf_range": [23, 28],
        "preset": ["medium", "slow"],
        "tune": ["film", "animation", "grain", None],
        "audio_bitrate_range": [96, 160],
    },
    "output_options": {
        "temp_dir": "./temp_processing",
        "cleanup_temp": True,
    },
}


RANDOMIZATION_PROFILES = {
    "none": {  # Turn everything off
        "visual": {"apply": False},
        "audio": {"apply": False},
        "metadata": {"apply": False},
        "encoding": {"apply": False},
        "output_options": DEFAULT_CONFIG["output_options"].copy(),
    },  # <<< Note comma separating dictionary entries
    "low": {
        "visual": {
            "apply": True,
            "camera_shake": {
                "smooth_motion": True,
                "prob": 0.7,
                "num_waves": 3,
                "max_freq": 0.2,
                "max_amp_dx": 8,
                "max_amp_dy": 8,
                "max_amp_angle": 2,
                "max_amp_scale": 0.02,
            },
            "time_varying_lab_shift": {"prob": 0.6},
            "time_varying_color_shift": {"prob": 0.6},
            "lut_application": {"prob": 0.5, "min_gamma": 0.95, "max_gamma": 1.05},
            "noise_overlay": {"prob": 0.6, "max_alpha": 0.05, "max_std_dev": 10},
            "analog_grain": {
                "prob": 1.0,
                "min_alpha": 0.05,
                "max_alpha": 0.09,
                "min_scale": 0.3,
                "max_scale": 0.6
            },
            "chromatic_aberration": {"prob": 0.5, "max_shift": 2},
            "sharpen": {
                "prob": 1.0,
                "min_amount": 0.8,
                "max_amount": 1.2,
                "min_kernel": 5,
                "max_kernel": 9
            },
            "difference_glow": {
                "prob": 0.4,
                "min_threshold": 160,
                "max_threshold": 190,
                "min_large_blur_ksize": 81,
                "max_large_blur_ksize": 121,
                "min_alpha": 0.3,
                "max_alpha": 0.5,
                "min_intensity": 1.2,
                "max_intensity": 2.0,
            },
            "white_speckles": {
                "enabled": False,
                "min_density": 0.005,
                "max_density": 0.01,
                "min_intensity": 250,
                "max_intensity": 255
            }
        },
        "audio": {
            "apply": True,
            "eq_shift": {"prob": 0.5},
            "pitch_shift": {"prob": 0.1},
            "time_stretch": {"prob": 0.1},
            "volume_change": {"prob": 0.4},
        },
        "metadata": {"apply": True, "strip_all": True, "add_random": {"prob": 1.0}},
        "encoding": {"apply": True, "crf_range": [23, 26], "preset": ["medium"]},
        "output_options": DEFAULT_CONFIG["output_options"].copy(),
    },
    # ==========================================================
    "medium": {  # Iteration 39 Base - Now Testing Difference Glow
        "visual": {
            "apply": True,
            # --- All other effects DISABLED (as per your Iteration 39 base) ---
            "camera_shake": {
                "smooth_motion": True,
                "prob": 1.0,
                "num_waves": 9,
                "max_freq": 0.6,
                "max_amp_dx": 60,
                "max_amp_dy": 30,
                "max_amp_angle": 15,
                "max_amp_scale": 0.25,
            },
            "time_varying_lab_shift": {"prob": 1.0},
            "time_varying_color_shift": {"prob": 1.0},
            "lut_application": {"prob": 1.0},
            "noise_overlay": {"prob": 1.0, "max_alpha": 0.1, "max_std_dev": 15},
            "analog_grain": {
                "prob": 1.0,
                "min_alpha": 0.05,
                "max_alpha": 0.09,
                "min_scale": 0.3,
                "max_scale": 0.6
            },
            "chromatic_aberration": {"prob": 1.0, "max_shift": 4},
            "sharpen": {
                "prob": 1.0,
                "min_amount": 0.8,
                "max_amount": 1.2,
                "min_kernel": 5,
                "max_kernel": 9
            },
            "glow": {"prob": 0.0},
            "edge_glow": {"prob": 0.0},
            "luminance_glow": {"prob": 0.0},
            "bright_edge_glow": {"prob": 0.0},
            "edge_glow_v2": {"prob": 0.0},
            "simulated_bloom": {"prob": 0.0},  # Disable previous attempt
            # --- Enable NEW Difference Glow ---
            "difference_glow": {
                "prob": 0.0,  # Activate this effect for testing
                # --- Parameters for Testing ---
                "min_threshold": 145,  # Lower threshold to define initial bright area
                "max_threshold": 175,
                "min_large_blur_ksize": 101,  # Large blur determines the spread
                "max_large_blur_ksize": 151,
                "min_alpha": 0.4,  # Opacity of the final haze layer
                "max_alpha": 0.7,
                "min_intensity": 1.8,  # Brightness multiplier for the haze map
                "max_intensity": 3.0,
                # glow_color uses function default
            },
            "white_speckles": {
                "enabled": False,
                "min_density": 0.005,
                "max_density": 0.01,
                "min_intensity": 250,
                "max_intensity": 255
            }
        },
        "audio": {
            "apply": True,
            "eq_shift": {"prob": 0.8},
            "pitch_shift": {"prob": 0.0},
            "time_stretch": {"prob": 0.0},
            "volume_change": {"prob": 0.7},
        },
        "metadata": {"apply": True, "strip_all": True, "add_random": {"prob": 1.0}},
        "encoding": {"apply": True, "crf_range": [24, 29], "preset": ["medium"]},
        "output_options": DEFAULT_CONFIG[
            "output_options"
        ].copy(),  # Inherit temp dir etc.
    },
    # End of "medium" profile (Testing Difference Glow)
    "high": {
        "visual": {
            "apply": True,
            "camera_shake": {
                "smooth_motion": True,
                "prob": 1.0,
                "num_waves": 12,
                "max_freq": 0.9,
                "max_amp_dx": 80,
                "max_amp_dy": 80,
                "max_amp_angle": 25,
                "max_amp_scale": 0.35
            },
            "time_varying_lab_shift": { "prob": 1.0 },
            "time_varying_color_shift": { "prob": 1.0 },
            "lut_application": { "prob": 1.0, "min_gamma": 0.85, "max_gamma": 1.25 },
            "noise_overlay": { "prob": 1.0, "max_alpha": 0.18, "max_std_dev": 30 },
            "analog_grain": {
                "prob": 1.0,
                "min_alpha": 0.05,
                "max_alpha": 0.09,
                "min_scale": 0.3,
                "max_scale": 0.6
            },
            "chromatic_aberration": { "prob": 1.0, "max_shift": 6 },
            "sharpen": {
                "prob": 1.0,
                "min_amount": 0.8,
                "max_amount": 1.4,
                "min_kernel": 5,
                "max_kernel": 13
            },
            "difference_glow": {
                "prob": 1.0,
                "min_threshold": 135,
                "max_threshold": 165,
                "min_large_blur_ksize": 111,
                "max_large_blur_ksize": 161,
                "min_alpha": 0.5,
                "max_alpha": 0.8,
                "min_intensity": 2.0,
                "max_intensity": 3.5
            },
            "white_speckles": {
                "enabled": False,
                "min_density": 0.005,
                "max_density": 0.01,
                "min_intensity": 250,
                "max_intensity": 255
            }
        },
        "audio": {
            "apply": True,
            "eq_shift": { "prob": 0.9 },
            "pitch_shift": { "prob": 0.3 },
            "time_stretch": { "prob": 0.3 },
            "volume_change": { "prob": 0.8 }
        },
        "metadata": { "apply": True, "strip_all": True, "add_random": { "prob": 1.0 } },
        "encoding": { "apply": True, "crf_range": [25, 30], "preset": ["medium", "slow"] },
        "output_options": DEFAULT_CONFIG["output_options"].copy()
    },
}  # <<< Final closing brace for the whole RANDOMIZATION_PROFILES dictionary


# --- Frame Processing Function (Re-integrating Standard Effects) ---
def process_video_frame_by_frame_randomized(
    input_video_path, temp_video_path, config, applied_effects_log, temp_dir
):
    """
    Processes video, applies effects based on profile config.
    Includes logic for standard effects (shake, noise, color, etc.)
    and the currently disabled difference_glow.
    Returns path, fps, log dict.
    """
    visual_cfg = config.get("visual", {})
    # Initialize log structure correctly
    visual_log = {
        "applied": visual_cfg.get("apply", False),
        "effects_applied_detail": {},
        "error": None,
        "total_frames": 0,
    }

    if not visual_log["applied"]:
        visual_log["error"] = "Processing function called when visual.apply was False"
        print("Visual processing skipped: visual.apply is False in config.")
        return None, None, visual_log

    # ==================================================================
    # === Generate Parameters/Waves Before Loop ===
    # ==================================================================
    print(f"DEBUG: Initializing visual effect parameters...")
    effect_params = {}  # Store consistent params chosen once per video
    wave_params = {}  # Store wave definitions for time-varying params

    # --- Initialize Standard Effects ---

    # Camera Shake (Assuming Smooth Shake variant based on profile structure)
    shake_cfg = visual_cfg.get("camera_shake", {})
    visual_log["effects_applied_detail"]["camera_shake"] = {
        "applied": False,
        "prob": shake_cfg.get("prob", 0),
    }
    # NOTE: Your provided 'medium' profile just had {"prob": 1.0}.
    # It needs full parameters if smooth_motion=True is intended.
    # Using example parameters here - ADJUST OR ENSURE they exist in your profile/DEFAULT_CONFIG.
    if shake_cfg.get("smooth_motion", True) and random.random() < shake_cfg.get(
        "prob", 0
    ):
        print("DEBUG: Initializing Smooth Camera Shake waves...")
        num_waves = shake_cfg.get("num_waves", 3)  # Example defaults if missing
        max_freq = shake_cfg.get("max_freq", 0.4)
        max_amp_dx = shake_cfg.get("max_amp_dx", 6)
        max_amp_dy = shake_cfg.get("max_amp_dy", 6)
        max_amp_angle = shake_cfg.get("max_amp_angle", 0.8)
        max_amp_scale = shake_cfg.get("max_amp_scale", 0.015)
        # --- Simple Sine Wave Generation (replace with your actual wave generation if different) ---
        current_waves = {}
        for axis in ["dx", "dy", "angle", "scale"]:
            current_waves[axis] = []
            max_amp = locals().get(
                f"max_amp_{axis}", 0
            )  # Get corresponding max amplitude
            for _ in range(num_waves):
                amplitude = random_float(0, max_amp / num_waves)  # Distribute amplitude
                frequency = random_float(0.05, max_freq)  # Hz
                phase = random_float(0, 2 * math.pi)
                current_waves[axis].append(
                    {"amp": amplitude, "freq": frequency, "phase": phase}
                )
        wave_params["smooth_shake"] = current_waves  # Store the generated waves

        # === Time-Varying HSV Color Shift ===
        tvc_cfg = visual_cfg.get("time_varying_color_shift", {})
        visual_log["effects_applied_detail"]["time_varying_color_shift"] = {
            "applied": False,
            "prob": tvc_cfg.get("prob", 0),
        }
        if random.random() < tvc_cfg.get("prob", 0):
            print("DEBUG: Initializing time-varying HSV color shift...")
            # Create simple sine wave for each channel
            wave_params["time_varying_color_shift"] = {
                "saturation_amp": random_float(0.02, 0.06),
                "hue_amp": random_float(1.0, 4.0),
                "sat_freq": random_float(0.05, 0.3),
                "hue_freq": random_float(0.05, 0.3),
                "sat_phase": random_float(0, 2 * math.pi),
                "hue_phase": random_float(0, 2 * math.pi),
            }
            visual_log["effects_applied_detail"]["time_varying_color_shift"][
                "applied"
            ] = True
            visual_log["effects_applied_detail"]["time_varying_color_shift"][
                "params"
            ] = wave_params["time_varying_color_shift"]

        # === Time-Varying LAB Color Shift ===
        tvl_cfg = visual_cfg.get("time_varying_lab_shift", {})
        visual_log["effects_applied_detail"]["time_varying_lab_shift"] = {
            "applied": False,
            "prob": tvl_cfg.get("prob", 0),
        }
        if random.random() < tvl_cfg.get("prob", 0):
            print("DEBUG: Initializing time-varying LAB color shift...")
            wave_params["time_varying_lab_shift"] = {
                "a_amp": random_float(2.0, 5.0),
                "b_amp": random_float(2.0, 5.0),
                "a_freq": random_float(0.03, 0.2),
                "b_freq": random_float(0.03, 0.2),
                "a_phase": random_float(0, 2 * math.pi),
                "b_phase": random_float(0, 2 * math.pi),
            }
            visual_log["effects_applied_detail"]["time_varying_lab_shift"][
                "applied"
            ] = True
            visual_log["effects_applied_detail"]["time_varying_lab_shift"]["params"] = (
                wave_params["time_varying_lab_shift"]
            )

        # --- End Sine Wave Generation ---
        visual_log["effects_applied_detail"]["camera_shake"]["applied"] = True
        visual_log["effects_applied_detail"]["camera_shake"][
            "params_generated"
        ] = True  # Indicate waves were made
        visual_log["effects_applied_detail"]["camera_shake"]["type"] = "smooth"

    # Noise Overlay
    noise_cfg = visual_cfg.get("noise_overlay", {})
    visual_log["effects_applied_detail"]["noise_overlay"] = {
        "applied": False,
        "prob": noise_cfg.get("prob", 0),
    }
    if random.random() < noise_cfg.get("prob", 0):
        # Fetch ranges from config, fall back to DEFAULT_CONFIG if needed via .get
        alpha_val = random_float(
            0,
            noise_cfg.get(
                "max_alpha", DEFAULT_CONFIG["visual"]["noise_overlay"]["max_alpha"]
            ),
        )
        std_dev_val = random_float(
            0,
            noise_cfg.get(
                "max_std_dev", DEFAULT_CONFIG["visual"]["noise_overlay"]["max_std_dev"]
            ),
        )
        if alpha_val > 0.005 and std_dev_val > 0.5:
            effect_params["noise_overlay"] = {
                "alpha": alpha_val,
                "std_dev": std_dev_val,
            }
            visual_log["effects_applied_detail"]["noise_overlay"]["applied"] = True
            visual_log["effects_applied_detail"]["noise_overlay"]["params"] = (
                effect_params["noise_overlay"]
            )
            print(f"  Noise Overlay Initialized: {effect_params['noise_overlay']}")

    # LUT Application (Assuming gamma based on DEFAULT_CONFIG and your profile just having prob: 1.0)
    lut_cfg = visual_cfg.get("lut_application", {})
    visual_log["effects_applied_detail"]["lut_application"] = {
        "applied": False,
        "prob": lut_cfg.get("prob", 0),
    }
    if random.random() < lut_cfg.get("prob", 0):
        # Check if using gamma or LUT dir based on config/defaults
        if (
            "min_gamma" in lut_cfg
            or "min_gamma" in DEFAULT_CONFIG["visual"]["lut_application"]
        ):
            print("DEBUG: Initializing Gamma LUT...")
            min_g = lut_cfg.get(
                "min_gamma", DEFAULT_CONFIG["visual"]["lut_application"]["min_gamma"]
            )
            max_g = lut_cfg.get(
                "max_gamma", DEFAULT_CONFIG["visual"]["lut_application"]["max_gamma"]
            )
            gamma_val = random_float(min_g, max_g)
            if abs(gamma_val - 1.0) > 0.01:
                effect_params["lut_gamma"] = {"gamma_value": gamma_val}
                visual_log["effects_applied_detail"]["lut_application"][
                    "applied"
                ] = True
                visual_log["effects_applied_detail"]["lut_application"][
                    "type"
                ] = "gamma"
                visual_log["effects_applied_detail"]["lut_application"]["params"] = (
                    effect_params["lut_gamma"]
                )
                print(f"  Gamma LUT Initialized: {effect_params['lut_gamma']}")
        # Add logic here for file-based LUTs if 'lut_dir' is the primary method

    # Analog Grain (Updated for per-video randomization using ranges)
    grain_cfg = visual_cfg.get("analog_grain", {})
    visual_log["effects_applied_detail"]["analog_grain"] = {
        "applied": False,
        "prob": grain_cfg.get("prob", 0), # Log the probability from config
    }
    # Decide if the effect should be applied for this video based on probability
    if random.random() < grain_cfg.get("prob", 0):
        print("DEBUG: Initializing Randomized Analog Grain...")

        # Get min/max ranges from config (provide sensible defaults)
        min_a = grain_cfg.get("min_alpha", 0.05)
        max_a = grain_cfg.get("max_alpha", 0.09)
        min_s = grain_cfg.get("min_scale", 0.3)
        max_s = grain_cfg.get("max_scale", 0.6)

        # Choose random alpha and scale for THIS video within the ranges
        alpha_val = random.uniform(min_a, max_a)
        scale_val = random.uniform(min_s, max_s)

        # Apply the effect only if chosen values are significant enough
        if alpha_val > 0.005 and scale_val > 0.05:
            # Store the chosen randomized values in effect_params for use in the loop
            effect_params["analog_grain"] = {"alpha": alpha_val, "scale": scale_val}
            visual_log["effects_applied_detail"]["analog_grain"]["applied"] = True
            # Log the chosen parameters and the ranges used
            visual_log["effects_applied_detail"]["analog_grain"]["params"] = {
                "chosen_alpha": round(alpha_val, 4),
                "chosen_scale": round(scale_val, 4)
            }
            visual_log["effects_applied_detail"]["analog_grain"]["config_ranges"] = {
                 "min_alpha": min_a, "max_alpha": max_a,
                 "min_scale": min_s, "max_scale": max_s
            }
            print(f"  Analog Grain Initialized with Random Params: alpha={alpha_val:.4f}, scale={scale_val:.4f}")
        else:
            # Log if randomization resulted in values too low to apply
             visual_log["effects_applied_detail"]["analog_grain"]["applied"] = False # Explicitly set applied to false
             visual_log["effects_applied_detail"]["analog_grain"]["reason"] = "Chosen alpha or scale too low after randomization"
             print(f"DEBUG: Analog Grain skipped (randomized values too low: alpha={alpha_val:.4f}, scale={scale_val:.4f})")
    # else: # Optional: Log if effect skipped due to probability check
    #    visual_log["effects_applied_detail"]["analog_grain"]["applied"] = False
    #    visual_log["effects_applied_detail"]["analog_grain"]["reason"] = "Skipped by probability"

    # Chromatic Aberration
    ca_cfg = visual_cfg.get("chromatic_aberration", {})
    visual_log["effects_applied_detail"]["chromatic_aberration"] = {
        "applied": False,
        "prob": ca_cfg.get("prob", 0),
    }
    if random.random() < ca_cfg.get("prob", 0):
        print("DEBUG: Initializing Chromatic Aberration...")
        def_ca = DEFAULT_CONFIG["visual"]["chromatic_aberration"]
        max_s = ca_cfg.get("max_shift", def_ca["max_shift"])
        r_sx, r_sy = random_float(-max_s, max_s), random_float(-max_s, max_s)
        b_sx, b_sy = random_float(-max_s, max_s), random_float(-max_s, max_s)
        if abs(r_sx) > 0.1 or abs(r_sy) > 0.1 or abs(b_sx) > 0.1 or abs(b_sy) > 0.1:
            effect_params["chromatic_aberration"] = {
                "r_shift_x": r_sx,
                "r_shift_y": r_sy,
                "b_shift_x": b_sx,
                "b_shift_y": b_sy,
            }
            visual_log["effects_applied_detail"]["chromatic_aberration"][
                "applied"
            ] = True
            visual_log["effects_applied_detail"]["chromatic_aberration"]["params"] = (
                effect_params["chromatic_aberration"]
            )
            print(
                f"  Chromatic Aberration Initialized: {effect_params['chromatic_aberration']}"
            )

    # Sharpen (Updated for min/max amount and optional min/max kernel)
    sharpen_cfg = visual_cfg.get("sharpen", {})
    visual_log["effects_applied_detail"]["sharpen"] = {
        "applied": False,
        "prob": sharpen_cfg.get("prob", 0),
    }
    # Decide if sharpen should run based on probability
    if random.random() < sharpen_cfg.get("prob", 0):
        print("DEBUG: Initializing Randomized Sharpen...")

        # Get min/max amount ranges from config (provide sensible defaults)
        min_a = sharpen_cfg.get("min_amount", 0.1)  # Default min > 0 if not set
        max_a = sharpen_cfg.get("max_amount", 0.8)  # Default max lowered
        # Clamp/validate ranges
        min_a = max(0.0, min_a)
        max_a = max(min_a, max_a) # Ensure max >= min

        # Get min/max kernel ranges (ensure kernel is odd and >= 3)
        min_k = sharpen_cfg.get("min_kernel", 5) # Default min kernel if not set
        max_k = sharpen_cfg.get("max_kernel", 9) # Default max kernel if not set
        min_k = max(3, min_k // 2 * 2 + 1) # Ensure min is odd >= 3
        max_k = max(min_k, max_k // 2 * 2 + 1) # Ensure max is odd >= min

        # Choose random amount and kernel size for THIS video
        amount_val = random.uniform(min_a, max_a)
        # Choose kernel size (fixed if min==max, random otherwise)
        if min_k == max_k:
            kernel_val = min_k
        else:
            # random.choice is good for picking from odd steps
            kernel_val = random.choice(range(min_k, max_k + 1, 2))

        # Apply only if amount is significant
        if amount_val > 0.01: # Use a small threshold
            # Store the chosen values
            effect_params["sharpen"] = {"amount": amount_val, "kernel_size": kernel_val}
            visual_log["effects_applied_detail"]["sharpen"]["applied"] = True
            # Log chosen parameters and ranges
            visual_log["effects_applied_detail"]["sharpen"]["params"] = {
                "chosen_amount": round(amount_val, 4),
                "chosen_kernel_size": kernel_val
            }
            visual_log["effects_applied_detail"]["sharpen"]["config_ranges"] = {
                 "min_amount": min_a, "max_amount": max_a,
                 "min_kernel": min_k, "max_kernel": max_k
            }
            print(f"  Sharpen Initialized: amount={amount_val:.4f}, kernel={kernel_val}")
        else:
            # Log if amount randomized too low
            visual_log["effects_applied_detail"]["sharpen"]["applied"] = False
            visual_log["effects_applied_detail"]["sharpen"]["reason"] = "Chosen amount too low after randomization"
            print(f"DEBUG: Sharpen skipped (randomized amount too low: {amount_val:.4f})")
    # else: # Optional logging if skipped by probability
    #    visual_log["effects_applied_detail"]["sharpen"]["applied"] = False
    #    visual_log["effects_applied_detail"]["sharpen"]["reason"] = "Skipped by probability"


    # Initialize White Speckles parameters ONCE per video call
    chosen_density = 0
    chosen_intensity = 0
    speckles_active_for_this_video = False # Flag to control application in the loop
    init_speckle_cfg = config.get("visual", {}).get("white_speckles", {})

    # Check if the effect is enabled in the config profile for potential use
    if init_speckle_cfg.get("enabled", False):
        speckles_active_for_this_video = True # Mark as active for the loop below

        # Get ranges from config (using defaults if keys are missing)
        min_d = init_speckle_cfg.get("min_density", 0.0007)
        max_d = init_speckle_cfg.get("max_density", 0.0015)
        min_i = init_speckle_cfg.get("min_intensity", 200)
        max_i = init_speckle_cfg.get("max_intensity", 255)

        # Generate random values specifically for THIS video
        chosen_density = random.uniform(min_d, max_d)
        chosen_intensity = random.randint(min_i, max_i)

        # Log the chosen values for debugging/confirmation for this video
        print(f"DEBUG: White Speckles enabled for this video (Density: {chosen_density:.5f}, Intensity: {chosen_intensity})")

        # Store details in the log for the final report
        visual_log["effects_applied_detail"]["white_speckles"] = {
            "applied": True,
            "chosen_density": round(chosen_density, 5),
            "chosen_intensity": chosen_intensity,
            "config_ranges": { # Log the ranges used for clarity
                "min_density": min_d, "max_density": max_d,
                "min_intensity": min_i, "max_intensity": max_i
                }
        }
    else:
        # Ensure log reflects disabled status if not enabled in config
        if "white_speckles" not in visual_log["effects_applied_detail"]: # Avoid overwriting if already logged somehow
            visual_log["effects_applied_detail"]["white_speckles"] = {
                "applied": False,
                "reason": "Not enabled in profile config"
            }

    # --- Initialize Difference Glow (Keep logic, but should be disabled by profile prob:0.0) ---
    diff_glow_cfg = visual_cfg.get("difference_glow", {})
    visual_log["effects_applied_detail"]["difference_glow"] = {
        "applied": False,
        "prob": diff_glow_cfg.get("prob", 0),
    }
    if random.random() < diff_glow_cfg.get("prob", 0):
        print(f"DEBUG: Initializing constant Difference Glow parameters...")
        thresh_val = random_int(
            diff_glow_cfg.get("min_threshold", 145),
            diff_glow_cfg.get("max_threshold", 175),
        )
        blur_k = random_int(
            diff_glow_cfg.get("min_large_blur_ksize", 101),
            diff_glow_cfg.get("max_large_blur_ksize", 151),
        )
        alpha_val = random_float(
            diff_glow_cfg.get("min_alpha", 0.4), diff_glow_cfg.get("max_alpha", 0.7)
        )
        intensity_val = random_float(
            diff_glow_cfg.get("min_intensity", 1.8),
            diff_glow_cfg.get("max_intensity", 3.0),
        )
        blur_k = max(1, blur_k // 2 * 2 + 1)
        if alpha_val > 0.01 and intensity_val > 0.01:
            effect_params["difference_glow"] = {
                "threshold_value": thresh_val,
                "large_blur_ksize": blur_k,
                "alpha": alpha_val,
                "glow_intensity": intensity_val,
            }
            visual_log["effects_applied_detail"]["difference_glow"]["applied"] = True
            visual_log["effects_applied_detail"]["difference_glow"]["params"] = (
                effect_params["difference_glow"]
            )
            print(
                f"  Difference Glow Initialized (but likely disabled by profile prob): {effect_params['difference_glow']}"
            )
        else:
            visual_log["effects_applied_detail"]["difference_glow"][
                "reason"
            ] = "alpha or intensity too low"
            print(f"DEBUG: Difference Glow skipped (alpha/intensity too low).")

    # --- End of Parameter Initialization ---
    # ==============================================================

    try:
        video_cap = cv2.VideoCapture(input_video_path)
        if not video_cap.isOpened():
            raise RuntimeError(f"Could not open video capture for {input_video_path}")
        fps = video_cap.get(cv2.CAP_PROP_FPS)
        fps = fps if fps > 0 else 30.0
        width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width == 0 or height == 0:
            raise RuntimeError(
                f"Could not get valid dimensions from {input_video_path}"
            )

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))
        if not video_writer.isOpened():
            raise RuntimeError(f"Could not open video writer for {temp_video_path}")

        frame_count = 0
        # Determine active effects for print message
        active_effects_list = [
            k
            for k, v in visual_log["effects_applied_detail"].items()
            if v.get("applied")
        ]
        print(
            f"Processing frames (Active Effects: {', '.join(active_effects_list) or 'None'}) at {fps:.2f} FPS..."
        )

        while True:
            ret, frame = video_cap.read()
            if not ret:
                break  # End of video
            processed_frame = frame.copy()
            current_time_sec = frame_count / fps  # Needed for smooth shake

            # Apply White Speckles (if flag was set to active before the loop)
            if speckles_active_for_this_video:
                # Use the density and intensity chosen ONCE before the loop started
                processed_frame = apply_white_speckles(processed_frame, chosen_density, chosen_intensity)

            # --- Apply Active Effects (Add Calls Back In) ---
            # Order can matter. Suggested order: Shake -> Color/LUT -> Glow -> Grain/Noise/Sharpen

            # Apply Camera Shake
            if wave_params.get("smooth_shake"):
                # --- Simple Sine Wave Calculation (replace if using different waves) ---
                dx, dy, angle, scale = 0.0, 0.0, 0.0, 1.0
                shake_waves = wave_params["smooth_shake"]
                temp_scale = 0.0  # Accumulate scale effect relative to 0
                for axis, waves in shake_waves.items():
                    total_offset = 0.0
                    for wave in waves:
                        total_offset += wave["amp"] * math.sin(
                            2 * math.pi * wave["freq"] * current_time_sec
                            + wave["phase"]
                        )
                    if axis == "dx":
                        dx = total_offset
                    elif axis == "dy":
                        dy = total_offset
                    elif axis == "angle":
                        angle = total_offset
                    elif axis == "scale":
                        temp_scale = total_offset
                scale = 1.0 + temp_scale  # Final scale is 1 + accumulated offset
                # --- End Sine Wave Calculation ---
                processed_frame = apply_camera_shake_smooth(
                    processed_frame, dx, dy, angle, scale
                )
                print(
                    f"[Shake] Frame {frame_count}: dx={dx:.2f}, dy={dy:.2f}, angle={angle:.2f}, scale={scale:.4f}"
                )

            # Time-Varying Color Shift
            if wave_params.get("time_varying_color_shift"):
                params = wave_params["time_varying_color_shift"]
                h = params["hue_amp"] * math.sin(
                    params["hue_freq"] * current_time_sec + params["hue_phase"]
                )
                s = 1.0 + params["saturation_amp"] * math.sin(
                    params["sat_freq"] * current_time_sec + params["sat_phase"]
                )
                processed_frame = apply_consistent_color_shift(
                    processed_frame, b_shift=0, c_mult=1.0, s_mult=s, h_shift_deg=h
                )

            # Time-Varying LAB Shift
            if wave_params.get("time_varying_lab_shift"):
                params = wave_params["time_varying_lab_shift"]
                a = params["a_amp"] * math.sin(
                    params["a_freq"] * current_time_sec + params["a_phase"]
                )
                b = params["b_amp"] * math.sin(
                    params["b_freq"] * current_time_sec + params["b_phase"]
                )
                processed_frame = apply_lab_color_shift(
                    processed_frame, a_shift=a, b_shift=b
                )

            # Apply LUT/Gamma
            if effect_params.get("lut_gamma"):
                processed_frame = apply_random_lut(
                    processed_frame, **effect_params["lut_gamma"]
                )
            # Add logic here for file-based LUTs

            # Apply Chromatic Aberration
            if effect_params.get("chromatic_aberration"):
                processed_frame = apply_chromatic_aberration(
                    processed_frame, **effect_params["chromatic_aberration"]
                )

            # Apply Difference Glow (if activated - should be off based on profile)
            # if effect_params.get("difference_glow"):
            #    processed_frame = apply_difference_glow(processed_frame,
            #                                          frame_num=frame_count,
            #                                          **effect_params["difference_glow"])

            # Apply Analog Grain
            if effect_params.get("analog_grain"):
                processed_frame = apply_analog_grain(
                    processed_frame, **effect_params["analog_grain"]
                )

            # Apply Noise Overlay
            if effect_params.get("noise_overlay"):
                processed_frame = apply_noise_overlay(
                    processed_frame, **effect_params["noise_overlay"]
                )

            # Apply Sharpen
            if effect_params.get("sharpen"):
                processed_frame = apply_sharpen(
                    processed_frame, **effect_params["sharpen"]
                )

            # --- Write Frame ---
            if frame_count == 50:  # Keep the single frame save
                debug_save_path = os.path.join(temp_dir, "debug_frame_50_final.png")
                cv2.imwrite(debug_save_path, processed_frame)
                print(f"DEBUG: Saved frame 50 to {debug_save_path}")

            # ADD THIS periodic check:
            if frame_count % 30 == 0:  # Print every 30 frames
                print(
                    f"Frame {frame_count}: Writing frame with shape {processed_frame.shape} and dtype {processed_frame.dtype}"
                )

            # Existing line below:
            video_writer.write(processed_frame)
            frame_count += 1

        # --- Cleanup after loop ---
        video_cap.release()
        video_writer.release()
        print(f"Finished processing {frame_count} frames visually.")
        visual_log["total_frames"] = frame_count

        # Update counts logging (simplified for effects active for all frames if applied)
        for effect_key, data in visual_log.get("effects_applied_detail", {}).items():
            if data.get("applied"):  # If params were successfully initialized
                data["applied_count"] = frame_count
                data["applied_ratio"] = 1.0
                if "reason" in data:
                    del data["reason"]  # Clean up reason if applied

        return temp_video_path, fps, visual_log

    except Exception as e:
        print(f"ERROR during visual frame processing loop: {e}")
        traceback.print_exc()
        visual_log["error"] = str(e)
        if "video_cap" in locals() and video_cap.isOpened():
            video_cap.release()
        if "video_writer" in locals() and video_writer.isOpened():
            try:
                video_writer.release()
                if os.path.exists(temp_video_path):
                    os.remove(temp_video_path)
                    print(f"Removed incomplete temp video file: {temp_video_path}")
            except Exception as release_e:
                print(
                    f"Warning: Error releasing video writer or removing temp file on error: {release_e}"
                )
        return None, None, visual_log


# Make sure all referenced apply_... functions are defined correctly in the file

# Make sure the apply_difference_glow function is defined correctly elsewhere in your file


# --- Audio Processing Function (Helper for randomize_video) ---
# Replace the existing function with this version
def apply_audio_effects_librosa_randomized(
    audio_path,  # Input is likely .aac
    output_base_path,  # Base path for naming temp files
    config,
    applied_effects_log,
):
    """
    Applies Librosa audio effects (pitch, time, volume), saves as WAV,
    and returns success status, the path to the processed audio (WAV or original AAC on failure),
    and a log dictionary.
    """
    # Define WAV output path based on the base path
    wav_output_path = f"{output_base_path}_rand_proc_audio.wav"  # << SAVE AS WAV

    audio_cfg = config.get("audio", {})
    # Initialize log for Librosa specific effects for this run
    audio_log = {
        "applied": audio_cfg.get("apply", False),
        "effects_applied_detail": {},
        "output_path": None,
        "status": "skipped",
    }

    # Only proceed if audio processing is enabled in config
    if not audio_log["applied"]:
        print(f"DEBUG: Librosa audio effects skipped (audio.apply is False).")
        # Return success=True (as nothing failed), but indicate skipped and provide original path for FFmpeg
        return True, audio_path, audio_log

    try:
        print(f"DEBUG: Loading audio for Librosa: {audio_path}")
        # Suppress specific UserWarning/FutureWarning if desired, or let them show
        y, sr = librosa.load(audio_path, sr=None)
        y_processed = y.copy()
        audio_log["status"] = "processing"

        # --- Apply Effects (Volume, Pitch, Time Stretch) ---
        # Volume Change
        vol_cfg = audio_cfg.get("volume_change", {})
        if random.random() < vol_cfg.get("prob", 0):
            db_change = random_float(vol_cfg.get("min_db", 0), vol_cfg.get("max_db", 0))
            y_processed = y_processed * (10 ** (db_change / 20.0))
            audio_log["effects_applied_detail"]["volume_change"] = {
                "prob": vol_cfg.get("prob"),
                "db_change": round(db_change, 2),
            }

        # Pitch Shift
        pitch_cfg = audio_cfg.get("pitch_shift", {})
        if random.random() < pitch_cfg.get("prob", 0):
            semitones = random_float(
                -pitch_cfg.get("max_semitones", 0), pitch_cfg.get("max_semitones", 0)
            )
            if abs(semitones) > 0.01:  # Avoid zero shift
                y_processed = librosa.effects.pitch_shift(
                    y=y_processed, sr=sr, n_steps=semitones
                )
                audio_log["effects_applied_detail"]["pitch_shift"] = {
                    "prob": pitch_cfg.get("prob"),
                    "max_semitones": pitch_cfg.get("max_semitones"),
                    "applied_semitones": round(semitones, 3),
                }

        # Time Stretch
        stretch_cfg = audio_cfg.get("time_stretch", {})
        if random.random() < stretch_cfg.get("prob", 0):
            rate = random_float(
                stretch_cfg.get("min_rate", 1.0), stretch_cfg.get("max_rate", 1.0)
            )
            if abs(rate - 1.0) > 0.005:  # Avoid identity stretch
                y_processed = librosa.effects.time_stretch(y=y_processed, rate=rate)
                audio_log["effects_applied_detail"]["time_stretch"] = {
                    "prob": stretch_cfg.get("prob"),
                    "min_rate": stretch_cfg.get("min_rate"),
                    "max_rate": stretch_cfg.get("max_rate"),
                    "applied_rate": round(rate, 4),
                }

        # Ensure audio doesn't clip excessively after processing
        y_processed = np.clip(y_processed, -0.98, 0.98)

        # --- Save as WAV file ---
        print(f"DEBUG: Writing processed audio as WAV: {wav_output_path}")
        # Using standard PCM 16-bit WAV format
        sf.write(wav_output_path, y_processed, sr, format="WAV", subtype="PCM_16")
        if not os.path.exists(wav_output_path) or os.path.getsize(wav_output_path) == 0:
            raise RuntimeError(
                f"Failed to write processed WAV audio to {wav_output_path}"
            )

        audio_log["status"] = "success"
        audio_log["output_path"] = wav_output_path
        # Return success=True, the path to the NEW WAV file, and the log
        return True, wav_output_path, audio_log

    except Exception as e:
        print(f"Error applying Librosa audio effects: {e}")
        # Don't print full traceback unless debugging needed, error message is often enough
        # traceback.print_exc()
        audio_log["status"] = "failed_fallback"
        audio_log["error"] = str(e)
        # Return success=False, the ORIGINAL audio path (AAC), and the log
        return False, audio_path, audio_log


# --- FFmpeg Combine/Encode Function (Helper for randomize_video) ---
def apply_ffmpeg_effects_and_reencode_randomized(
    video_in, audio_in, output_path, video_fps, config, applied_effects_log
):
    """Applies FFmpeg filters, encoding, metadata; returns success, error, and log dict."""
    metadata_cfg = config.get("metadata", {})
    encoding_cfg = config.get("encoding", {})
    audio_cfg = config.get("audio", {})  # For FFmpeg audio filters section
    ffmpeg_log = {
        "applied_audio_filters": False,
        "applied_encoding": False,
        "applied_metadata": False,
        "filters": [],
        "encoding": {},
        "metadata": {},
    }

    input_options = ["-i", video_in, "-i", audio_in]

    # --- Build Filtergraph (FFmpeg Audio Effects) ---
    audio_filters = []
    if audio_cfg.get("apply", False):
        # EQ Shift
        eq_cfg = audio_cfg.get("eq_shift", {})
        if random.random() < eq_cfg.get("prob", 0):
            ffmpeg_log["applied_audio_filters"] = True
            num_bands = eq_cfg.get("num_bands", 0)
            max_gain = eq_cfg.get("max_gain_db", 0)
            applied_eq_bands = []
            for _ in range(num_bands):
                center_freq = random_int(100, 8000)
                gain = random_float(-max_gain, max_gain)
                bandwidth = random_int(50, 200)
                if abs(gain) > 0.1:
                    filter_str = f"equalizer=f={center_freq}:width_type=h:width={bandwidth}:g={gain}"
                    audio_filters.append(filter_str)
                    applied_eq_bands.append(
                        {"f": center_freq, "w": bandwidth, "g": round(gain, 2)}
                    )
            if applied_eq_bands:
                ffmpeg_log["filters"].append(
                    {
                        "type": "equalizer",
                        "prob": eq_cfg.get("prob"),
                        "num_bands": num_bands,
                        "max_gain_db": max_gain,
                        "bands_applied": applied_eq_bands,
                    }
                )

        # Reverb
        # reverb_cfg = audio_cfg.get('reverb', {})
        # if random.random() < reverb_cfg.get('prob', 0):
        #     ffmpeg_log["applied_audio_filters"] = True
        #     reverb = random_int(0, reverb_cfg.get('max_reverberance', 0))
        #     room = random_int(5, reverb_cfg.get('max_room_scale', 5))
        #     if reverb > 5 or room > 10:
        #          filter_str = f"areverb=reverberance={reverb}:room_scale={room}"
        #          audio_filters.append(filter_str)
        #          ffmpeg_log["filters"].append({"type": "areverb", "prob": reverb_cfg.get('prob'), "reverberance": reverb, "room_scale": room})

    filter_complex = []
    if audio_filters:
        filter_complex.extend(
            ["-filter_complex", f"[1:a]{','.join(audio_filters)}[a_out]"]
        )
        audio_mapping = ["-map", "0:v", "-map", "[a_out]"]
    else:
        audio_mapping = ["-map", "0:v", "-map", "1:a"]

    # --- Encoding Parameters ---
    encoding_params = []
    enc_log = {}
    if encoding_cfg.get("apply", False):
        ffmpeg_log["applied_encoding"] = True
        crf_range = encoding_cfg.get("crf_range", [23, 28])
        preset_options = encoding_cfg.get("preset", ["medium", "slow"])
        tune_options = encoding_cfg.get("tune", ["film", "animation", "grain", None])
        bitrate_range = encoding_cfg.get("audio_bitrate_range", [96, 160])

        crf = random_int(crf_range[0], crf_range[1])
        preset = random.choice(preset_options)
        tune = random.choice(tune_options)
        audio_bitrate_val = random_int(bitrate_range[0], bitrate_range[1])
        audio_bitrate = f"{audio_bitrate_val}k"

        encoding_params.extend(["-c:v", "libx264"])
        encoding_params.extend(["-crf", str(crf)])
        encoding_params.extend(["-preset", preset])
        if tune:
            encoding_params.extend(["-tune", tune])
        encoding_params.extend(["-c:a", "aac"])
        encoding_params.extend(["-b:a", audio_bitrate])
        # Add pix_fmt for compatibility
        encoding_params.extend(["-pix_fmt", "yuv420p"])

        enc_log = {
            "codec": "libx264",
            "crf": crf,
            "preset": preset,
            "tune": tune,
            "audio_codec": "aac",
            "audio_bitrate_k": audio_bitrate_val,
        }
    else:
        # Default sensible encoding
        encoding_params.extend(
            [
                "-c:v",
                "libx264",
                "-crf",
                "23",
                "-preset",
                "medium",
                "-pix_fmt",
                "yuv420p",
            ]
        )
        encoding_params.extend(["-c:a", "aac", "-b:a", "128k"])
        enc_log = {
            "codec": "libx264",
            "crf": 23,
            "preset": "medium",
            "audio_codec": "aac",
            "audio_bitrate_k": 128,
        }
    ffmpeg_log["encoding"] = enc_log

    # --- Metadata Handling ---
    metadata_params = []
    meta_log = {}
    if metadata_cfg.get("apply", False):
        ffmpeg_log["applied_metadata"] = True
        if metadata_cfg.get("strip_all", True):  # Default to strip if applying
            metadata_params.extend(["-map_metadata", "-1"])
            meta_log["stripped_all"] = True

        add_random_cfg = metadata_cfg.get("add_random", {})
        if random.random() < add_random_cfg.get("prob", 0):
            added_keys = []
            keys_to_add = add_random_cfg.get("keys", ["comment"])  # Default keys
            for key in keys_to_add:
                val = generate_random_string()
                metadata_params.extend([f"-metadata", f"{key}={val}"])
                added_keys.append(key)
            if added_keys:
                meta_log["added_random_keys"] = added_keys
    ffmpeg_log["metadata"] = meta_log

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    # --- Assemble FFmpeg Command ---
    cmd = [ffmpeg_exe, "-y", "-hide_banner", "-loglevel", "warning"]  # Less verbose
    cmd.extend(input_options)
    cmd.extend(filter_complex)
    cmd.extend(audio_mapping)
    cmd.extend(encoding_params)
    cmd.extend(metadata_params)
    cmd.append(output_path)

    success, error = run_ffmpeg_command(cmd)
    return success, error, ffmpeg_log


# --- Main Randomization Function ---
def randomize_video(
    input_path,
    output_base_path,
    intensity="medium",
    config_profiles=RANDOMIZATION_PROFILES,
    randomization_log_path=None,  # <- New optional param
):
    """
    Main function to apply configured randomizations to a video file based on intensity.
    Returns: tuple[str | None, dict | None]: Path to randomized video, and log dict.
    """
    start_time = time.time()
    print(f"Starting randomization for: {input_path} with intensity: {intensity}")

    # --- Select Config based on Intensity ---
    base_config = config_profiles.get(intensity.lower(), config_profiles["medium"])
    config = (
        base_config.copy()
    )  # Shallow copy ok if profiles don't modify nested dicts directly
    if "output_options" not in config:
        config["output_options"] = DEFAULT_CONFIG["output_options"]

    # --- Define Output Path ---
    randomized_output_path = f"{output_base_path}_randomized.mp4"
    if randomization_log_path is None:
        randomization_log_path = f"{output_base_path}_randomizations.json"

    applied_settings = {
        "intensity": intensity,
        "input": input_path,
        "output": randomized_output_path,
        "effects": {},
    }

    # --- Temporary File Setup ---
    temp_dir = config["output_options"].get("temp_dir", "./temp_processing")
    os.makedirs(temp_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    unique_id = (
        str(int(time.time() * 1000)) + "_" + str(random.randint(100, 999))
    )  # More uniqueness
    temp_audio_original_path = os.path.join(
        temp_dir, f"{base_name}_{unique_id}_rand_orig_audio.aac"
    )
    temp_audio_processed_path = os.path.join(
        temp_dir, f"{base_name}_{unique_id}_rand_proc_audio.aac"
    )
    temp_video_processed_path = os.path.join(
        temp_dir, f"{base_name}_{unique_id}_rand_proc_video.mp4"
    )
    files_to_cleanup = {
        temp_audio_original_path,
        temp_audio_processed_path,
        temp_video_processed_path,
    }  # Use a set

    # --- Check if randomization is disabled ---
    # Simplified check looking only at top-level apply flags
    is_disabled = not (
        config.get("visual", {}).get("apply", False)
        or config.get("audio", {}).get("apply", False)
        or config.get("metadata", {}).get("apply", False)
        or config.get("encoding", {}).get("apply", False)
    )

    if is_disabled:
        print(
            f"[{intensity} intensity] Randomization effectively disabled by config 'apply' flags. Copying input to output."
        )
        try:
            shutil.copy(input_path, randomized_output_path)
            applied_settings["status"] = "skipped_disabled"
            try:
                with open(randomization_log_path, "w") as f:
                    json.dump(applied_settings, f, indent=4)
            except Exception as log_e:
                print(f"Warning: Failed to write randomization log: {log_e}")
            return randomized_output_path, applied_settings
        except Exception as e:
            print(f"Error copying file when randomization disabled: {e}")
            return None, None

    # --- Main Randomization Pipeline ---
    try:
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

        # 1. Extract Original Audio
        print("Extracting audio for randomization...")
        extract_cmd = [
            ffmpeg_exe,
            "-y",
            "-i",
            input_path,
            "-vn",
            "-acodec",
            "copy",
            temp_audio_original_path,
        ]
        success, err = run_ffmpeg_command(extract_cmd)
        if not success or not os.path.exists(temp_audio_original_path):
            raise RuntimeError(f"Failed to extract audio from {input_path}: {err}")

        # 2. Process Video Frames (Visual Randomization)
        print("Processing video frames for randomization...")
        processed_video_file, video_fps, visual_effects_log = (
            process_video_frame_by_frame_randomized(
                input_path,
                temp_video_processed_path,
                config,
                applied_settings["effects"],
                temp_dir,
            )
        )
        applied_settings["effects"]["visual"] = visual_effects_log
        if not processed_video_file:
            raise RuntimeError("Video frame randomization failed.")

        # 3. Process Audio (Librosa Effects)
        print("Applying Librosa audio effects for randomization...")
        audio_success, final_audio_path, audio_effects_log = (
            apply_audio_effects_librosa_randomized(
                temp_audio_original_path,
                temp_audio_processed_path,
                config,
                applied_settings["effects"],
            )
        )
        applied_settings["effects"]["audio_librosa"] = audio_effects_log
        if not audio_success:
            print(
                "Librosa processing failed or skipped, using original audio for FFmpeg."
            )
            final_audio_path = temp_audio_original_path
            if os.path.exists(temp_audio_processed_path):
                files_to_cleanup.discard(
                    temp_audio_processed_path
                )  # Don't cleanup if using original

        # 4. Combine, Apply FFmpeg Audio Filters, Re-encode, Handle Metadata
        print("Applying FFmpeg effects, re-encoding, and handling metadata...")
        ffmpeg_success, ffmpeg_error, ffmpeg_effects_log = (
            apply_ffmpeg_effects_and_reencode_randomized(
                processed_video_file,
                final_audio_path,
                randomized_output_path,
                video_fps,
                config,
                applied_settings["effects"],
            )
        )
        applied_settings["effects"]["ffmpeg_combine"] = ffmpeg_effects_log
        if not ffmpeg_success:
            raise RuntimeError(
                f"FFmpeg randomization processing failed: {ffmpeg_error}"
            )

        # Final check on output file
        if (
            not os.path.exists(randomized_output_path)
            or os.path.getsize(randomized_output_path) == 0
        ):
            raise RuntimeError(
                f"Final randomized output file missing or empty: {randomized_output_path}"
            )

        end_time = time.time()
        applied_settings["status"] = "success"
        applied_settings["processing_time_seconds"] = round(end_time - start_time, 2)
        print(f"Successfully randomized video: {randomized_output_path}")
        print(
            f"Total randomization time: {applied_settings['processing_time_seconds']:.2f} seconds"
        )

        # Log applied settings
        try:
            with open(randomization_log_path, "w") as f:
                json.dump(applied_settings, f, indent=4)
            print(f"Randomization settings logged to: {randomization_log_path}")
        except Exception as log_e:
            print(
                f"Warning: Failed to write randomization log {randomization_log_path}: {log_e}"
            )

        return randomized_output_path, applied_settings

    except Exception as e:
        print(f"An error occurred during randomization pipeline: {e}")
        traceback.print_exc()
        applied_settings["status"] = "failed"
        applied_settings["error"] = str(e)
        # Attempt to log failure info
        try:
            with open(randomization_log_path, "w") as f:
                json.dump(applied_settings, f, indent=4)
        except Exception as log_e:
            print(f"Warning: Failed to write failure log: {log_e}")
        return None, applied_settings

    finally:
        # Cleanup Temporary Files
        if config["output_options"].get("cleanup_temp", True):
            print("Cleaning up randomization temporary files...")
            # Add source audio path to potential cleanup list if Librosa failed
            if "audio_success" in locals() and not audio_success:
                files_to_cleanup.add(temp_audio_original_path)
            # Add processed video path to cleanup
            if "processed_video_file" in locals() and processed_video_file:
                files_to_cleanup.add(processed_video_file)

            for f_path in files_to_cleanup:
                if f_path and os.path.exists(f_path):  # Check path is not None
                    try:
                        os.remove(f_path)
                        # print(f" Cleaned up: {f_path}") # Optional: Verbose cleanup log
                    except Exception as e:
                        print(f"Warning: Could not remove temp file {f_path}: {e}")


# === END OF RANDOMIZER CODE BLOCK ===
