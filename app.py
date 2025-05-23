import os
import json # For saving embeddings
import face_recognition # For face recognition
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
from werkzeug.utils import secure_filename
from diffusers import StableDiffusionPipeline
import torch # For managing device (CPU/GPU) and data types
from PIL import Image # To save images
import uuid # For unique filenames

UPLOAD_FOLDER = 'uploads'
GENERATED_IMAGES_FOLDER = 'generated_images'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
FACIAL_DATA_FILE = 'data/facial_data.json'
STYLE_TEMPLATES_FILE = 'data/templates.json' # Path to the new templates file

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['GENERATED_IMAGES_FOLDER'] = GENERATED_IMAGES_FOLDER
app.secret_key = 'super secret key'  # Needed for flash messages

# Ensure the necessary folders exist
try:
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(os.path.dirname(FACIAL_DATA_FILE), exist_ok=True) # data/
    os.makedirs(GENERATED_IMAGES_FOLDER, exist_ok=True)
except OSError as e:
    app.logger.error(f"Error creating initial directories: {e}")
    # Depending on the severity, you might want to exit or disable features.
    # For now, we log and continue, Flask might not start if critical paths are missing.

# --- Stable Diffusion Setup ---
# Initialize pipeline - this can take time and memory
# For now, load it on demand. Consider loading at startup for performance in a real app.
# Check for CUDA availability
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIPE = None # Initialize later in the generate route to avoid loading on startup if not used

def load_sd_pipeline():
    global PIPE
    if PIPE is None:
        try:
            # Using a smaller, faster model for CPU for now.
            # Replace "runwayml/stable-diffusion-v1-5" with a lighter model if needed for CPU,
            # or ensure enough resources for "runwayml/stable-diffusion-v1-5"
            # For this task, we stick to the specified model.
            pipe_model_id = "runwayml/stable-diffusion-v1-5"
            
            # Use float32 for CPU for broader compatibility, float16 for CUDA if supported
            dtype = torch.float16 if DEVICE == "cuda" else torch.float32
            
            app.logger.info(f"Loading Stable Diffusion model: {pipe_model_id} onto {DEVICE} with dtype {dtype}")
            PIPE = StableDiffusionPipeline.from_pretrained(pipe_model_id, torch_dtype=dtype)
            PIPE = PIPE.to(DEVICE)
            app.logger.info("Stable Diffusion model loaded successfully.")
        except Exception as e:
            app.logger.error(f"Failed to load Stable Diffusion model: {e}")
            PIPE = None # Ensure it's None if loading fails
            raise # Re-raise exception to be caught by the route
    return PIPE

# --- End Stable Diffusion Setup ---

def load_style_templates():
    """Safely loads style templates from the JSON file."""
    try:
        if os.path.exists(STYLE_TEMPLATES_FILE) and os.path.getsize(STYLE_TEMPLATES_FILE) > 0:
            with open(STYLE_TEMPLATES_FILE, 'r') as f:
                return json.load(f)
        else:
            app.logger.info(f"{STYLE_TEMPLATES_FILE} not found or empty. Returning empty list.")
            return []
    except json.JSONDecodeError:
        app.logger.error(f"Error decoding JSON from {STYLE_TEMPLATES_FILE}. Returning empty list.")
        return []
    except (IOError, PermissionError) as e:
        app.logger.error(f"File error reading {STYLE_TEMPLATES_FILE}: {e}. Returning empty list.")
        return []


def load_facial_data():
    """Safely loads facial data from the JSON file."""
    try:
        if os.path.exists(FACIAL_DATA_FILE) and os.path.getsize(FACIAL_DATA_FILE) > 0:
            with open(FACIAL_DATA_FILE, 'r') as f:
                return json.load(f)
        else:
            # No error if file simply doesn't exist or is empty, it's an expected state.
            return {}
    except json.JSONDecodeError:
        app.logger.error(f"Error decoding JSON from {FACIAL_DATA_FILE}. Returning empty data.")
        return {} 
    except (IOError, PermissionError) as e:
        app.logger.error(f"File error reading {FACIAL_DATA_FILE}: {e}. Returning empty data.")
        return {}


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['image']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            file.save(filepath)
            flash('File successfully uploaded.', 'success')

            # Facial recognition part
            try:
                image = face_recognition.load_image_file(filepath)
                face_encodings = face_recognition.face_encodings(image)

                if face_encodings:
                    facial_data = load_facial_data() # Use the robust loader
                    facial_data[filename] = [enc.tolist() for enc in face_encodings]
                    
                    try:
                        with open(FACIAL_DATA_FILE, 'w') as f:
                            json.dump(facial_data, f, indent=4)
                        flash('Facial features extracted and stored.', 'success')
                    except (IOError, PermissionError) as e:
                        flash(f'Could not save facial data: {e}', 'error')
                        app.logger.error(f"Error saving facial data to {FACIAL_DATA_FILE}: {e}")
                else:
                    flash('No faces found in the uploaded image. Only the image was saved.', 'message')

            except Exception as e: # Catch errors from face_recognition library
                flash(f'Error processing image for facial recognition: {str(e)}', 'error')
                app.logger.error(f"Face recognition error for {filename}: {e}")

        except (IOError, PermissionError, Exception) as e: # Catch errors from file.save() or other unexpected
            flash(f'Could not save uploaded file: {str(e)}', 'error')
            app.logger.error(f"Error saving uploaded file {filename}: {e}")
        
        return redirect(url_for('index'))
    else:
        flash('Invalid file type. Allowed image types are -> png, jpg, jpeg.', 'error')
        return redirect(request.url)

def generate_image_from_prompt(prompt: str, selected_selfie_filename: str = None) -> str:
    """Generates an image from a prompt, optionally modifying it with a selfie reference, and saves it."""
    final_prompt = prompt

    # Subject replacement logic
    subject_placeholder = "[subject]"
    if subject_placeholder in prompt:
        if selected_selfie_filename and selected_selfie_filename != "none":
            subject_replacement = f"the person from the uploaded image [{selected_selfie_filename}]"
            final_prompt = prompt.replace(subject_placeholder, subject_replacement)
            app.logger.info(f"Replaced '{subject_placeholder}' with selfie reference: '{subject_replacement}'")
        else:
            # Generic replacement if no selfie is selected but [subject] is in the prompt
            subject_replacement = "a person" # Or "a character", "an object", etc.
            final_prompt = prompt.replace(subject_placeholder, subject_replacement)
            app.logger.info(f"Replaced '{subject_placeholder}' with generic term: '{subject_replacement}'")
    elif selected_selfie_filename and selected_selfie_filename != "none":
        # If no [subject] placeholder, but a selfie is selected, append/prepend selfie info (current behavior)
        # This maintains the previous logic for non-template prompts with selfies.
        final_prompt = f"photo of the person from the uploaded image [{selected_selfie_filename}] as {prompt}, detailed, high quality"
        app.logger.info(f"Selfie selected ('{selected_selfie_filename}'), prepending to custom prompt.")


    app.logger.info(f"Original prompt: '{prompt}'")
    if selected_selfie_filename and selected_selfie_filename != "none":
        app.logger.info(f"Selected selfie: '{selected_selfie_filename}'")
    app.logger.info(f"Final prompt for generation: '{final_prompt}'")
    
    try:
        pipe = load_sd_pipeline()
        if pipe is None:
            app.logger.error("Stable Diffusion pipeline is not loaded. Cannot generate image.")
            raise RuntimeError("Stable Diffusion model is not available. Please try again later or contact support.")

        app.logger.info(f"Generating image for prompt: '{final_prompt}'")
        image = pipe(final_prompt).images[0] # This can raise various errors from diffusers/torch
        app.logger.info("Image generated.")

        unique_filename = f"{uuid.uuid4()}.png"
        image_save_path = os.path.join(app.config['GENERATED_IMAGES_FOLDER'], unique_filename)
        
        # This os.makedirs is a bit redundant if created at startup, but good for safety.
        os.makedirs(app.config['GENERATED_IMAGES_FOLDER'], exist_ok=True) 
        image.save(image_save_path)
        app.logger.info(f"Image saved to {image_save_path}")
        return unique_filename 
    except RuntimeError as e: # Catch specific runtime errors we raise (like model not loaded)
        app.logger.error(f"Runtime error during image generation: {e}")
        raise
    except Exception as e: # Catch other potential errors (from .save(), pipe(), etc.)
        app.logger.error(f"Unexpected error generating image: {e}")
        # Provide a more generic error to the user for unexpected issues
        raise RuntimeError(f"An unexpected error occurred while generating the image: {str(e)}. Please check logs.")

@app.route('/generate', methods=['GET', 'POST'])
def generate():
    available_selfies = []
    style_templates = [] # For GET request

    if request.method == 'GET':
        facial_data = load_facial_data()
        available_selfies = list(facial_data.keys())
        style_templates = load_style_templates()
        prompt = request.args.get('prompt', '') # Keep existing prompt if any
        selected_selfie = request.args.get('selected_selfie_filename', '')
        # Pass style_templates to the template
        return render_template('generate_form.html', 
                               available_selfies=available_selfies, 
                               prompt=prompt, 
                               style_templates=style_templates,
                               selected_selfie_filename=selected_selfie)

    # POST request
    prompt = request.form.get('prompt')
    selected_selfie_filename = request.form.get('selected_selfie_filename')
    # The actual template prompt is now directly submitted via the 'prompt' field by JS
    # No need to handle template_name on backend if using Option A (JS populates prompt field)

    if not prompt:
        flash('Please provide a prompt for image generation.', 'error')
        # Reload data for form re-rendering
        facial_data = load_facial_data()
        available_selfies = list(facial_data.keys())
        style_templates = load_style_templates()
        return render_template('generate_form.html', 
                               available_selfies=available_selfies, 
                               prompt=prompt, 
                               style_templates=style_templates,
                               selected_selfie_filename=selected_selfie_filename), 400

    original_prompt_for_template_check = request.form.get('prompt_before_js_fill', prompt)
    style_template_name_used = None
    if original_prompt_for_template_check != prompt and "[subject]" in prompt : # Heuristic: if JS changed it and it has [subject], likely a template
        # Find the template name
        style_templates_for_name = load_style_templates()
        for t in style_templates_for_name:
            if t['prompt'] == prompt: # The prompt submitted IS the template prompt
                style_template_name_used = t['name']
                break

    try:
        image_filename = generate_image_from_prompt(prompt, selected_selfie_filename)
        
        # Determine flash message category and content
        flash_message = "Image generated successfully!"
        flash_category = "success"

        if selected_selfie_filename and selected_selfie_filename != "none":
            if style_template_name_used:
                flash_message = f"Image generated from '{style_template_name_used}' template, influenced by selected selfie. Likeness is experimental."
            else: # Custom prompt with selfie
                flash_message = "Image generated with selected selfie. Likeness is experimental."
            flash_category = "message" # Use 'message' for experimental features
        elif style_template_name_used:
            flash_message = f"Image generated from '{style_template_name_used}' template."
        
        flash(flash_message, flash_category)
            
        # Redirect to the new display page
        return redirect(url_for('show_generated_image_page', 
                                filename=image_filename, 
                                prompt_used=prompt, # The actual prompt used for generation
                                selected_selfie_filename=selected_selfie_filename or '',
                                style_template_name=style_template_name_used or ''))

    except Exception as e:
        flash(f'Error generating image: {str(e)}', 'error')
        app.logger.error(f"Exception in /generate POST: {e}")
        facial_data = load_facial_data() # For re-rendering the form
        available_selfies = list(facial_data.keys())
        style_templates = load_style_templates()
        return render_template('generate_form.html', 
                               available_selfies=available_selfies, 
                               prompt=prompt, # Return the user's original prompt
                               style_templates=style_templates,
                               selected_selfie_filename=selected_selfie_filename)

@app.route('/show_image/<filename>')
def show_generated_image_page(filename):
    prompt_used = request.args.get('prompt_used', '[Prompt not available]')
    selected_selfie = request.args.get('selected_selfie_filename', '')
    style_template_name = request.args.get('style_template_name', '')
    return render_template('display_image.html', 
                           image_filename=filename, 
                           prompt_used=prompt_used,
                           selected_selfie_filename=selected_selfie,
                           style_template_name=style_template_name)

@app.route('/generated_images/<filename>')
def serve_generated_image(filename):
    """Serves the actual image file."""
    return send_from_directory(app.config['GENERATED_IMAGES_FOLDER'], filename)


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO) # Basic logging setup
    # Consider more advanced logging for production (e.g., file-based, rotation)
    app.run(debug=True)
