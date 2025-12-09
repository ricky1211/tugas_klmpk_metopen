import os
import cv2
import numpy as np
import base64
import time
import json
from datetime import datetime
from flask import Flask, render_template, Response, request, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from collections import defaultdict
import logging

# --- Konfigurasi Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Konfigurasi Aplikasi Flask ---
app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['PROCESSED_FOLDER'] = 'static/processed/' # Folder untuk menyimpan gambar/video hasil deteksi
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'avi', 'mov'}

# Pastikan folder ada
try:
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)
    logging.info(f"Folders created/ensured: {app.config['UPLOAD_FOLDER']}, {app.config['PROCESSED_FOLDER']}")
except OSError as e:
    logging.error(f"Error creating folders: {e}")

# --- Inisialisasi Model YOLO ---
# Ganti 'yolov8n.pt' dengan path ke model YOLO Anda yang sudah dilatih jika berbeda
# Pastikan model ini ada di direktori yang sama dengan app.py, atau berikan path absolut/relatif yang benar.
MODEL_PATH = 'yolov8n.pt' # Ganti jika nama atau lokasi model Anda berbeda
model = None # Inisialisasi model sebagai None secara default
try:
    logging.info(f"Attempting to load YOLO model from: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    logging.info("Model YOLO berhasil dimuat!")
except Exception as e:
    logging.error(f"Gagal memuat model YOLO dari '{MODEL_PATH}': {e}")
    logging.error("Pastikan file model ada di direktori yang sama dengan app.py atau berikan path yang benar.")
    logging.error("Jika menggunakan GPU, pastikan driver CUDA dan PyTorch terinstal dengan benar.")
    model = None # Set None jika model gagal dimuat

# --- Variabel Global untuk Live Detection dan Analytics ---
camera = None
is_detecting_live = False
global_frame_count = 0
global_total_objects = 0
global_processing_time = 0.0
global_avg_confidence_sum = 0.0
global_avg_confidence_count = 0
global_start_time = 0.0

# Data untuk Chart (Timeline) - Batasi ukuran agar tidak boros memori
MAX_TIMELINE_POINTS = 500
detection_timeline_labels = [] # Label sumbu X (Frame number)
detection_timeline_data = [] # Jumlah objek per frame
confidence_timeline_data = [] # Rata-rata confidence per frame

# Data untuk Chart (Class Distribution)
class_counts = defaultdict(int)
class_colors = {} # Untuk menyimpan warna konsisten untuk setiap kelas
class_color_palette = [
    'rgba(255, 99, 132, 0.5)', 'rgba(54, 162, 235, 0.5)', 'rgba(255, 206, 86, 0.5)',
    'rgba(75, 192, 192, 0.5)', 'rgba(153, 102, 255, 0.5)', 'rgba(255, 159, 64, 0.5)',
    'rgba(199, 199, 199, 0.5)', 'rgba(83, 102, 255, 0.5)', 'rgba(255, 0, 0, 0.5)',
    'rgba(0, 255, 0, 0.5)', 'rgba(128, 0, 128, 0.5)', 'rgba(0, 128, 0, 0.5)',
    'rgba(0, 0, 128, 0.5)', 'rgba(128, 128, 0, 0.5)'
]
color_idx = 0

# Data untuk Heatmap dan Zone
# Ukuran heatmap harus konsisten. Sesuaikan dengan rata-rata resolusi input Anda.
# Contoh: 640x480 atau 1280x720. Sesuaikan dengan kebutuhan memori dan performa.
HEATMAP_WIDTH = 640
HEATMAP_HEIGHT = 480
heatmap_data = np.zeros((HEATMAP_HEIGHT, HEATMAP_WIDTH), dtype=np.float32) 
zone_names = ['Top-Left', 'Top-Right', 'Bottom-Left', 'Bottom-Right']
zone_counts = [0, 0, 0, 0] 

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# --- Live Detection Logic ---
def generate_frames():
    global camera, is_detecting_live, global_frame_count, global_total_objects, \
           global_processing_time, global_avg_confidence_sum, global_avg_confidence_count, \
           global_start_time, detection_timeline_labels, detection_timeline_data, \
           confidence_timeline_data, class_counts, heatmap_data, zone_counts, class_colors, color_idx

    if not camera: # Pastikan kamera diinisialisasi hanya sekali
        camera = cv2.VideoCapture(0)  # Gunakan kamera default (0). Coba 1, 2, dst jika ada kamera lain.
        if not camera.isOpened():
            logging.error("Error: Could not open camera. Please check if camera is connected and not in use.")
            is_detecting_live = False
            # Mengirimkan placeholder image agar frontend tidak kosong
            with open('static/placeholder.jpg', 'rb') as f:
                placeholder_img = f.read()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + placeholder_img + b'\r\n')
            return

    logging.info("Starting video stream generation...")
    global_start_time = time.time()
    last_frame_time = time.time()

    while is_detecting_live:
        success, frame = camera.read()
        if not success:
            logging.warning("Failed to read frame from camera. Exiting stream.")
            break

        current_time = time.time()
        
        # Lakukan deteksi YOLO
        if model:
            # Perhatikan: stream=True sangat dianjurkan untuk live video feed
            # conf=0.25 (default) dan iou=0.7 (default) bisa disesuaikan
            results = model(frame, stream=True, verbose=False) # verbose=True untuk debugging lebih banyak
            frame_objects_count = 0
            frame_confidences = []
            current_frame_heatmap_points = []
            
            current_zone_counts = [0, 0, 0, 0] # Reset zone counts for current frame processing

            annotated_frame = frame.copy() # Buat salinan untuk anotasi
            
            for r in results:
                # `r.plot()` akan menggambar bounding box dan label pada frame.
                # Pastikan r.plot() mengembalikan frame yang sudah dianotasi.
                annotated_frame = r.plot() 
                
                boxes = r.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    # Pastikan model.names tersedia. Jika model gagal dimuat, ini bisa error.
                    name = model.names[cls] if model and hasattr(model, 'names') else f"Class_{cls}"
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0]) # Bounding box coordinates
                    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                    
                    frame_objects_count += 1
                    frame_confidences.append(conf)
                    
                    # Update class counts
                    class_counts[name] += 1
                    if name not in class_colors:
                        class_colors[name] = class_color_palette[color_idx % len(class_color_palette)]
                        color_idx += 1

                    # Update heatmap data
                    # Normalize coordinates to heatmap size
                    if annotated_frame.shape[0] > 0 and annotated_frame.shape[1] > 0:
                        norm_x = int((center_x / annotated_frame.shape[1]) * HEATMAP_WIDTH)
                        norm_y = int((center_y / annotated_frame.shape[0]) * HEATMAP_HEIGHT)
                        
                        # Pastikan koordinat dalam batas heatmap_data
                        norm_x = max(0, min(norm_x, HEATMAP_WIDTH - 1))
                        norm_y = max(0, min(norm_y, HEATMAP_HEIGHT - 1))
                        
                        current_frame_heatmap_points.append((norm_y, norm_x))
                        
                    # Update zone counts (simple 4-quadrant example)
                    frame_h, frame_w = annotated_frame.shape[:2]
                    if center_x < frame_w / 2 and center_y < frame_h / 2:
                        current_zone_counts[0] += 1 # Top-Left
                    elif center_x >= frame_w / 2 and center_y < frame_h / 2:
                        current_zone_counts[1] += 1 # Top-Right
                    elif center_x < frame_w / 2 and center_y >= frame_h / 2:
                        current_zone_counts[2] += 1 # Bottom-Left
                    else:
                        current_zone_counts[3] += 1 # Bottom-Right

            # Update heatmap for this frame
            for (ny, nx) in current_frame_heatmap_points:
                # Apply a small increase to the heatmap point
                heatmap_data[ny, nx] += 1 
                # Optional: apply blur or spread for visual effect, but can be costly
                # cv2.circle(heatmap_data, (nx, ny), 5, 1, -1) # Draw a circle (requires scaling to 0-1)


            # Accumulate zone counts
            for i in range(len(zone_counts)):
                zone_counts[i] += current_zone_counts[i]


            global_frame_count += 1
            global_total_objects += frame_objects_count

            # Update timeline data, limit size
            detection_timeline_labels.append(global_frame_count)
            detection_timeline_data.append(frame_objects_count)
            
            if frame_confidences:
                avg_conf = sum(frame_confidences) / len(frame_confidences)
                confidence_timeline_data.append(avg_conf)
                global_avg_confidence_sum += avg_conf
                global_avg_confidence_count += 1
            else:
                confidence_timeline_data.append(0) # No objects, no confidence

            # Trim timeline data to prevent excessive memory usage over long runs
            if len(detection_timeline_labels) > MAX_TIMELINE_POINTS:
                detection_timeline_labels.pop(0)
                detection_timeline_data.pop(0)
                confidence_timeline_data.pop(0)

        else: # If model is not loaded, just send the original frame
            annotated_frame = frame.copy()
            cv2.putText(annotated_frame, "YOLO Model Not Loaded!", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            logging.warning("YOLO model is not loaded, sending raw frames with warning.")


        # Hitung FPS dan waktu pemrosesan
        elapsed_time_per_frame = current_time - last_frame_time
        global_processing_time += elapsed_time_per_frame
        last_frame_time = current_time

        # Encode frame sebagai JPEG
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        if not ret:
            logging.error("Failed to encode frame to JPEG.")
            continue
        
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    # Clean up when stream stops
    if camera:
        camera.release()
        logging.info("Camera released.")
    logging.info("Live detection stream stopped.")

# --- Routes ---
@app.route('/')
def index():
    logging.info("Serving index.html")
    return render_template('index.html')

@app.route('/start_detection')
def start_detection():
    global is_detecting_live, global_frame_count, global_total_objects, \
           global_processing_time, global_avg_confidence_sum, global_avg_confidence_count, \
           global_start_time, detection_timeline_labels, detection_timeline_data, \
           confidence_timeline_data, class_counts, heatmap_data, zone_counts, camera

    if not model:
        logging.error("Attempted to start detection, but YOLO model is not loaded.")
        return jsonify({'status': 'error', 'message': 'YOLO model not loaded. Please check backend console for errors.'}), 500

    if not is_detecting_live:
        is_detecting_live = True
        logging.info("Detection started.")
        # Reset analytics on start
        global_frame_count = 0
        global_total_objects = 0
        global_processing_time = 0.0
        global_avg_confidence_sum = 0.0
        global_avg_confidence_count = 0
        
        # Clear timeline data
        detection_timeline_labels.clear()
        detection_timeline_data.clear()
        confidence_timeline_data.clear()
        
        class_counts.clear()
        
        # Reset heatmap and zones
        heatmap_data = np.zeros((HEATMAP_HEIGHT, HEATMAP_WIDTH), dtype=np.float32) 
        zone_counts = [0, 0, 0, 0] 
        
        # Re-initialize camera on start
        if camera: # Release if it was left open from a previous run
            camera.release()
            camera = None
        
        return jsonify({'status': 'success', 'message': 'Detection started'})
    logging.info("Detection already running.")
    return jsonify({'status': 'info', 'message': 'Detection already running'})

@app.route('/stop_detection')
def stop_detection():
    global is_detecting_live, camera
    if is_detecting_live:
        is_detecting_live = False
        if camera:
            camera.release()
            camera = None
        logging.info("Detection stopped and camera released.")
        return jsonify({'status': 'success', 'message': 'Detection stopped'})
    logging.info("Detection not running.")
    return jsonify({'status': 'info', 'message': 'Detection not running'})

@app.route('/video_feed')
def video_feed():
    # Pastikan is_detecting_live True untuk memulai stream
    if is_detecting_live:
        logging.info("Client requested video feed.")
        return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        # Jika tidak mendeteksi, tampilkan placeholder
        logging.info("Client requested video feed, but detection is not active. Serving placeholder.")
        with open('static/placeholder.jpg', 'rb') as f:
            placeholder_img = f.read()
        return Response(placeholder_img, mimetype='image/jpeg')

@app.route('/get_stats')
def get_stats():
    current_fps = 0
    if global_processing_time > 0 and global_frame_count > 0:
        current_fps = global_frame_count / global_processing_time
    
    avg_conf = (global_avg_confidence_sum / global_avg_confidence_count) if global_avg_confidence_count > 0 else 0
    
    stats = {
        'objects_detected': global_total_objects,
        'fps': round(current_fps, 2), # Round for cleaner display
        'total_objects': global_total_objects,
        'avg_confidence': round(avg_conf, 4), # Round for cleaner display
        'processing_time': round(global_processing_time, 2),
        'total_frames': global_frame_count 
    }
    # logging.debug(f"Sending stats: {stats}") # Too verbose for regular logging
    return jsonify({'status': 'success', 'data': stats})

@app.route('/get_chart_data')
def get_chart_data():
    # Prepare class distribution data
    # For a line chart of class distribution over time, you'd need to store
    # class_counts per frame or at intervals. This current implementation
    # provides total counts, which is more suited for a bar/doughnut chart.
    # If the frontend *must* have a line chart for this, it needs more complex
    # data collection in `generate_frames` (e.g., a list of dictionaries, each
    # representing class counts at a specific frame/time).
    
    current_class_labels = list(class_counts.keys())
    current_class_values = list(class_counts.values())
    
    # Create datasets for individual classes for a line chart (if frontend supports)
    # This will create a separate line for each class, showing its cumulative count.
    class_line_datasets = []
    # If you want real-time dynamic class counts *per frame*, you need to store
    # a history of `class_counts` dictionary in `generate_frames`.
    # For now, let's keep it simple: total counts.
    for cls_name in sorted(current_class_labels): # Sort for consistent order
        if cls_name in class_colors:
            color = class_colors[cls_name]
        else: # Assign a color if not already assigned
            color = class_color_palette[color_idx % len(class_color_palette)]
            class_colors[cls_name] = color
            color_idx += 1
        
        class_line_datasets.append({
            'label': cls_name,
            'data': [class_counts[cls_name]], # Simple: total count as a single point
            'backgroundColor': color,
            'borderColor': color.replace('0.5', '1'), # Brighter border
            'borderWidth': 1,
            'fill': False,
            'tension': 0.1
        })


    detection_chart_data = {
        'labels': detection_timeline_labels,
        'datasets': [{
            'label': 'Objects per frame',
            'data': detection_timeline_data,
            'borderColor': 'rgb(75, 192, 192)',
            'backgroundColor': 'rgba(75, 192, 192, 0.2)',
            'tension': 0.1,
            'fill': False
        }]
    }

    confidence_chart_data = {
        'labels': detection_timeline_labels, 
        'datasets': [{
            'label': 'Average confidence',
            'data': confidence_timeline_data,
            'borderColor': 'rgb(255, 99, 132)',
            'backgroundColor': 'rgba(255, 99, 132, 0.2)',
            'tension': 0.1,
            'fill': False
        }]
    }

    # For the frontend's "Class Distribution" chart, if it expects a bar/doughnut chart:
    class_bar_doughnut_data = {
        'labels': current_class_labels,
        'datasets': [{
            'label': 'Total Object Count',
            'data': current_class_values,
            'backgroundColor': [class_colors.get(c, 'rgba(0,0,0,0.5)') for c in current_class_labels],
            'borderColor': [class_colors.get(c, 'rgba(0,0,0,1)') for c in current_class_labels],
            'borderWidth': 1
        }]
    }

    # Heatmap zones data
    heatmap_zones_data = {
        'zone_names': zone_names,
        'zone_counts': zone_counts
    }

    # logging.debug("Sending chart data.") # Too verbose
    return jsonify({
        'status': 'success',
        'data': {
            'detection_data': detection_chart_data,
            'confidence_data': confidence_chart_data,
            # Provide both, let frontend decide which to use
            'class_distribution_bar_doughnut': class_bar_doughnut_data, 
            'class_distribution_lines': class_line_datasets, # If frontend expects line per class
            'heatmap_zones': heatmap_zones_data
        }
    })

@app.route('/get_heatmap_image')
def get_heatmap_image():
    # If detection is not active or no data has been accumulated, send a default/blank image
    if not is_detecting_live or heatmap_data.sum() == 0:
        logging.info("No live detection or no heatmap data. Sending default heatmap image.")
        # Create a blank image to return
        blank_heatmap = np.zeros((HEATMAP_HEIGHT, HEATMAP_WIDTH, 3), dtype=np.uint8)
        cv2.putText(blank_heatmap, "No Heatmap Data", (HEATMAP_WIDTH // 2 - 100, HEATMAP_HEIGHT // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        ret, buffer = cv2.imencode('.jpg', blank_heatmap)
        encoded_image = base64.b64encode(buffer).decode('utf-8')
        return jsonify({'status': 'success', 'image': f'data:image/jpeg;base64,{encoded_image}'})

    # Normalize heatmap_data to 0-255 range
    normalized_heatmap = cv2.normalize(heatmap_data, None, 0, 255, cv2.NORM_MINMAX)
    normalized_heatmap = np.uint8(normalized_heatmap)
    
    # Apply a colormap (e.g., COLORMAP_JET)
    heatmap_colored = cv2.applyColorMap(normalized_heatmap, cv2.COLORMAP_JET)
    
    ret, buffer = cv2.imencode('.jpg', heatmap_colored)
    if ret:
        encoded_image = base64.b64encode(buffer).decode('utf-8')
        # Penting: Tambahkan prefix data URI untuk gambar base64 agar bisa langsung digunakan di <img> tag
        return jsonify({'status': 'success', 'image': f'data:image/jpeg;base64,{encoded_image}'})
    
    logging.error("Failed to encode heatmap image.")
    # Fallback if encoding fails
    return jsonify({'status': 'error', 'message': 'Failed to generate heatmap image.'}), 500

@app.route('/reset_analytics', methods=['POST'])
def reset_analytics():
    global global_frame_count, global_total_objects, global_processing_time, \
           global_avg_confidence_sum, global_avg_confidence_count, global_start_time, \
           detection_timeline_labels, detection_timeline_data, confidence_timeline_data, \
           class_counts, heatmap_data, zone_counts

    global_frame_count = 0
    global_total_objects = 0
    global_processing_time = 0.0
    global_avg_confidence_sum = 0.0
    global_avg_confidence_count = 0
    global_start_time = 0.0

    detection_timeline_labels.clear()
    detection_timeline_data.clear()
    confidence_timeline_data.clear()
    class_counts.clear()
    heatmap_data = np.zeros(heatmap_data.shape, dtype=np.float32) # Reset heatmap
    zone_counts = [0, 0, 0, 0] # Reset zone counts
    
    logging.info("Analytics data reset.")
    return jsonify({'status': 'success', 'message': 'Analytics data reset.'})

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'files' not in request.files:
        logging.warning("No file part in upload request.")
        return jsonify({'status': 'error', 'message': 'No file part in the request.'}), 400
    
    files = request.files.getlist('files')
    uploaded_filenames = []
    errors = []

    for file in files:
        if file.filename == '':
            errors.append('No selected file.')
            continue
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            try:
                file.save(filepath)
                uploaded_filenames.append(filename)
                logging.info(f"File uploaded: {filename}")
            except Exception as e:
                errors.append(f'Failed to save {filename}: {str(e)}')
                logging.error(f"Error saving uploaded file {filename}: {e}")
        else:
            errors.append(f'File type not allowed for {file.filename}.')
            logging.warning(f"File type not allowed: {file.filename}")
    
    if errors:
        return jsonify({'status': 'error', 'message': 'Some files failed to upload.', 'details': errors, 'uploaded': uploaded_filenames}), 400
    
    return jsonify({'status': 'success', 'message': f'{len(uploaded_filenames)} file(s) uploaded successfully.', 'filenames': uploaded_filenames})

@app.route('/process_uploads', methods=['POST'])
def process_uploads():
    data = request.json
    filenames = data.get('filenames', [])
    
    if not filenames:
        logging.warning("No filenames provided for processing uploads.")
        return jsonify({'status': 'error', 'message': 'No filenames provided for processing.'}), 400
    
    if not model:
        logging.error("Attempted to process uploads, but YOLO model is not loaded.")
        return jsonify({'status': 'error', 'message': 'YOLO model not loaded. Cannot process files.'}), 500

    processed_results = []
    for filename in filenames:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(filepath):
            processed_results.append({'filename': filename, 'status': 'failed', 'message': 'File not found.'})
            logging.warning(f"File not found for processing: {filepath}")
            continue

        try:
            file_extension = filename.rsplit('.', 1)[1].lower()
            if file_extension in ['png', 'jpg', 'jpeg', 'gif']:
                img = cv2.imread(filepath)
                if img is None:
                    processed_results.append({'filename': filename, 'status': 'failed', 'message': 'Could not read image.'})
                    logging.error(f"Could not read image file: {filepath}")
                    continue

                logging.info(f"Processing image: {filename}")
                results = model(img) # Run inference on the image
                
                output_filename = f"processed_{filename}"
                output_filepath = os.path.join(app.config['PROCESSED_FOLDER'], output_filename)
                
                # Draw detections on image. Use `results[0].plot()` for single image results.
                # If `results` is a list, typically it contains one result object per input.
                annotated_img = results[0].plot() 
                cv2.imwrite(output_filepath, annotated_img) # Save the annotated image
                logging.info(f"Processed image saved to: {output_filepath}")

                processed_results.append({
                    'filename': filename,
                    'status': 'success',
                    'processed_url': f'/static/processed/{output_filename}',
                    'detections': len(results[0].boxes) # Number of detections
                })
            elif file_extension in ['mp4', 'avi', 'mov']:
                logging.info(f"Attempting to process video: {filename} (Partial implementation)")
                # --- Video Processing Placeholder (More Complex) ---
                # Untuk video, Anda perlu membaca frame demi frame, memproses setiap frame,
                # dan kemudian menulis frame yang dianotasi ke file video baru.
                # Ini adalah operasi yang membutuhkan waktu dan bisa diblokir jika dilakukan langsung
                # di thread Flask utama. Sebaiknya menggunakan background task queue.

                # Contoh sangat dasar untuk video:
                cap = cv2.VideoCapture(filepath)
                if not cap.isOpened():
                    processed_results.append({'filename': filename, 'status': 'failed', 'message': 'Could not open video file.'})
                    logging.error(f"Could not open video file: {filepath}")
                    continue

                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                output_video_filename = f"processed_{filename}"
                output_video_filepath = os.path.join(app.config['PROCESSED_FOLDER'], output_video_filename)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for .mp4
                out = cv2.VideoWriter(output_video_filepath, fourcc, fps, (width, height))

                frame_idx = 0
                video_total_detections = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    results = model(frame, verbose=False) # Inferensi per frame
                    annotated_frame = results[0].plot() # Ambil frame yang sudah dianotasi
                    out.write(annotated_frame) # Tulis ke file output
                    video_total_detections += len(results[0].boxes)
                    frame_idx += 1
                    # logging.info(f"Processing frame {frame_idx}/{total_frames} for {filename}") # Too verbose

                cap.release()
                out.release()
                logging.info(f"Finished processing video: {filename}. Saved to {output_video_filepath}")

                processed_results.append({
                    'filename': filename,
                    'status': 'success',
                    'processed_url': f'/static/processed/{output_video_filename}',
                    'detections': video_total_detections,
                    'message': f'Video processed. Total frames: {total_frames}. Total objects: {video_total_detections}'
                })

            else:
                processed_results.append({'filename': filename, 'status': 'failed', 'message': 'Unsupported file type.'})
                logging.warning(f"Unsupported file type for processing: {filename}")

        except Exception as e:
            processed_results.append({'filename': filename, 'status': 'failed', 'message': str(e)})
            logging.error(f"Error processing {filename}: {e}")

    logging.info("Processing complete for uploaded files.")
    return jsonify({'status': 'success', 'message': 'Processing complete.', 'results': processed_results})

@app.route('/clear_uploads', methods=['POST'])
def clear_uploads():
    """Menghapus semua file dari folder uploads dan processed."""
    try:
        for folder in [app.config['UPLOAD_FOLDER'], app.config['PROCESSED_FOLDER']]:
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    import shutil
                    shutil.rmtree(file_path)
        logging.info("Uploads and processed folders cleared.")
        return jsonify({'status': 'success', 'message': 'Uploads and processed folders cleared.'})
    except Exception as e:
        logging.error(f"Error clearing folders: {e}")
        return jsonify({'status': 'error', 'message': f'Failed to clear folders: {str(e)}'}), 500

# --- Main execution ---
if __name__ == '__main__':
    logging.info("Starting Flask application...")
    # Ini harusnya di run terpisah dari flask dev server untuk production
    # Tapi untuk pengembangan, ini cukup
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True, use_reloader=False) 
    # use_reloader=False menghindari model dimuat dua kali pada beberapa lingkungan