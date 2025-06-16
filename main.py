import cv2
import os
import json
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import threading
import queue
import time
from collections import deque
import logging
import deepface
from PIL import Image

# Configure logging for performance monitoring
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealTimeTracker:

    def __init__(self):

        # Initializes the tracker with directories, models, and state variables.
        os.makedirs("tracked_people", exist_ok=True)
        os.makedirs("video_footage", exist_ok=True)

        self.MAPPINGS_PATH = "id_mappings.json"
        self.EMBEDDINGS_PATH = "face_embeddings.json"

        # Load the YOLOv8 model for object detection and tracking.
        self.model = YOLO("yolov8n.pt")

        # Load existing ID mappings and face embeddings
        self.track_last_snapshot = {}
        self.id_mappings = self.load_json(self.MAPPINGS_PATH)
        self.face_embeddings = self.load_embeddings()

        # Initialize a counter for new person IDs.
        self.person_id_counter = self.get_max_person_id() + 1

        # Initialize data structures for FPS calculation.
        self.frame_times = deque(maxlen=30)
        self.last_fps_update = time.time()
        self.current_fps = 0

        # Face recognition parameters
        self.FACE_RECOGNITION_THRESHOLD = 0.6  # Lower = more strict matching
        self.MIN_FACE_SIZE = 50  # Minimum face size in pixels
        

    def load_json(self, path):
        
        # Loads JSON data from a specified file path.
        return json.load(open(path)) if os.path.exists(path) else {}

    def save_json(self, path, data):
        
        # Saves data to a JSON file at the specified path.
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load_embeddings(self):
      
        if os.path.exists(self.EMBEDDINGS_PATH):
            with open(self.EMBEDDINGS_PATH, 'r') as f:
                data = json.load(f)
                # Convert lists back to numpy arrays
                for person_id in data:
                    data[person_id] = [np.array(embedding) for embedding in data[person_id]]
                return data
        return {}

    def save_embeddings(self):
       
        # Convert numpy arrays to lists for JSON serialization
        data_to_save = {}
        for person_id, embeddings in self.face_embeddings.items():
            data_to_save[person_id] = [embedding.tolist() for embedding in embeddings]
        
        with open(self.EMBEDDINGS_PATH, 'w') as f:
            json.dump(data_to_save, f, indent=2)

    def get_max_person_id(self):
       
        max_id = 0
        
        # Check existing embeddings
        for person_id in self.face_embeddings.keys():
            try:
                max_id = max(max_id, int(person_id))
            except ValueError:
                continue
                
        # Check existing directories
        if os.path.exists("tracked_people"):
            for dir_name in os.listdir("tracked_people"):
                try:
                    max_id = max(max_id, int(dir_name))
                except ValueError:
                    continue
                    
        return max_id

    def extract_face_encoding(self, image, bbox=None):
       
        try:
            # Convert BGR to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = image

            # If bbox is provided, crop the image
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                rgb_image = rgb_image[y1:y2, x1:x2]

            # Check if the cropped image is large enough
            if rgb_image.shape[0] < self.MIN_FACE_SIZE or rgb_image.shape[1] < self.MIN_FACE_SIZE:
                return None

            # Find face locations and encodings
            face_locations = deepface.face_locations(rgb_image, model="hog")
            
            if not face_locations:
                return None

            # Get the largest face (assuming it's the main subject)
            largest_face = max(face_locations, key=lambda loc: (loc[2] - loc[0]) * (loc[1] - loc[3]))
            
            # Get face encoding
            face_encodings = deepface.face_encodings(rgb_image, [largest_face])
            
            if face_encodings:
                return face_encodings[0]
            else:
                return None

        except Exception as e:
            logger.warning(f"Error extracting face encoding: {e}")
            return None

    def find_matching_person(self, face_encoding):
        
        if face_encoding is None:
            return None

        best_person_id = None
        best_distance = float('inf')

        for person_id, stored_encodings in self.face_embeddings.items():
            for stored_encoding in stored_encodings:
                try:
                    distance = deepface.face_distance([stored_encoding], face_encoding)[0]
                    if distance < self.FACE_RECOGNITION_THRESHOLD and distance < best_distance:
                        best_distance = distance
                        best_person_id = person_id
                except Exception as e:
                    logger.warning(f"Error comparing face encodings: {e}")
                    continue

        return best_person_id

    def add_face_encoding(self, person_id, face_encoding):
    
        if face_encoding is None:
            return

        if person_id not in self.face_embeddings:
            self.face_embeddings[person_id] = []
        
        # Limit the number of stored encodings per person to avoid memory issues
        max_encodings_per_person = 5
        if len(self.face_embeddings[person_id]) >= max_encodings_per_person:
            
            # Remove the oldest encoding
            self.face_embeddings[person_id].pop(0)
        
        self.face_embeddings[person_id].append(face_encoding)

    def update_fps(self, frame_time):
        
        # Calculates and updates the current frames per second (FPS).
        self.frame_times.append(frame_time)
        current_time = time.time()

        if current_time - self.last_fps_update > 1.0:
            if len(self.frame_times) > 1:
                avg_frame_time = sum(self.frame_times) / len(self.frame_times)
                self.current_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
            self.last_fps_update = current_time

    def process_frame(self, frame, source_name="webcam"):
        
        # Processes a single video frame by performing object tracking and face recognition.
        start_time = time.time()

        # Run YOLO tracking to detect and track objects (people, class 0).
        results = self.model.track(frame, persist=True, classes=[0],
                                     tracker="bytetrack.yaml", verbose=False)
        boxes = results[0].boxes

        # If no objects are detected, just update FPS and return the frame.
        if boxes.id is None:
            self.update_fps(time.time() - start_time)
            return frame

        current_time = time.time()

        # Define consistent colors and font settings for drawing.
        BORDER_COLOR = (0, 255, 0)     
        TEXT_COLOR = (0, 0, 0)          
        LABEL_BG_COLOR = (0, 255, 0)
        FONT = cv2.FONT_HERSHEY_SIMPLEX
        FONT_SCALE = 0.4               
        FONT_THICKNESS = 1            

        # FPS display settings.
        FPS_FONT_SCALE = 0.6            
        FPS_COLOR = (0, 255, 0)       
        FPS_THICKNESS = 2               
        FPS_MARGIN = 10                

        for i in range(len(boxes)):
            track_raw_id = int(boxes.id[i].item())
            track_id = f"{source_name}_{track_raw_id}"
            x1, y1, x2, y2 = map(int, boxes.xyxy[i].tolist())

            # Skip if bounding box coordinates are invalid.
            if x2 <= x1 or y2 <= y1:
                continue

            # Get the person ID for the current track.
            person_id = self.id_mappings.get(track_id)

            if person_id is None:
                # Extract face from the detected person
                cropped = frame[y1:y2, x1:x2]
                face_encoding = self.extract_face_encoding(frame, (x1, y1, x2, y2))
                
                # Try to find a matching person
                matching_person_id = self.find_matching_person(face_encoding)
                
                if matching_person_id:
                    # Found a match with existing person
                    person_id = matching_person_id
                    logger.info(f"Matched track {track_id} with existing person {person_id}")
                else:
                    # Create new person ID
                    person_id = str(self.person_id_counter)
                    self.person_id_counter += 1
                    logger.info(f"Created new person ID {person_id} for track {track_id}")
                
                # Map this track_id to the person_id
                self.id_mappings[track_id] = person_id
                
                # Add face encoding to the person's profile
                if face_encoding is not None:
                    self.add_face_encoding(person_id, face_encoding)
                
                # Save a snapshot of the person
                folder = f"tracked_people/{person_id}"
                Path(folder).mkdir(exist_ok=True)
                cv2.imwrite(f"{folder}/{source_name}_t{int(current_time)}.jpg", cropped)

            # Draw the main bounding box around the detected person.
            cv2.rectangle(frame, (x1, y1), (x2, y2), BORDER_COLOR, FONT_THICKNESS)

            # Prepare the label text with the person ID.
            label = f"ID: {person_id}"
            
            # Calculate the size of the label text.
            (label_width, label_height), baseline = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICKNESS)

            # Calculate coordinates for the label background box.
            label_bg_x1 = x1
            label_bg_y1 = y1 - label_height - 5
            label_bg_x2 = x1 + label_width + 5
            label_bg_y2 = y1

            # Draw the filled rectangle for the label background.
            cv2.rectangle(frame, (label_bg_x1, label_bg_y1), (label_bg_x2, label_bg_y2), LABEL_BG_COLOR, -1)

            # Draw the label text on the background.
            cv2.putText(frame, label, (x1 + 3, y1 - 5),
                                     FONT, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS)

        fps_text = f"FPS: {self.current_fps:.1f}"

        # Get text size to position it accurately.
        (fps_text_width, fps_text_height), fps_baseline = cv2.getTextSize(fps_text, FONT, FPS_FONT_SCALE, FPS_THICKNESS)

        # Position 'FPS_MARGIN' pixels from the right and 'FPS_MARGIN' pixels from the bottom.
        fps_x = frame.shape[1] - fps_text_width - FPS_MARGIN
        fps_y = frame.shape[0] - FPS_MARGIN

        cv2.putText(frame, fps_text, (fps_x, fps_y), FONT,
                                     FPS_FONT_SCALE, FPS_COLOR, FPS_THICKNESS)

        # Update FPS calculation for the current frame.
        self.update_fps(time.time() - start_time)
        return frame

    def process_webcam(self):
        
        # Initiates real-time person tracking and identification using a webcam feed.
        print("Starting real-time webcam tracking with face recognition. Press ESC to quit.")

        cap = cv2.VideoCapture(0) # Open the default webcam.

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30) 
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) 

        if not cap.isOpened():
            print("Error: Could not open webcam")
            return

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame from webcam")
                    break

                processed_frame = self.process_frame(frame, "webcam")

                cv2.imshow("Real-Time Person Tracking with Face Recognition", processed_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    break
                elif key == ord('s'):
                    self.save_data()
                    print("Data saved!")

        except KeyboardInterrupt:
            print("\nInterrupted by user")

        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.save_data()

    def process_video_playback(self, video_path):
        
        # Plays a video file with real-time person tracking and face recognition.
        if not os.path.exists(video_path):
            print(f"Error: Video file '{video_path}' not found")
            return

        print(f"Processing video: {video_path}")
        print("Press ESC to quit, SPACE to pause/resume, 's' to save data")

        cap = cv2.VideoCapture(video_path)
        video_name = Path(video_path).stem

        # Get video properties for playback control.
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Calculate the desired time per frame in milliseconds
        desired_frame_ms = 1000 / fps if fps > 0 else 33 # Fallback to ~30 FPS

        paused = False

        try:
            while True:
                frame_start_time = time.time() # Capture start time for current frame processing

                if not paused:
                    ret, frame = cap.read()
                    if not ret:
                        print("End of video reached")
                        break

                    processed_frame = self.process_frame(frame, video_name)

                    # Add playback controls information to the frame.
                    cv2.putText(processed_frame, "SPACE: Pause/Resume | ESC: Quit | S: Save",
                                         (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
                                         0.5, (255, 255, 255), 1) # Kept white for instructions

                    cv2.imshow("Video Playback - Person Tracking with Face Recognition", processed_frame)

                # Calculate time taken to process the frame
                processing_time_ms = (time.time() - frame_start_time) * 1000

                # Calculate the remaining delay to achieve desired FPS
                wait_time_ms = int(max(1, desired_frame_ms - processing_time_ms))

                key = cv2.waitKey(wait_time_ms) & 0xFF
                if key == 27:
                    break
                elif key == ord(' '):
                    paused = not paused
                    print("Paused" if paused else "Resumed")
                elif key == ord('s'):
                    self.save_data()
                    print("Data saved!")

        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.save_data()

    def save_data(self):
        
        # Saves the current ID mappings and face embeddings to JSON files.
        self.save_json(self.MAPPINGS_PATH, self.id_mappings)
        self.save_embeddings()
        logger.info("Tracking data and face embeddings saved successfully")

    def list_video_files(self):
        # Lists available video files in the 'video_footage' directory.
        video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.wmv', '*.flv', '*.webm']
        video_files = []

        for ext in video_extensions:
            video_files.extend(Path("video_footage").glob(ext))
            video_files.extend(Path("video_footage").glob(ext.upper()))

        return sorted([str(f) for f in video_files])

    def load_existing_snapshots(self):
       
        print("Loading existing snapshots for face recognition...")
        
        if not os.path.exists("tracked_people"):
            print("No existing snapshots found.")
            return
        
        for person_dir in os.listdir("tracked_people"):
            person_path = Path("tracked_people") / person_dir
            if not person_path.is_dir():
                continue
                
            try:
                person_id = person_dir  # Assuming directory name is the person ID
                print(f"Processing snapshots for person {person_id}...")
                
                # Process all images in the person's directory
                image_files = []
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                    image_files.extend(person_path.glob(ext))
                    image_files.extend(person_path.glob(ext.upper()))
                
                for image_path in image_files:
                    try:
                        # Load image
                        image = cv2.imread(str(image_path))
                        if image is None:
                            continue
                        
                        # Extract face encoding
                        face_encoding = self.extract_face_encoding(image)
                        if face_encoding is not None:
                            self.add_face_encoding(person_id, face_encoding)
                            
                    except Exception as e:
                        logger.warning(f"Error processing image {image_path}: {e}")
                        continue
                
                if person_id in self.face_embeddings:
                    print(f"Loaded {len(self.face_embeddings[person_id])} face encodings for person {person_id}")
                    
            except Exception as e:
                logger.warning(f"Error processing person directory {person_dir}: {e}")
                continue
        
        print(f"Loaded face encodings for {len(self.face_embeddings)} people")
        self.save_embeddings()  # Save the loaded embeddings

def main():
    # Main function to run the Real-Time Person Tracking System.
    tracker = RealTimeTracker()
    
    # Load existing snapshots for face recognition
    tracker.load_existing_snapshots()

    print("Real-Time Person Tracking System with Face Recognition")
    print("=" * 60)
    print("Features:")
    print("- Real-time person tracking")
    print("- Face recognition using existing snapshots")
    print("- Persistent person identification across sessions")
    print("- Automatic new person ID assignment")
    print("=" * 60)

    while True:
        print("\nChoose an option:")
        print("1. Real-time webcam tracking")
        print("2. Video playback with tracking")
        print("3. List available videos")
        print("4. Reload existing snapshots")
        print("5. Exit")

        choice = input("\nEnter your choice (1-5): ").strip()

        if choice == '1':
            tracker.process_webcam()

        elif choice == '2':
            video_files = tracker.list_video_files()
            if not video_files:
                print("No video files found in 'video_footage' directory")
                print("Please add video files to the 'video_footage' folder")
                continue

            print("\nAvailable videos:")
            for i, video in enumerate(video_files, 1):
                print(f"{i}. {Path(video).name}")

            try:
                video_choice = int(input(f"\nSelect video (1-{len(video_files)}): ")) - 1
                if 0 <= video_choice < len(video_files):
                    tracker.process_video_playback(video_files[video_choice])
                else:
                    print("Invalid selection")
            except ValueError:
                print("Please enter a valid number")

        elif choice == '3':
            video_files = tracker.list_video_files()
            if video_files:
                print(f"\nFound {len(video_files)} video files:")
                for i, video in enumerate(video_files, 1):
                    size = os.path.getsize(video) / (1024*1024)   # MB
                    print(f"{i}. {Path(video).name} ({size:.1f} MB)")
            else:
                print("No video files found in 'video_footage' directory")

        elif choice == '4':
            tracker.load_existing_snapshots()
            print("Existing snapshots reloaded!")

        elif choice == '5':
            print("Exiting...")
            break

        else:
            print("Invalid choice. Please try again.")

    # Final save of all tracking data.
    tracker.save_data()
    print("All data saved successfully!")

if __name__ == "__main__":
    main()