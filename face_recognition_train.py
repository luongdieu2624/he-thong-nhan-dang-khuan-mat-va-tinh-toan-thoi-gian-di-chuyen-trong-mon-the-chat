import face_recognition
import os
import numpy as np
from PIL import Image

def load_images_from_folder(folder):
    """
    Load images from folder and return face encodings with labels
    """
    encodings = []
    labels = []
    
    for filename in os.listdir(folder):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.webp')):
            # Get image path
            image_path = os.path.join(folder, filename)
            
            # Load image
            image = face_recognition.load_image_file(image_path)
            
            # Get face locations
            face_locations = face_recognition.face_locations(image)
            
            if len(face_locations) > 0:
                # Get face encodings
                face_encoding = face_recognition.face_encodings(image, face_locations)[0]
                
                # Get label from filename (remove extension)
                # Nếu có dấu gạch ngang thì lấy phần sau dấu gạch ngang
                # Nếu không có thì lấy tên file không có extension
                base_name = os.path.splitext(filename)[0]
                label = base_name.split('-')[1] if '-' in base_name else base_name
                
                encodings.append(face_encoding)
                labels.append(label)
                print(f"Processed {filename}")
            else:
                print(f"No face found in {filename}")
                
    return np.array(encodings), np.array(labels)

def save_encodings(encodings, labels, file_path):
    """
    Save encodings and labels to file
    """
    np.savez(file_path, encodings=encodings, labels=labels)
    print(f"Saved encodings to {file_path}")

def load_encodings(file_path):
    """
    Load encodings and labels from file
    """
    data = np.load(file_path)
    return data['encodings'], data['labels']

def recognize_face(image_path, known_encodings, known_labels, tolerance=0.6):
    """
    Recognize face in image by comparing with known encodings
    """
    # Load image
    image = face_recognition.load_image_file(image_path)
    
    # Get face locations and encodings
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)
    
    results = []
    
    for face_encoding in face_encodings:
        # Compare face with known encodings
        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance)
        
        if True in matches:
            # Get indexes of matching faces
            matched_idxs = [i for i, match in enumerate(matches) if match]
            
            # Get distances to all known faces
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)
            
            # Get best match (smallest distance)
            best_match_idx = np.argmin(face_distances[matched_idxs])
            name = known_labels[matched_idxs[best_match_idx]]
        else:
            name = "Unknown"
            
        results.append(name)
        
    return results, face_locations

if __name__ == "__main__":
    # Directory containing face images
    dataset_dir = "dataset"
    
    # Train model
    print("Training model...")
    encodings, labels = load_images_from_folder(dataset_dir)
    
    # Save encodings
    save_encodings(encodings, labels, "face_encodings.npz")
    print(f"Saved {len(encodings)} face encodings")
    
    # Test recognition
    print("\nTesting recognition...")
    test_image = "dataset/1.webp"  # Change this to test different images
    
    # Load saved encodings
    known_encodings, known_labels = load_encodings("face_encodings.npz")
    
    # Recognize faces
    names, locations = recognize_face(test_image, known_encodings, known_labels)
    
    print(f"\nFound {len(names)} faces in test image:")
    for name, location in zip(names, locations):
        print(f"Found {name} at location {location}")