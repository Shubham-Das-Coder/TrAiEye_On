import cv2
import numpy as np
from PIL import Image, ImageGrab

def overlay_image_alpha(img, img_overlay, pos):
    """Overlay img_overlay on top of img at the position specified by pos."""
    x, y = pos
    h, w = img_overlay.shape[0:2]

    # Ensure overlay is within image bounds
    y1, y2 = max(0, y), min(img.shape[0], y + h)
    x1, x2 = max(0, x), min(img.shape[1], x + w)

    # Extract the overlay region
    overlay_image = img_overlay[0:y2 - y1, 0:x2 - x1]

    # If the overlay has an alpha channel
    if overlay_image.shape[2] == 4:
        alpha = overlay_image[:, :, 3] / 255.0
        alpha_inv = 1.0 - alpha

        # Blend images using alpha channel
        for c in range(0, 3):
            img[y1:y2, x1:x2, c] = (alpha * overlay_image[:, :, c] + alpha_inv * img[y1:y2, x1:x2, c])
    else:
        img[y1:y2, x1:x2] = overlay_image[:, :, :3]  # No alpha, just copy

    return img

def get_pasted_image():
    """Get an image from the clipboard if present."""
    img = ImageGrab.grabclipboard()
    if isinstance(img, Image.Image):
        img = img.convert("RGBA")  # Ensure it has an alpha channel
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGRA)
    return None

def create_sidebar(glasses_images, current_glasses, frame_height):
    sidebar = np.zeros((frame_height, 100, 3), dtype=np.uint8)
    if len(glasses_images) == 0:
        return sidebar  # Return empty sidebar if no glasses are available

    thumbnail_height = frame_height // len(glasses_images)
    for i, img in enumerate(glasses_images):
        y = i * thumbnail_height
        resized = cv2.resize(img[:, :, :3], (80, thumbnail_height - 20))
        sidebar[y + 10:y + thumbnail_height - 10, 10:90] = resized
        if i == current_glasses:
            cv2.rectangle(sidebar, (5, y + 5), (95, y + thumbnail_height - 5), (0, 255, 0), 2)
    return sidebar

def fit_glasses_to_face(glasses_img, face_rect):
    """Fit glasses to the detected face rectangle."""
    x, y, w, h = face_rect
    glasses_width = w  # Fit glasses width to face width
    glasses_height = int(h / 4)  # Adjust height accordingly

    # Resize the glasses image to fit the face
    fitted_glasses = cv2.resize(glasses_img, (glasses_width, glasses_height))
    return fitted_glasses

def main():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)

    glasses_images = []
    current_glasses = 0

    cv2.namedWindow('Glasses Try-On App')
    cv2.resizeWindow('Glasses Try-On App', 800, 600)  # Set a larger window size

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # Check if the user wants to add a glasses image
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('a'):  # Press 'a' to add the image from clipboard
            pasted_img = get_pasted_image()
            if pasted_img is not None:
                glasses_images.append(pasted_img)  # Add the pasted image directly
                current_glasses = len(glasses_images) - 1  # Automatically select the newly pasted image

        sidebar = create_sidebar(glasses_images, current_glasses, frame.shape[0])  # Call sidebar creation

        if glasses_images:
            for (x, y, w, h) in faces:
                fitted_glasses = fit_glasses_to_face(glasses_images[current_glasses], (x, y, w, h))
                glasses_pos = (x, y + int(h / 4))  # Positioning glasses to fit on the face
                frame = overlay_image_alpha(frame, fitted_glasses, glasses_pos)

            # Stack sidebar and display the frame
            combined_frame = np.hstack((frame, sidebar))
            cv2.imshow('Glasses Try-On App', combined_frame)
        else:
            # Display just the frame if no glasses are added yet
            cv2.imshow('Glasses Try-On App', frame)

        if key == ord('q') or key == 27:  # 27 is the ESC key code
            break
        elif key == ord('n') and glasses_images:
            current_glasses = (current_glasses + 1) % len(glasses_images)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
