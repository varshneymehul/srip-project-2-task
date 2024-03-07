import cv2
import numpy as np


def process_frame(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Perform Fourier Transform to get frequency domain from time domain, get frequencies of pixel intensities to analyse slow moving components (0-1 Hz)
    f_transform = np.fft.fft2(gray)
    f_transform_shifted = np.fft.fftshift(f_transform)

    # Define the frequency range for filtering (0-1 Hz)
    rows, cols = gray.shape
    center_row, center_col = rows // 2, cols // 2
    freq_range = 1  # Frequency range in Hz
    cutoff_frequency = freq_range / 2
    cutoff_pixel_radius = int(cutoff_frequency * min(rows, cols) / 2)

    # Create a circular mask to filter the desired frequency range
    mask = np.zeros((rows, cols), np.uint8)
    cv2.circle(mask, (center_col, center_row), cutoff_pixel_radius, 1, -1)

    # Apply the mask to the shifted Fourier Transform which is used to bring the lower frequencies to the center of distribution
    f_transform_shifted_filtered = f_transform_shifted * mask

    # Inverse Fourier Transform to obtain the filtered image (get spatial domain)
    filtered_image = np.abs(
        np.fft.ifft2(np.fft.ifftshift(f_transform_shifted_filtered))
    )

    # Thresholding to obtain a binary image
    _, binary_image = cv2.threshold(
        filtered_image.astype(np.uint8), 50, 255, cv2.THRESH_BINARY
    )

    # Connected components analysis
    _, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_image, connectivity=8
    )

    # Draw bounding boxes around the connected components
    for i in range(1, len(stats)):
        x, y, w, h, area = stats[i]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return frame


# Open the video file
video_path = "1705951007967.mp4"
cap = cv2.VideoCapture(video_path)

# Check if the video opened successfully
if not cap.isOpened():
    print("Error opening video file")
    exit()

# Create a VideoWriter object for output 
out = cv2.VideoWriter(
    "output.avi",
    cv2.VideoWriter_fourcc(*"XVID"),
    30,
    (int(cap.get(3)), int(cap.get(4))),
)

# Process each frame in the video
while True:
    ret, frame = cap.read()

    if not ret:
        break  # Breaks the loop if the video is finished

    # Process the frame
    processed_frame = process_frame(frame)

    # Display the processed frame
    cv2.imshow("Thermal video analysis", processed_frame)

    # Write the frame to the output video file 
    out.write(processed_frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(25) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
out.release()  # Release the VideoWriter if used
cv2.destroyAllWindows()
