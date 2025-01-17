import cv2
import threading
import encoder
import ctypes
def encode_image(image_path, format, quality):
  # Load the image using OpenCV
  image = cv2.imread(image_path)
  if image is None:
      print(f"Failed to load image: {image_path}")
      return

  # Encode the image using the C++ extension
  try:
       
        
        media_type, size, encoded_image = encoder.encode_image_binary(image, format, quality)


        print(f"Encoded {image_path} to {format.upper()}, media type: {media_type}, size {size} bytes, pointer: {hex(encoded_image)} ")
        g = (ctypes.c_char*size).from_address(encoded_image)
        with open(f'{format}.{format}', 'wb') as file:
            # Write the bytearray to the file
            file.write(g)



  except Exception as e:
      print(f"Error encoding {image_path}: {e}")

if __name__ == "__main__":
  # List of image paths and formats
  images = [
    #   ("/home/hamit/Downloads/images/20.png", "jpeg", 90),
      ("/home/hamit/Downloads/images/21.png", "jpeg", 3),
       ("/home/hamit/Downloads/images/22.png", "jpeg", 3),
        # ("/home/hamit/Downloads/images/23.png", "jpeg", 90),
  ]

  # Create and start threads for each image
  threads = []
  for image_path, format, quality in images:
      t = threading.Thread(target=encode_image, args=(image_path, format, quality))
      threads.append(t)
      t.start()

  # Wait for all threads to complete
  for t in threads:
      t.join()

  print("Completed encoding all images.")
