import cv2
import numpy as np


def nothing(x):
  pass


def main():

  image_path = 'tiger.jpg'
  if not image_path:
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    cv2.putText(
        img,
        "Sample Image",
        (50, 200),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
    )
  else:
    img = cv2.imread(image_path)
    if img is None:
      print("Error: Could not load image. Please check the file path.")
      return

  max_dim = 600
  h, w = img.shape[:2]
  if max(h, w) > max_dim:
    scale = max_dim / max(h, w)
    img = cv2.resize(img, (int(w * scale), int(h * scale)))

  window_name = "Edge Detection & Filtering Interactive App"
  cv2.namedWindow(window_name)

  cv2.createTrackbar("Filter Type\n(0:None,1:Gauss,2:Med)", window_name, 0, 2, nothing)

  cv2.createTrackbar("Kernel Size (val*2+1)", window_name, 1, 7, nothing)

  cv2.createTrackbar("Edge Detector\n(0:None,1:Canny,2:Sobel,3:Lap)", window_name, 1, 3, nothing)

  cv2.createTrackbar("Canny Low Thresh", window_name, 100, 255, nothing)
  cv2.createTrackbar("Canny High Thresh", window_name, 200, 255, nothing)

  print("\n[INFO] Interactive window opened. Press 'q' on your keyboard to exit.")

  while True:
    f_type = cv2.getTrackbarPos("Filter Type\n(0:None,1:Gauss,2:Med)", window_name)
    k_val = cv2.getTrackbarPos("Kernel Size (val*2+1)", window_name)
    kernel_size = (
        k_val * 2 + 1
    )  
    e_method = cv2.getTrackbarPos("Edge Detector\n(0:None,1:Canny,2:Sobel,3:Lap)", window_name)
    canny_low = cv2.getTrackbarPos("Canny Low Thresh", window_name)
    canny_high = cv2.getTrackbarPos("Canny High Thresh", window_name)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if f_type == 1:
      filtered = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    elif f_type == 2:

      med_k = kernel_size if kernel_size >= 3 else 3
      filtered = cv2.medianBlur(gray, med_k)
    else:
      filtered = gray

    if e_method == 1:
      edges = cv2.Canny(filtered, canny_low, canny_high)
    elif e_method == 2:
      sobel_x = cv2.Sobel(filtered, cv2.CV_64F, 1, 0, ksize=kernel_size)
      sobel_y = cv2.Sobel(filtered, cv2.CV_64F, 0, 1, ksize=kernel_size)
      sobel_mag = cv2.magnitude(sobel_x, sobel_y)
      edges = cv2.normalize(
          sobel_mag, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
      )
    elif e_method == 3:
      lap = cv2.Laplacian(filtered, cv2.CV_64F, ksize=kernel_size)
      edges = cv2.normalize(lap, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    else:
      edges = filtered

    filtered_bgr = cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    combined_view = np.hstack((img, filtered_bgr, edges_bgr))

    cv2.imshow(window_name, combined_view)

    if cv2.waitKey(1) & 0xFF == ord("q"):
      break

  cv2.destroyAllWindows()


if __name__ == "__main__":
  main()