# Average Face Generator

## Overview
This project involves creating an average face by merging and enhancing facial images using computer vision techniques. The key steps include face detection, cropping to the detected face, and triangulation-based image warping.

## Requirements
- Python
- OpenCV
- DLib
- Numpy
- Scipy
To install the required libraries, run this command in the terminal:
```bash
pip install -r requirements.txt
```

## Getting Started
1. ### Clone the repository:
   ```bash
   git clone https://github.com/Eliaskhal/Average-Face.git
   ```
2. ### Download the shape predictor file:
Download the "shape_predictor_68_face_landmarks.dat" file and place it in the project directory.

3. ### Add your images:
Place the facial images you want to merge in the "pics" directory.

Run the script:
```bash
python main.py
```
## Output
The processed images will be saved in the "output" directory. The final average face will be saved as "average.jpeg."

## Notes
- The algorithm uses facial landmarks and Delaunay triangulation for image warping.
- Adjustments to cropping and warping parameters can be made in the script.
- The average face is computed from the processed images.
Feel free to experiment and improve the results!

Make sure to include the shape predictor file as specified in the "Getting Started" section. Also, provide any additional details or instructions if needed.
