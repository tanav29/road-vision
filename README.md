 # 🚗 Advanced Driver Assistance System (ADAS)

 A comprehensive real-time driver assistance system that combines **YOLOv8 object detection** with **computer vision-based lane detection** to provide intelligent driving assistance and safety alerts.

![ADAS Preview](https://github.com/user-attachments/assets/c15a9601-6340-4369-bdbb-969ba44f9aa7)

## 🎯 Project Overview

This ADAS system provides real-time monitoring and assistance for drivers by detecting:

- **Objects**: Cars, pedestrians, traffic signs, and other vehicles
- **Lane Departure**: Real-time lane detection and departure warnings
- **Proximity Alerts**: Audio-visual warnings for objects too close to the vehicle
- **Safety Monitoring**: Continuous assessment of driving environment

## ✨ Key Features

 ### 🎯 Object Detection & Classification

 - **YOLOv8 Model**: Trained on car camera photos dataset in Google Colab
- **Multi-class Detection**: Cars, pedestrians, traffic signs, and more
- **Real-time Processing**: 30+ FPS on standard hardware
- **Confidence-based Filtering**: Adjustable detection thresholds

### 🛣️ Lane Detection System

- **Computer Vision Pipeline**: Canny edge detection + Hough transform
- **Region of Interest (ROI)**: Focused analysis on driving lanes
- **Lane Departure Warning**: Visual indicators for lane boundaries
- **Adaptive Processing**: Handles various road conditions and lighting

### 🔊 Safety Alerts & Warnings

- **Proximity Detection**: Audio alerts for objects too close to vehicle
- **Visual Overlays**: Real-time bounding boxes and labels
- **Smart Thresholding**: Distance-based warning system
- **Multi-threaded Audio**: Non-blocking alert system

### 📊 Real-time Analytics

- **Object Counting**: Live count of detected vehicles and signs
- **Performance Metrics**: FPS monitoring and detection statistics
- **Visual Feedback**: Color-coded detection results

## 🛠️ Tech Stack

### Core Technologies

- **Python 3.8+**: Primary programming language
- **OpenCV 4.x**: Computer vision and image processing
 - **Ultralytics YOLOv8**: State-of-the-art object detection
- **NumPy**: Numerical computing and array operations

### Computer Vision

- **Canny Edge Detection**: Lane boundary identification
- **Hough Transform**: Line detection in images
- **Gaussian Blur**: Noise reduction and smoothing
- **Region of Interest (ROI)**: Focused analysis areas

### Audio & UI

- **playsound**: Audio alert system
- **OpenCV GUI**: Real-time video display
- **Threading**: Non-blocking audio playback

 ### Data Processing
 - **Pandas**: Data manipulation and analysis
 - **Matplotlib**: Visualization and plotting
 - **Plotly**: Interactive data visualization

 ## 🧠 Model Training

 The YOLOv8 model was trained using Google Colab for its free GPU resources. The training process involved:

 - **Dataset**: Car camera photos with annotated objects (cars, pedestrians, traffic signs)
 - **Environment**: Google Colab with CUDA-enabled GPU
 - **Framework**: Ultralytics YOLOv8
 - **Training Script**: Available in `model.ipynb`

 ### Challenges and Difficulties

 Training the YOLOv8 model in Google Colab presented several challenges:

 - **Session Timeouts**: Colab's 12-hour session limit required careful checkpointing and resuming training
 - **GPU Availability**: Limited free GPU time necessitated efficient training schedules
 - **Data Upload**: Large datasets required uploading to Google Drive or using cloud storage
 - **Memory Constraints**: Balancing batch size with Colab's RAM limitations
 - **Model Optimization**: Tuning hyperparameters for real-time performance on edge devices
 - **Integration**: Ensuring the trained model integrates seamlessly with the lane detection and alert systems
 - **Data Quality**: Collecting and annotating diverse driving scenarios for robust detection

 ## 📁 Project Structure

```
adas/
 ├── model.pt                 # Trained YOLOv8 model
├── model.ipynb             # Model training notebook
├── README.md               # Project documentation
├── sounds/
│   └── warning.mp3        # Audio alert file
└── testing/
    ├── frames.py          # Frame-by-frame processing
    ├── lane.py            # Lane detection algorithms
    └── webcam.py          # Real-time webcam processing
```

## 🚀 Quick Start

### Prerequisites

```bash
# Install Python dependencies
pip install ultralytics opencv-python numpy pandas matplotlib plotly playsound
```

### Running the System

#### Browser game screen capture

1. Open the game in your browser and position it on your primary monitor.
2. Run `run-browser-game.bat` from the project folder, or run
   `.venv\Scripts\python.exe testing\webcam.py --screen` from the project root.
3. Drag a rectangle around the game in the selection window, then press Enter or
   Space. Press Esc to cancel.
4. Keep the game visible at the same position and play as usual. Detections appear
   in a separate OpenCV window; keep that window outside the selected game area.
   Click the detection window and press `q` to stop. Select the area again after
   moving or resizing the game.

This captures screen pixels, so another window covering the game will also be
captured. The model was trained on real driving images, so game graphics and
third-person views may require tuning and can reduce accuracy. Warnings are based
on bounding-box size, not measured distance.

The detector defaults to at most 30 frames per second and a 320-pixel inference
size to reduce resource use. To adjust them, launch from the project root, for
example:

```powershell
.venv\Scripts\python.exe testing\webcam.py --screen --fps 3 --imgsz 416
```

Lower values generally use fewer resources, but can miss fast-moving or small
objects. Actual frame rate can be lower than the limit if processing takes longer.

#### Experimental Vice City autonomous mode (Windows)

This is an experimental game controller, **not for real vehicles**. It assumes a
front-facing in-car view and the game's WASD controls (W accelerates, S brakes/reverses, A/D steer). It cannot read speed, detect every obstacle,
or navigate intersections; the real-road-trained object model may miss game objects.
Test in a clear area at low speed and be ready to take over. S can reverse the car
when it is already stopped.

1. Open the browser game on the primary screen, choose the in-car view, and leave it
   visible. Run from the project root:

   ```powershell
   .venv\Scripts\python.exe testing\webcam.py --screen --autodrive
   ```

2. Select the game rectangle and press Enter. Keep the preview outside that rectangle.
   Click the game to focus it, then press **Ctrl+Alt+G** to arm. The preview displays the
   controller state and detected lane boundaries in a resizable, larger preview. The
processing FPS shown is before the target-rate wait. Both lane boundaries are required;
   when lanes are missing it releases the driving keys. When a large vehicle/person
   is detected ahead it applies short brake pulses. Steering and acceleration are
   also pulsed. Release Ctrl and Alt after arming before the car can move.
3. Press **Ctrl+Alt+X globally** to immediately disarm and release the keys. Ctrl+Alt+G re-arms only
   while the selected game window has focus. Losing focus or an inference timeout
   also disarms and releases keys. Click the preview and press `q`, close it, or use
   Ctrl+C to exit. Stop before moving/resizing or covering the capture area; select it
   again after restarting. The watchdog depends on the Python process and Windows;
   a forcibly killed process or system failure cannot guarantee key release. If needed,
   press and release W, A, S and D manually to clear them.

The controller uses screen-space heuristics rather than measured distances; missing
markings, shadows, HUD elements, unusual roads, and false detections can cause bad
steering or braking. It has not been calibrated to a particular game installation.

#### 1. Real-time Webcam Processing

```bash
cd testing
python webcam.py
```

#### 2. Frame-by-frame Analysis

```bash
cd testing
python frames.py
```

## 🔧 Configuration

### Model Parameters

- **Confidence Threshold**: `conf=0.5` (adjustable)
- **Detection Classes**: Cars, pedestrians, traffic signs
- **Processing Speed**: Real-time (30+ FPS)

### Lane Detection Settings

- **ROI Coordinates**: Focused on lower 60% of frame
- **Edge Detection**: Canny thresholds (180, 240)
- **Line Detection**: Hough transform parameters optimized

### Alert System

- **Proximity Threshold**: 800 pixels (adjustable)
- **Audio Frequency**: 1-second cooldown between alerts
- **Visual Warnings**: Color-coded alert system

## 📊 Performance Metrics

### Detection Accuracy

- **Object Detection**: 95%+ accuracy on test dataset
- **Lane Detection**: Robust across various road conditions
- **Real-time Processing**: 30+ FPS on standard hardware

### System Requirements

- **CPU**: Intel i5 or equivalent
- **RAM**: 8GB minimum, 16GB recommended
- **GPU**: Optional (CUDA support for faster inference)
- **Storage**: 2GB for model and dependencies

## 🎯 Use Cases

### 🚗 Automotive Applications

- **Dashcam Integration**: Real-time monitoring systems
- **Fleet Management**: Commercial vehicle safety
- **Driver Training**: Educational and training platforms
- **Research & Development**: ADAS algorithm development

### 🏭 Industrial Applications

- **Warehouse Safety**: Forklift and pedestrian detection
- **Construction Sites**: Heavy machinery safety
- **Security Systems**: Perimeter monitoring
- **Quality Control**: Manufacturing process monitoring

## 🔬 Technical Implementation

### Object Detection Pipeline

1. **Frame Capture**: Real-time video input
2. **Preprocessing**: Resize and normalize
 3. **YOLOv8 Inference**: Object detection and classification
4. **Post-processing**: Confidence filtering and NMS
5. **Visualization**: Bounding boxes and labels

### Lane Detection Algorithm

1. **Grayscale Conversion**: Color to intensity mapping
2. **Gaussian Blur**: Noise reduction
3. **Canny Edge Detection**: Boundary identification
4. **ROI Masking**: Focus on driving lanes
5. **Hough Transform**: Line detection
6. **Slope Analysis**: Lane classification (left/right)
7. **Visual Overlay**: Lane marking display

### Safety Alert System

1. **Proximity Calculation**: Distance-based analysis
2. **Threshold Checking**: Configurable alert triggers
3. **Multi-threaded Audio**: Non-blocking alerts
4. **Visual Feedback**: On-screen warnings

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines for:

- Bug reports and feature requests
- Code improvements and optimizations
- Documentation enhancements
- Performance optimizations

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

 - **Ultralytics**: YOLOv8 implementation
- **OpenCV**: Computer vision library
- **Kaggle Dataset**: Car camera photos for training
- **Research Community**: Computer vision and ADAS research

## 📞 Support

For questions, issues, or contributions:

- Create an issue on GitHub
- Contact the development team
- Check documentation for common solutions

---

**Built with ❤️ for safer roads and smarter driving**

## Testing GTA Vice City with the rotating minimap

On Windows, open the game on the primary screen, switch to a forward-facing
in-car view and run this command from the repository root.

```powershell
.venv\Scripts\python.exe testing\webcam.py --screen --autodrive --map
```

Select the visible game rectangle and press Enter. Inside that still image,
select a tight square around the entire circular radar and press Enter again.
The radar can face any direction. Keep both preview windows outside the captured
area. Click the game and press Ctrl+Alt+G, then release the modifiers to drive.
Ctrl+Alt+X disarms globally. If you move or resize the game, restart and select
the two rectangles again.

The map preview searches the supplied Vice City map across rotations and zoom
levels in a background worker. The blue dot marks a recent position estimate.
?Searching for position? means the matcher has no recent, sufficiently distinct
match. Sparse roads, HUD icons, and differences between editions can prevent a
fix. The angle is the rotation used to align the radar with the map. Click the map
to mark a destination. Driving follows visible lane markings and reacts to detected
obstacles independently of the marker. Destination routing is still experimental
work for the future.

For a map-only session run `python testing/navigation.py` in the project
environment. For detection and localization without sending game keys, use
`python testing/webcam.py --screen --map`.
