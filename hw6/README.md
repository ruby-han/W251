# Fall 2021 W251 Homework 6 - GStreamer and Model Optimization

## Part 1: GStreamer

1. Convert the following sink to use nveglglessink.
```
gst-launch-1.0 v4l2src device=/dev/video0 ! xvimagesink
```

**Answer:**
```
gst-launch-1.0 v4l2src device=/dev/video0 ! nvvidconv ! nvegltransform ! nveglglessink -e
```

2. What is the difference between a property and a capability? How are they each expressed in the pipeline?

**Answer:**
A property is used to describe extra information for a capability and is used to modify or configure an element behavior, separated by spaces.

A capability describes the type of data streamed between two pads (element's interface to the outside world), separated by commas.

3. Explain the following pipeline, that is explain each piece of the pipeline, describing if it is an element (if so, what type), property, or capability. What does this pipeline do?
```
gst-launch-1.0 v4l2src device=/dev/video0 ! video/x-raw, framerate=30/1 ! videoconvert ! agingtv scratch-lines=10 ! videoconvert ! xvimagesink sync=false
```

**Answer:**
- `gst-launch-1.0`: Tool that runs GStreamer pipeline
- `v4l2src`: Video source element such as a web camera
- `device=/dev/video0`: Property that sets device location at /dev/video0
- `video/x-raw`: Capability that sets format to x-raw video feed
- `framerate=30/1`: Capability that sets frame rate of 30 fps
- `videoconvert`: Filter element that converts video format
- `agingtv scratch-lines=10`: Filter element that adds aging effect to video 
- `scratch-lines=10`: Property that adds scratch lines of 10 
- `videoconvert`: Filter element that converts video format
- `xvimagesink`: Sink element that outputs video
- `sync=false`: Property that pushes images to display immediately without sync-ing on the clock (higher CPU utilization)

4. GStreamer pipelines may also be used from Python and OpenCV. Write a Python application that listens for images streamed from a Gstreamer pipeline ensuring that image displays in color.

```
gst-launch-1.0 v4l2src device=/dev/video0 ! video/x-raw, framerate=30/1, width=640,height=480  ! nvvidconv ! omxh265enc insert-vui=1 ! h265parse ! rtph265pay config-interval=1 ! udpsink host=192.168.1.81 port=5000 sync=false -e 
```