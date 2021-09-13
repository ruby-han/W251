docker build -t rubyhan/face_detection:v1 -f Dockerfile.face_detection .
docker push rubyhan/face_detection:v1
docker run -ti --network=host --rm --privileged -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix rubyhan/face_detection:v1