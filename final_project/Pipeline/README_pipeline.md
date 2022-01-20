## The following directions can be used to Spin up the Inference Pipeline. The following actions must be run in order. 

### Setting up the MQTT Stream
- Start mqtt broker: `kubectl apply -f mosquitto.yaml`
- Start mqtt service: `kubectl apply -f mosquittoService.yaml`
- Check mqtt status in kubernetes: `kubectl get all`
- Set up listener in cloud: `docker run -ti --network host --rm --privileged -e DISPLAY=$DISPLAY -v /tmp/.X11-unix/:/tmp/.X11-unix lensgerrit/mosquitto_x86:v1`

### Setting up Cloud "Listener" 
- `docker run -ti --network host --rm --privileged -e DISPLAY=$DISPLAY -v /data:/data lensgerrit/listener_fp:v3`

### Setting up Edge "Forwarder"
- `kubectl apply -f forwarder.yaml`
- check to make sure forwarder is live: `kubectl get all`
  - should also see "New client connected..." in mqtt docker container
- Subscribe to mqtt locally (Edge) to make sure messages are being transmitted
    - `mosquitto_sub -h localhost -p 32003 -t lesion`

### Setting up Edge "Publisher"
Spin up the cluster: `kubectl apply -f publisher.yaml`

### Running the inference model on the Cloud
- In the cloud docker container, start `jupyter notebook`, and run all cells in `inference-notebook-2.ipynb`
