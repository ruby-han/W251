sudo systemctl start k3s

kubectl apply -f broker/mosquittoService.yaml
kubectl apply -f broker/mosquitto.yaml
kubectl apply -f face_detection/face_detection.yaml
kubectl apply -f logger/logger.yaml
kubectl apply -f forwarder/forwarder.yaml
kubectl get service

sleep 10

kubectl get pods