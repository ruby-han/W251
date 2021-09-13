#!/bin/bash
kubectl delete deployment mosquitto
kubectl delete deployment face_detection
kubectl delete deployment logger
kubectl delete deployment forwarder
kubectl delete service mosquitto-service