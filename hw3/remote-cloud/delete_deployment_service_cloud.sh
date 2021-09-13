#!/bin/bash
kubectl delete deployment mosquitto
kubectl delete deployment processor
kubectl delete service mosquitto-service