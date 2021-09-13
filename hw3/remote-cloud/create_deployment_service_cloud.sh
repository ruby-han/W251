#!/bin/bash
kubectl apply -f ~/github/w251/hw3/remote-cloud/broker/mosquitto-deployment.yaml
kubectl apply -f ~/github/w251/hw3/remote-cloud/broker/mosquitto-service.yaml
kubectl apply -f ~/github/w251/hw3/remote-cloud/processor/processor-deployment.yaml
