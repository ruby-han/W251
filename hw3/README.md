# Fall 2021 W251 Homework 3 - Containers, Kubernetes and IoT/Edge

## Objectives:
- Build a lightweight containerized application pipeline with components running on the edge (Jetson device) and in the cloud (VM in AWS)
    - Edge components include Nvidia Jetson device and a webcam which uses a face recognition algorithm to detect and capture faces
- Write application in a modular/cloud native way to be run on any edge device or hub and any cloud VM
- Deploy using MQTT, Kubernetes and Docker
- The .png image files are sent to an EC2 instance on AWS and stored in a S3 object storage

## Overall Architecture Flow:

![image](./hw3.png)

### Local - Edge Device
**USB Camera**

**Kubernetes - Jetson**
1. Face Detector 
    - System: Ubuntu
    - Repo files:
        - `Dockerfile.face_detection`
        - `face_detection.py`
        - `face_detection.yaml`
2. Message Logger
    - System: Alpine
    - Repo files:
        - `Dockerfile.logger`
        - `logger.py`
        - `logger.yaml`
3. MQTT Broker
    - System: Alpine
    - Repo files:
        - `Dockerfile.mosquitto`
        - `mosquitto.yaml`
        - `mosquittoService.yaml`
4. MQTT Forwarder
    - System: Alpine
    - Repo files:
        - `Dockerfile.forwarder`
        - `forwarder.py`
        - `forwarder.yaml`
        
### Remote - Cloud
**AWS EC2 t2.large**
1. MQTT Broker
    - System: Alpine
    - Repo files:
        - `Dockerfile.mosquitto_x86`
        - `mosquitto_x86.yaml`
        - `mosquittoService_x86.yaml`
2. Image Processor 
    - System: Ubuntu
    - Repo files:
        - `Dockerfile.processor`
        - `processor.py`
        - `processor.yaml`

### S3 Object Storage
- `rubyhan-w251-hw3`

## Order of Operations
**Local - Edge Device**
1. Build Docker containers

Example using shell script:
```
docker build -t rubyhan/face_detection:v1 -f Dockerfile.face_detection .
docker push rubyhan/face_detection:v1
```
2. Deploy container into Kubernetes

Example using shell script:
```
kubectl apply -f broker/mosquitto.yaml
```

**Remote - Cloud**
1. Set up EC2 instance
```
aws ec2 create-security-group --group-name PublicSG --description "Bastion Host Security group" --vpc-id vpc-XXXXXXXX
aws ec2 authorize-security-group-ingress --group-id YOUR_PUBLIC_GROUP_ID --protocol tcp --port 22 --cidr 0.0.0.0/0
aws ec2 run-instances --image-id ami-0bcc094591f354be2 --instance-type t2.large --security-group-ids
YOUR_PUBLIC_GROUP_ID --associate-public-ip-address --key-name YOUR_KEY_NAME
ssh -A ubuntu@YOUR_PUBLIC_EC2_NAME.compute-1.amazonaws.com
```
2. Install Docker and log into Docker Hub
3. Expose mosquitto NodePort
```
aws ec2 authorize-security-group-ingress --group-id YOUR_PUBLIC_GROUP_ID --protocol tcp --port 1883 --cidr 0.0.0.0/0
```
4. Create user defined network
```
docker network create mqtt
```
5. Build additional Docker containers
```
docker build -t rubyhan/mosquitto:alpine_x86_remote -f Dockerfile.mosquitto_x86 .
docker push rubyhan/mosquitto:alpine_x86_remote
docker build -t rubyhan/processor-ec2:ubuntu -f Dockerfile.processor .
docker push rubyhan/processor-ec2:ubuntu
```
6. Create S3 bucket
```
rubyhan-w251-hw3
```
- Add IAM role using policy AmazonS3FullAccess to EC2 instance
```
S3MountBucket
```
7. Run containers
```
docker run -d rubyhan/mosquitto:alpine_x86_remote
docker run --network=host -d rubyhan/processor-ec2:ubuntu
```

## Face Image Example
- **S3 bucket:** 

https://rubyhan-w251-hw3.s3.ca-central-1.amazonaws.com/face_10.png
- **Sample Image:**

![image](./face_10.png)

## MQTT
MQTT is a Client Server publish/subscribe messaging transport protocol. It is light weight, open, simple, and designed so as to be easy to implement. These characteristics make it ideal for use in many situations, including constrained environments such as for communication in Machine to Machine (M2M) and Internet of Things (IoT) contexts where a small code footprint is required and/or network bandwidth is at a premium.

### MQTT Topics 
In MQTT, the word topic refers to an UTF-8 string that the broker uses to filter messages for each connected client. The topic `face_detection` was used in this assignment as it was self-explanatory in describing the face detecting sensors (USB camera and haarcascade_frontalface_default.xml). The topic is represented in both the publisher and subscriber codes.

### Quality of Service (QoS)
The Quality of Service (QoS) level is an agreement between the sender of a message and the receiver of a message that defines the guarantee of delivery for a specific message. There are 

At most once (0) 
- guarantees a best-effort delivery but no guarantee of delivery
- recipient does not acknowledge receipt of the message and the message is not stored and re-transmitted by the sender
- “fire and forget”

At least once (1)
- guarantees that a message is delivered at least one time to the receiver
- possible for a message to be sent or delivered multiple times

Exactly once (2).
- guarantees that each message is received only once by the intended recipients.
- safest and slowest service quality

QoS of 0 was used in this assignment as it is not necessary to publish every face image captured as reflected in the below code snippet. If security is an issue (i.e. identity theft), a higher level of QoS should probably be used.

```
local_mqttclient.publish(LOCAL_MQTT_TOPIC, msg, qos=0, retain=False)
```