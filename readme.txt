###if permission issue on gpu or hopper write --user in the end
pip3 install flwr
pip3 install scikit-learn  
pip3 install torch torchvision
pip install ultralytics
pip install matplotlib

###to run write following command on terminal
## --model_type: cnn , resnet, yolo
## --data_type: mnist, cifar, video
##  --p_type: local, global, quant

#To run server: 
python3 server.py --model_type cnn --data_type mnist --p_type global

#To run client1:
python3 client1.py --model_type cnn --data_type mnist --p_type global

#To run client2:
python3 client2.py --model_type cnn --data_type mnist --p_type global
