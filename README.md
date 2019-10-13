# End-to-End-Self-Driving-Simulation
With Udacity's open source simulator available, I tried to implement Nvidia's End to End Self Driving research paper. The implementation of the model is the same as that given in the paper. 

1 normalization layer
5 conv layers
3 fully connected layers 

video link -- https://www.youtube.com/watch?v=Tp0A63jrhsg
Further steps
1) Training the network for predicting steering and throttle together. Currently, I have 2 different models for steering and throttle prediction.
2) Implement conditional Imitation Learning on CARLA
3) Try using RNN on the extracted features and compute the steering values
