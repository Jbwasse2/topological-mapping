# Topological Mapping on Real World Robot

# Architecture
At building time
trajectory collection -> similarity detector -> sparse trajectory collection -> similarity detector -> topological map

At run time
current view -> nav policy -> action -> if reached local goal, update goal -> repeat until at final goal

## Similarity Detector
Given two images, what is the distance between the two images. Another metric that is often used is ?what is the probability the two images are close?"
Working Implementations:  
TODO:  
Meng et al  
Mine trained on real world data  

## Image - Nav Policy
Given the current view and a (local) goal image, predict an action in order to reach the goal  
Working Implementations:  
Meng et al  
TODO:  
