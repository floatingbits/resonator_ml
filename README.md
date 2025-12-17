# Resonator ML

## Goals
The purpose of this repository is mainly training my machine learning skills. 
The goal of the code is to train a resonator, that mimicks a real life string in all its details.

### Inspiration
Inspired by a relatively simple algorithm, the Karplus-Strong-Algorithm, I want to see if it is possible to extend the 
base algorithm's idea of simulating a plucked string by feeding back filtered signal, that was delayed exactly by the 
string's base frequency's period duration. Although the real physical string example can be recognized in the 
algorithm's output, the results are still quite boring and naive, when compared to the complex sound of real 
physical instruments. I already found out during my diploma thesis that introducing non-linearities into the feedback
of the KS-algorithms loop, really complex sounds can be achieved.

### The idea
Back then, neural networks were still kind of science fiction. Even though I heard of these networks and their potential,
ML libraries like PyTorch were not available. Now, with this project, I want to find out to what extent it is possible 
to recreate complex behaviour as in real acoustic (-electric) instruments. What features of the real instrument will 
this ML resonator be able to learn? And, almost more importantly: What will I need to learn to achieve this task?
Just to name a few properties that the Karplus-Strong-System does NOT include, but what I hope can be learned with the 
right parameters and training data: 

- String stiffness resulting in harmonics with non-integer multiples of the base frequency
- Non-linear behaviour: Loud signals might decay at a faster rate (or show a different behaviour in another domain)
than smaller signals.
- Interaction with an instrument's body/complex reflections

Also, the network should be able to handle the parameters of the KS-algorithm automatically: 
- decay rate
- LP filter parameters for higher frequencies decaying faster
- Non-integer delay interpolation to get the right pitch

### What I want to learn about ML from this project
My experience with machine learning was quite limited when I started the project. Next to the questions of what can be 
achieved using this algorithm, there are a lot of questions still to be answered and more to be discovered: 

- How to organize training? 
- What is the influence of the loss function?
- How can I integrate my knowledge about the physical system (without 
compromising the main idea of letting the network do the work)
- What does the training data need to look like to get good results?
- What more sophisticated, previously unknown (to me) techniques are there to train a network?

### Other ML techniques
I am aware that Machine Learning does not only consist of neural networks and that even neural networks could have been
integrated in a lot of other different ways. Maybe I will find the time to setup other architectures for the same goal to 
find out if the main idea has an advantage over other techniques and more realism can be achieved. 

## Results
### The journey is the destination
Since the project is about learning (concerning my skills), not only the final results should play a role, but much more 
importantly, my progress. Have a look at [progress.md](docs/progress.md) and [versions.md](docs/versions.md) for a 
detailed documentation of the steps, the thoughts, the takeaways and also the final results.  