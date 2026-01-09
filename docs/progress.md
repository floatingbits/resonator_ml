# Progress
## Initial setup
### Network, feedback and features
The basic idea was to approximate a real string with a refined Karplus-Strong-Algorithm, where the loop filter is 
replaced by a neural network. The first training data comes from a DI recording of the low E-string of my Stratocaster
guitar. The neural network is a 2x64 hidden layer tanh with a single output (next sample prediction) and a few integer
delays of the output. Since the main task of the resonator system should be: LP Filter the signal and delay it by the 
period duration, the first shot at the input features were the following delay structure:
- y_1, y_2, y_3 => To let the system design a kind of IIR Filter for the faster decay of high frequencies
- y_T+3,y_T+2,y_T+1, y_T, ...y_T-3 => To let the system swing in the correct frequency, find proper coefficients for 
the interpolation between samples and so on

Control Inputs were only present in the form of constant dummy data.

### Training data
The training data were taken from a "decay only" version of the original string sample. At first, I only took about 
5 sec from the start, leaving out no sample. 

### Loss function
I started with a simple MSE. 

### Training process
The training process is performed sample wise in batches. Due to a misunderstanding of the exact meaning of batch_size, 
I started off with a batch size of 20000, quickly reducing it to more practical values.

### Inference/sound generation
The resonator as such does not produce sound of its own without being initialized. For the time being, 
without yet having extracted exciter signals and functionality, we just initialize the delay buffers with the first 
samples of the original sound file.  

## First results
The very first result file was a bit misleading, because it only seemed to have a small issue that might be easy to fix:
It was a decaying string that stopped decaying at some point and rang on forever. My interpretation was as follows: 
The network is non-linear. So small signals can have a different decay than big ones. So far, so good. Big signals are 
apparently more accurately represented than small ones, and the small ones have a bigger amplification due to the 
tanh-function.
### First measures
Since big signals are more accurately represented, the loss function must be made relative. So I changed that to 
relative_l1. That didn't fix it, but nevertheless seemed a good idea, because the decay of the string is also relative.

The second thought was that more linearity is needed, so I tried to take out the tanh layers step by step, with 
different but no better results. 

In an attempt to gain some more knowledge about what was going on, I tried to throw more hidden neurons and thus 
processing power at the network, giving it more possibilities to reflect a broader range of linearity. Still no success. 

### More sophisticated thoughts
So there were quite a few measures taken, with no real improvement, although they all seemed to make sense in some way. 
Let's dive deeper into what went wrong. 

#### The different behaviors
How did the system behave exactly in the failed experiments. I could not only observe different types behaviors 
throughout the simulated string decay, but also find out, that the behavior does not fully depend on the systems parameters,
but is also quite random. So, I think the randomness introduced in training let the system show different behavior, but 
in the end, each model had all problems at once. Let me list the behaviors:
- Not decaying at some point (as described for the first result)
- Decaying, but decaying much too fast
- finally converging to a biased value, not 0

#### Questioning my mental model
My mental model was that a decaying string should be able to be modeled through a filtered feedback delay. 
In my imagination a good approximation would be to just take the appropriate delayed signal (some of the inputs or 
a mix of some sorts) and the results should be better than what I actually got. The nonlinearity of the hidden neurons
couldn't be the only culprit, since even removing them, the network didn't learn to approximate as imagined above. 
So, what was it that made it so difficult for the system to learn appropriate behavior?
The following ideas came to my mind: 
- We were trying to learn sample accurately from one sample to another. Tiny errors will have huge impacts due to the 
feedback nature of the system.
- Are the training samples of my real world strat really that clearly stating: Decay a tiny bit in every loop?
- Are the training data evenly spread across the WHOLE dynamic spectrum?
- Are they even symmetric?

#### ...and deriving measures
Armed with these questions a whole lot of more sophisticated measures can be taken:
- When we want the system to show a more statistically correct overall behavior, we must stop training it only 
sample-wise. Errors will show after 2,3,4...10 iterations much more clearly and more importantly: relevant for the 
outcome than just predicting the next sample. => Find a way to make training based on more long-term predictions, 
so the network gets the chance to learn the behavior more globally/statistically than per sample.     
- When having a look at the real world training data, I actually found out: No, the data do not state a clear decay 
per sample (even per loop/period). I guess it must have something to do with noise and/or the fact that a Strat's 
pickup transmits the energy depending on the plane it vibrates in. Since the plane can change, there is no 1:1 relationship 
between the energy in the string and the signal and the signal we see might get louder for a small while. To prevent this
I resumed to a Karplus string generated training signal for the time being. Also, different real recordings could be 
used to show a better sense for the statistical behavior. 
- Make the training use the whole input file until it is decayed. Not only 5 s.
- Force the training data to be symmetric by always adding a negated version of the samples.

## Further Progress and results
### Using Karplus Strong training sample
- Not the expected result at first
- Omitting nonlinear Hidden Layers, the general decay and LP behavior could be simulated, which was a giant step, 
regarding the poor first results. But there was a bias aggregating, so I assumed the model needs a protection 
against bias. Since there was no bias inherent in the model, I started with focusing on the training data. 
### Using symmetric input
- When forcing every training sample to contain its negative version (in terms of audio inputs), the model soon learned
to decay around the center line. Only just at the end of the result files, we are able to notice a small DC component. 
- I assume that at very small signal levels, training data is getting very scarce, so we need to address that.
### Using a broader range of training samples

So far the training samples consist of the first N samples of one decaying sample note, both in original and negated 
version.
Although it looks like many samples containing all the necessary information about the behavior of the original string, 
we have a systematic lack of information:
- We only have one training sample per phase and energy level. Noise might disturb the network.
- We cut the signal at some point to save training time, so no small-signal tail is present in training data.

Ideas:
- Using more different input files with different versions of the same tone will provide a more general idea
- Integrating smaller signals into the training data, randomly leaving out bigger ones to not further bloat the training
time

Results: 
To DO

### Thoughts about generalisation
When thinking about the training data, and the systems inability to handle small signals without generating biases, 
some thoughts came to my mind that didn't have to do too much with training data quality: It was the systems (in)ability to
generalize. At first, I had a very good approximation of a vibrating string in mind: Karplus-Strong - Delay, Decay, Filter.
This is a generalisation, that the system could have learned to solve the task with very little effort. But it didn't
and instead, in my eyes, performed much worse. 

Why? I guess for multiple reasons:
- I didn't really tell it, what mattered to me most (statistically correct decay), and taught it to be sample correct 
instead
- It might be a case of overfitting: While I want it to generalize and learn a smooth filter and decay in the first place,
the network tries to learn every detailed behavior at every dynamic stage, which it might even be capable of, considering
its amount of hidden layers. 

Considering the overfitting, the easiest solutions to try first are:

- (Already mentioned) Increasing the amount of training data files to enhance the variety of training samples
- Reducing the network complexity (until we want to improve realism and find our network is over-generalizing)
- More input features (adding delay lines like 1/4T ) would provide much broader input. E.g. before, without 1/4, input 
values near zero might be a zero cross or just a small value. Providing a 1/4T delay, the training samples will show a 
much bigger difference and thus provide valuable info to the model so that it can deduct meaning better instead of 
clamping to mere numbers and exact samples.     

If that doesn't work, other techniques generally recommended against overfitting can be tried, but we would first need to
find out about the specific nature of the system's behavior and decide then what makes most sense.






