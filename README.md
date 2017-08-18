# Project: XOR Learning Neural Network
# Author: Ricky-N

I used Anaconda for this which includes the numpy library and can be downloaded from a quick search  
The code is a nearly direct translation into python of the code from this book:

### Code Your Own Neural Network: A step-by-step explanation  
### Stephen C. Shaffer  
### Kindle Edition  
### Sold by Amazon Digital Services, Inc  

Though it is quite short, it is also quite cheap and in my opinion worth a pick up.
Some liberties have been taken to clean up the code and threshold / weight matrices a little.
Note that no real effort was made on the part of efficiency in either processing or memory use.  

Basically:  
  The network class is a good generalization for a neural net with a single arbitrarily
  large layer of hidden nodes connecting an arbitrary number of input and output nodes.
  These can be set among the other parameters at the top.  

  The sampleMaker is tightly coupled with the network class and changes the input values
  and expected output values each iteration to the next xor input set. It is a decent
  design to separate this functionality out as it is very function specific and should
  not be included in the network directly, though one could argue it would be better to
  have the network be instantiated with a sampleMaker instead of the other way around.  

  An instance of each class is created and the main loop of the program steps through:  
    1. set the next input and expected output  
    2. evaluate the hidden and output layers of the network based on input  
    3. calculate current error and update thresholds and weights  
