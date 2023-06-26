#we now need to dive deeper into LSTM and their structure 
"""
Long short term Memory or LSTM -- 
what my understanding is ---->
                                1.recurrent neural network
                                2.trained using back propogation 
                                3.solves the problme of gradient vanishing 
                                4.we use neurons traditionally but here we use what we call a memory block in layer 
type of a recurrent neural network with some more variables in them 

Gates used in a LSTM --->
forget 
state and 
I/O gates 

"""
import tensor2tensor as t2t
t2t.__loader__
