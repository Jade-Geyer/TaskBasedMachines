# TaskBasedMachines
general purpose task based mls and nns


=
|| Regression LSTM  ||
=
 
Purpose: periodic timeseries regression tasks


overview:
 
This is a single layer LSTM that is setup to take dataframes of any shape and process them as 3D arrays to 
train and test with. The existing sample implementation, after installation of dependencies, accesses
its training data through the DataBroker. It should be able to run out-of-the-box. This implementation generates 
quasi-random data that is unpredictable, thus serves as a baseline for "effectively 100% ineffective", although 
not statistically.
 
Once running, it uses a tuner to tune it's own hyperparameters by running trails on the given training data. Once
completed, it will output statistics as to it's accuracy. If given relational data in the appropriate format,
the LSTM should self-tune and make a test prediction on your set.

Note: large datasets may take an a long time. Start small and work your way up.

=========================
|| Relevant Parameters: ||
=========================
Timesteps: Default 15.

Timesteps are relevant to your datastructure. Current implementation assumes the data is seperated 
(although not necessearily demarked) into sets of 15. in essence, turning 15 timeseries increments 
into a 3d array representing a time-period. 

eg: a shape (1500, 14) is converted to (15, 100, 14). 100 15-minute periods. Predictions require a period as input.
 
Epochs: Default 3.
 
Defines how many epochs your tuner will run in each trial. more epochs take a longer time, but may be necessecary on
some models that need a tighter fit.

max_trials: Default 5.

Defines the number of times the tuner will run. Each will run X epochs, according to your settings for that variable.
