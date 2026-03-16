import cupy as cp
import numpy as np


from scipy.ndimage import correlate

DEFAULT_GAIN = 1.5
DEFAULT_THRESHOLD = 0.5

DEFAULT_WEIGHT_BASELINE = 0.1


# Tsodysk Makram parameters

DEFAULT_UTIL_BASE = 0.5


DEFAULT_TAU_A = 20

# Time decay for a availabilty 
DEFAULT_TAU_A_AV = 50

# Time decay for a utilizacion
DEFAULT_TAU_A_UTIL = 20



DEFAULT_TAU_TM_A = 20


DEFAULT_TAU_TM_A_AV = 50


DEFAULT_TAU_TM_A_UTIL = 20


# The default value for synaptic tagging, used to ensure learning in networks without emotional weighting

DEFAULT_SYNAPTIC_TAG = 0.2



class TMLayer:


    def sigmoid(self, z):
        z = np.clip(z, -100, 100)
        return np.exp(self.gain * (z-self.threshold)) / (1 + np.exp(self.gain * (z-self.threshold)))

    def inverse_sigmoid(self, z):
        z = np.clip(z, 1e-15, 1 - 1e-15)
        return -1/self.gain * np.log(1/z - 1) + self.threshold
    



    def __init__(self, dims=(32, 32, 8), temporal_axis=(2,), config={"distal_radius": 4, "proximal_radius": 4, "permanence_sigma": 0.5, "initial_permanence_mean": 0.1, "weight_baseline": DEFAULT_WEIGHT_BASELINE}):    
        self.dims = ()
        self.temporal_axis = ()



        self.dt = 1




        self.rng = np.random.default_rng()

        self.distal_radius = config.get("distal_radius", 8)
        self.proximal_radius = config.get("proximal_radius", 8)
        self.permanence_sigma = config.get("permanence_sigma", 0.5)
        self.initial_permanence_mean = config.get("initial_permanence_mean", 0.1)


        self.util_base = DEFAULT_UTIL_BASE





        # Maximum capacity of permanence

        self.p_capacity = 1.0 
        self.tm_p_capacity = 1.0

        '''
        Precision handles how much we trust feedforward/proximal inputs over distal connections. Effectively, how much
        do we listen to sensory data vs how much do we listen to internal thoughts.

        Gain describes how picky we want the neurons to be- if they will only weight the strongest synapses,
        or if they will integrate in a broader sense. Think the variance of relative weighting of the synapses.
        '''
        self.precision = 0.5

        self.gain = DEFAULT_GAIN

        self.threshold = DEFAULT_THRESHOLD

        self.dims = dims

        # axis for temporal coding (i.e. columns)
        self.temporal_axis = temporal_axis


        self.temporal_depth = int(dims[temporal_axis[0]])

        # Get the dimensions orthagonal to the temporal dimension
        self.spatial_dims = self.dims[0:self.temporal_axis[0]] + self.dims[self.temporal_axis[0]+1:]


        print(f"Spatial dims: {self.spatial_dims}")
        self.z = np.zeros(self.dims)



        # Permanences

        # The "normal" permanences here handle distal connections. This contrasts with the proximal/feedforward connections of below

        self.p = np.array(self.dims + (2*self.distal_radius+1, 2*self.distal_radius+1))
        self.p = self.rng.normal(self.initial_permanence_mean, self.permanence_sigma, self.dims + (2*self.proximal_radius+1, 2*self.proximal_radius+1))

        # Tensor for weights
        self.w = np.zeros(self.dims + (2*self.distal_radius+1, 2*self.distal_radius+1))

        # Set the weights to be some portion of the baseline
        
        self.w = self.p * DEFAULT_WEIGHT_BASELINE 



        # Distal activations, i.e. all cells in columns.
        self.a = np.zeros(self.dims)



        # Analogous to HTM SP layer. Treats temporal axis as if cells along that axis are ONE cell.
        self.tm_z = np.array([])
        

        
        # Permanence, analogous to HTM SP permanences


        self.tm_p = self.rng.normal(self.initial_permanence_mean, self.permanence_sigma, self.spatial_dims + (2*self.proximal_radius+1, 2*self.proximal_radius+1))


        # Analogous to HTM SP weights. 
        

        self.tm_w = self.tm_p * DEFAULT_WEIGHT_BASELINE

        print(np.shape(self.tm_w))


        # Proximal activations
        self.tm_a = np.zeros(self.spatial_dims)



        #
        #
        # Tdosysk Makram Math

        # Axon availability (distal)
        self.a_av = np.ones(self.a.shape)

        # Axon utilization (distal)

        self.a_util = np.ones(self.a.shape) * DEFAULT_UTIL_BASE



        self.tau_a = DEFAULT_TAU_A



        # Set the time decay constant for a availability
        self.tau_a_av = DEFAULT_TAU_A_AV



        #
        self.tau_a_util = DEFAULT_TAU_A_UTIL







        # Weight availability (proximal)

        self.tm_a_av = np.ones(self.tm_a.shape)


        # Weight util (proximal)


        self.tm_a_util = np.ones(self.tm_a.shape) * DEFAULT_UTIL_BASE



        self.tau_tm_a = DEFAULT_TAU_TM_A



        self.tau_tm_a_av = DEFAULT_TAU_TM_A_AV



        self.tau_tm_a_util = DEFAULT_TAU_TM_A_UTIL


        # Relative frequency of the neurons 

        self.frequency = np.array(self.dims)

        # For the freqeuncy along the collapsed time dim.
        self.tm_frequency = np.array(self.spatial_dims)



        self.base_synaptic_tag = DEFAULT_SYNAPTIC_TAG        



        # Synaptic tags

        self.a_tag = np.zeros(self.dims)


        self.tm_a_tag = np.zeros(self.spatial_dims)



        # Arrays to store the permanence weighted tags.

        self.p_tag = np.array([])


        self.tm_p_tag = np.array([])



        # Store the permanence util
        self.a_permanence_util = np.zeros(self.a.shape)



        # Set individual permanence capacities to one

        self.a_permanence_capacity = np.ones(self.a.shape) * self.p_capacity







        # Store the permanence utilization
        self.tm_a_permanence_util = np.zeros(self.tm_a.shape)


        # Set the individual permanence capacities to one 
        self.tm_a_permanence_capacity = np.ones(self.tm_a.shape) * self.tm_p_capacity

        
        # Set the permanence utilization for distal connections. Using the tm_a because we need to use the singular cell.


        #### !!!! PRETTY SURE HTIS WRONG
        self.tm_a_distal_permanence_util = np.sum(self.p, axis=(self.temporal_axis[0], -2, -1))

        print(self.tm_a_distal_permanence_util.shape)

        #print(self.tm_a.shape)
        # Use the temporal axis to get the total capacity we want.
        self.tm_a_distal_permanence_capacity = np.ones(self.tm_a.shape) * self.p_capacity * self.dims[self.temporal_axis[0]]


        print(self.tm_a_distal_permanence_capacity.shape)


        '''TO-DO: Init the weight tensor using a (normal?) distribution.'''


    def update(self, input: np.ndarray):
        if input.shape != self.spatial_dims:
            raise ValueError(f"Input shape {input.shape} incompatable with spatial dimensions {self.spatial_dims}")
        

        
        
        self.tm_z = self.inverse_sigmoid(self.tm_a)


        # Use kernels to map the input to our weighted synaptic values


        tm_a_padded = np.pad(input, self.proximal_radius, mode='constant')
        tm_a_windows = np.lib.stride_tricks.sliding_window_view(tm_a_padded, (2*self.proximal_radius+1, 2*self.proximal_radius+1))


        delta_tm_z = np.einsum("ijkl,ijkl->ij", tm_a_windows, self.tm_w) * self.tm_a_av * self.tm_a_util

        self.tm_z[np.where(self.tm_a < DEFAULT_THRESHOLD)] = 0


        # Tsodysk Makram model


        # Update the temporal activations
        self.tm_z = self.tm_z - (self.tm_z/self.tau_tm_a - delta_tm_z) * self.dt


        # Update temporal "neurotransmitter" availability
        self.tm_a_av += ((1 - self.tm_a_av)/self.tau_tm_a_av - self.util_base * (1 - self.tm_a_util) * self.tm_a) * self.dt


        # Update utilization
        self.tm_a_util += ((self.util_base - self.tm_a_util)/self.tau_tm_a_util + self.util_base * (1 - self.tm_a_util) * self.tm_a) * self.dt
        


        # Clip values to prevent overflow


        self.tm_z = np.clip(self.tm_z, -10, 10)

        self.tm_a_av = np.clip(self.tm_a_av, -10, 10)

        self.tm_a_util = np.clip(self.tm_a_util, -10, 10)


        self.tm_a = self.sigmoid(self.tm_z)



        print(self.tm_a)




        ##### Now, we must weight the tags relative to their combined strenth for a particular neuron in a, and use exponential weighting to determine
        ##### Which permanence values we want to update and by how much.

        

        self.a_tag = self.base_synaptic_tag * np.ones(self.a_tag.shape)


        a_tag_padded = np.pad(self.a_tag, [(self.distal_radius, self.distal_radius), (self.distal_radius, self.distal_radius), (0, 0)], mode='constant')

        weighted_a_tag_windows = np.lib.stride_tricks.sliding_window_view(a_tag_padded, (2*self.distal_radius+1, 2*self.distal_radius+1), axis=(0,1))


        print("Weighted a tag, self.p shapes: ")
        print(np.shape(weighted_a_tag_windows))
        print(np.shape(self.p))
        self.p_tag = np.einsum("ijmkl,ijmkl->ijm", weighted_a_tag_windows, self.p)


        


        # Weighting based on total amount of connections relative to the entire column. Not just counting active permanences

        total_capacity_weighting = np.exp((self.a_permanence_capacity - self.a_permanence_util))/np.exp((self.tm_a_distal_permanence_capacity - self.tm_a_distal_permanence_util)[..., np.newaxis])


        

        # Weighting based on the relative strength of currently existing connections to the neurons whose tag is in question relative to the rest of the 
        # connections of that cell.

        active_capacity_weighting = np.exp(self.p_tag) / np.exp((self.a_permanence_util))

        print("active capacity weighting shape: ")
        print(active_capacity_weighting.shape)

        print("total capacity weighting shape: ")
        print(total_capacity_weighting.shape)
        #
        #
        #
        # 
        ''' INtroduce term to maintain sparsity of connections by weighting TOTAL (not just column) increase?'''

        permanence_update_weighting = total_capacity_weighting * active_capacity_weighting

        permanence_update_weighting /= np.sum(permanence_update_weighting)

        print("Permanence update weighting shape")

        print(permanence_update_weighting.shape)


        p_raw = self.inverse_sigmoid(self.p) + permanence_update_weighting[..., np.newaxis, np.newaxis]/(np.sum(self.p) + np.sum(permanence_update_weighting))
        
        self.p = self.sigmoid(p_raw)
        #print(self.p)


        





        



        


#nn = TMLayer()


