import json
import numpy as np
import os
os.environ['MPLCONFIGDIR'] = 'tmp/matplotlib'
os.environ['TMPDIR'] = 'tmp'

# Ensure the directories exist
os.makedirs('tmp/matplotlib', exist_ok=True)
os.makedirs('tmp', exist_ok=True)
import datetime
import matplotlib
#matplotlib.use('Tkagg')
import matplotlib.pyplot as plt
import time
import pandas as pd
from tensorflow.python.client import device_lib
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention
import astropy.io.fits as pyfits
import pickle
import sys
from itertools import chain
from multiprocessing import Pool
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Dense, Input, Embedding, GlobalAveragePooling1D
import multiprocessing
import time
start_time = time.time()
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from keras.layers import Input, Dense, Conv1D, BatchNormalization, Flatten, Dropout
from tensorflow.keras.layers import Concatenate, Lambda
from tensorflow.keras.layers import Layer
from tensorflow.keras.initializers import Constant, Zeros
import psutil
#from memory_profiler import profile
import gc
import shutil
import math
import json
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
#log directory for tensorboard
logdir = "tf_logs" 



#Prime  physics information for model training
global len_data
global max_z
global wavelength_template
global means
global columns

#maximum expected redshift
max_z = 1.0

#number of wavelength bins
len_data = 4629

#maximum observed wavelength
x_max = np.log10(11000)

#resolution of the wavelength bins (in log10 angstroms)
resolution = 9.9897385e-05

#number of pixels in the wavelength template
num_pixels = len_data + math.ceil((np.log10(max_z+1))/resolution)

#generate the model's wavelength template
wavelength_template = np.arange(x_max - resolution*num_pixels, x_max, resolution)

#expected emission/absorption lines
means = np.log10([
                        1033.82, 1215.24, 1240.81, 1305.53, 1335.31,
                        1397.61, 1399.8, 1549.48, 1640.4, 1665.85,
                        1857.4, 1908.734, 2326.0, 2439.5, 2799.117,
                        3346.79, 3426.85, 3727.092, 3729.875, 3889.0,
                        4072.3, 4102.89, 4341.68, 4364.436, 4862.68,
                        4932.603, 4960.295, 5008.240, 6302.046, 6365.536,
                        6529.03, 6549.86, 6564.61, 6585.27, 6718.29,
                        6732.67, 3934.777, 3969.588, 4305.61, 5176.7,
                        5895.6, 8500.36, 8544.44, 8664.52
                        ])

columns = ["O VI", "Lyα", "N V", "O I", "C II", "Si IV", "Si IV + O IV", "C IV", "He II", "[O III]", "Al III", "C III", 
                        "C II", "Ne IV", "Mg II", "[Ne V]", "Ne VI", "[O II]", "[O II]", "He I", "[S II]", "Hδ", "Hγ", "[O III]", "Hβ", 
                        "[O III]", "[O III]", "[O III]", "O I", "O I", "N I", "[N II]", "Hα", "[N II]", "[S II]", "[S II]", "K", "H", "G", 
                        "Mg", "Na", "CaII", "CaII", "CaII", "Z"] #HB is idx 24  Ha is 32    



#set multithreading limits and gpu device
tf.config.threading.set_inter_op_parallelism_threads(6)  # For coordinating independent operations
tf.config.threading.set_intra_op_parallelism_threads(6)  # For speeding up individual operations
gpus = tf.config.experimental.list_physical_devices('GPU')



#set gpu memory growth
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)



#not in use yet but can be used to potentially distribute training onto multiple gpus
strategy = tf.distribute.MirroredStrategy()



def adjust_column_names(names):
    """
    Adjusts element name strings by appending a counter to duplicates.

    Parameters:
    names (list of str): The list of column names.  

    Returns:
    list of str: The adjusted column names.        
    """

    counts = {}
    new_names = []
    for name in names:
        if name in counts:
            counts[name] += 1
            new_name = f"{name}.{counts[name]}"
        else:
            counts[name] = 0
            new_name = name
        new_names.append(new_name)
    return new_names


#Simulates starburst galaxy spectra
class data_producer():
    def __init__(self, data_points, min_amplitude, max_amplitude, min_x_value, max_x_value):
        
        #Number of spectra to simulate
        self.data_points = data_points

        #emission/absorption line centers
        self.means = means
        
        #emission/absorption line names
        self.columns = columns
        
        #simulated spectra storage
        self.gaussians_batch = []

        #simulated spectra with noise storage
        self.noisy_gaussians_batch = []

        #redshift converted to pixel shift storage
        self.lambdas = []

        #line strengths storage
        self.line_strengths = []

        #dataframe storage
        self.dataframe = []

        #widths storage
        self.widths = []

        #noise storage
        self.noise_spectra = []

        #signal to noise ratio storage
        self.snrs = []

        #wavelength template storage
        self.wavelength_template = []

    def initialize_data(self, wavelength_template):
        """
        Initializes the data for the simulation.

        Parameters:
        wavelength_template (np.ndarray): The wavelength template for the simulation.

        Returns:
        np.ndarray: The simulated spectra.
        np.ndarray: The redshifts converted to pixel shifts.
        pd.DataFrame: The DataFrame containing the simulated data.
        np.ndarray: The widths of the Gaussian distributions.
        """

        self.wavelength_template = wavelength_template
        means = tf.constant(self.means, dtype = tf.float32)

        pre_compiled_data = np.zeros((self.data_points, len(self.means)+1))
        
        
        #initialize ratios
        Ha_Hb = 2.86

        Nii_Ha_max = 10
        Nii_Ha_min = 10**-4

        Oiii_Hb_max = 10
        Oiii_Hb_min = 10**-2.5

        Sii_Ha_max = 10
        Sii_Ha_min = 10**-2.5

        Nii_Oii_max = np.sqrt(10)
        Nii_Oii_min = 10**-1

        Oii_Oiii_max = 100
        Oii_Oiii_min = 10**-1

        
        #create line ratios using range of random values
        for i in range(len(pre_compiled_data)):
            #set line ratios
            Nii_Ha = np.random.uniform(Nii_Ha_min, Nii_Ha_max)
            Oiii_Hb = np.random.uniform(Oiii_Hb_min, Oiii_Hb_max)
            Sii_Ha = np.random.uniform(Sii_Ha_min, Sii_Ha_max)
            Nii_Oii = np.random.uniform(Nii_Oii_min, Nii_Oii_max)
            Oii_Oiii = np.random.uniform(Oii_Oiii_min, Oii_Oiii_max)

            #set subcontext line strengths
            Nii_subcontext = Ha_Hb*Nii_Ha/4
            Oiii_subcontext = Oiii_Hb
            Sii_coefficient = np.random.uniform(0.2, 2.0)
            Sii_subcontext = Ha_Hb*Sii_Ha/(Sii_coefficient+1)
            Sii_0 = Sii_subcontext*Sii_coefficient
            Sii_1 = Sii_subcontext

            #set line strengths
            pre_compiled_data[i][24] = 1
            pre_compiled_data[i][32] = Ha_Hb
            pre_compiled_data[i][31] = Nii_subcontext*3
            pre_compiled_data[i][33] = Nii_subcontext
            pre_compiled_data[i][27] = Oiii_subcontext
            pre_compiled_data[i][34] = Sii_0
            pre_compiled_data[i][35] = Sii_1
            pre_compiled_data[i][18] = (Nii_subcontext*3+Nii_subcontext)/Nii_Oii
            
            #set redshift
            pre_compiled_data[i][-1] = i / pre_compiled_data.shape[0]

        #convert redshift to pixel shift
        self.lambdas = pre_compiled_data[:, -1]*redshift_to_shift(0, wavelength_template)
        
        #create tensorflow dataset of line strengths and redshifts
        dataset = tf.data.Dataset.from_tensor_slices(pre_compiled_data).cache()
        dataset = dataset.batch(1024).prefetch(buffer_size=tf.data.AUTOTUNE)

        #collect templates and widths for each spectrum
        gaussians_batch = []
        widths = []

        #create templates
        for step, batch in enumerate(dataset):

            #cast values to tensorflow float32
            tmp_wavelength_template = tf.cast(wavelength_template, dtype=tf.float32)
            batch = tf.cast(batch, dtype=tf.float32)

            #choose random widths for each line
            widths_tmp = np.random.uniform(1.38, 20, size = len(batch))

            #compute template
            gaussians_batch_tmp = compute_batch_gaussians_tf(tmp_wavelength_template, batch[:, :-1], widths_tmp)
            
            #slice template by redshift
            gaussians_batch_tmp = slice_2d_tensor_by_1d_indices(gaussians_batch_tmp, batch[:, -1]*redshift_to_shift(0, wavelength_template))

            #append templates and widths to storage
            for i, j in zip(gaussians_batch_tmp.numpy(), widths_tmp):
                gaussians_batch.append(i)
                widths.append(j)

            #print progress
            sys.stdout.flush()
        
        #convert templates to numpy array
        gaussians_batch = np.array(gaussians_batch)

        #store simulated spectra
        self.gaussians_batch = gaussians_batch

        #line names
        adjusted_columns = adjust_column_names(self.columns)

        #create dataframe
        df = pd.DataFrame(pre_compiled_data, columns=adjusted_columns)

        #store line strengths, dataframe, and widths
        self.line_strengths = pre_compiled_data[:, :-1]
        self.dataframe = df
        self.widths = np.array(widths).flatten()


        return gaussians_batch, self.lambdas, df, np.array(widths).flatten()
    
    def noise_injector(self, dust_attenuation = True, dust_no_noise = False, plot_dust_curve = True):
        """
        Injects noise into the simulated spectra.

        Parameters:
        dust_attenuation (bool): Whether to apply dust attenuation.
        dust_no_noise (bool): Whether to apply dust attenuation without adding noise.
        plot_dust_curve (bool): Whether to plot the dust attenuation curve.
        
        Returns:
        np.ndarray: The noisy spectra.
        """

        #initialize noisy spectrum storage
        noisy_spectra = np.copy(self.gaussians_batch)
        num_spectra = noisy_spectra.shape[0]

        #initialize EBVs
        EBVs = np.random.uniform(0, 1.2345, size=(num_spectra))

        #convert EBVs to A_vs
        A_vs = EBVs*4.05

        #add dust to object spectra
        for i in range(num_spectra):
            if dust_attenuation == True:
                tau = calzetti_law(10**wavelength_template, EBVs[i])
                tau_z = slice_2d_tensor_by_1d_indices(np.array([tau], dtype=np.float32), np.array([self.dataframe['Z'][i]*redshift_to_shift(0, wavelength_template)], dtype=np.float32))[0]
                noisy_spectra[i] = noisy_spectra[i] / tau_z
                print(self.dataframe['Z'][i]*redshift_to_shift(0, wavelength_template))
                if plot_dust_curve == True:
                    if i == 0:
                        plt.figure('calzetti et al 2000 curve')
                        plt.plot(10**wavelength_template, tau, label = f'$A_{{v}}$ = {A_vs[0]:.2f}')
                        plt.legend()
                        plt.xlabel('Wavelength [$\AA$]')
                        plt.ylabel('$F_{e}/F_{ob}$')
                        



            #calculate snr and inject noise
            width = (self.widths[i])
            amps = self.dataframe.iloc[i]#noisy_spectra[i]
            signal = sum_gaussian_areas(amps, width, EBVs[i])
            noise = np.random.normal(loc=0.0, scale=0.0007, size=noisy_spectra.shape[1])
            self.noise_spectra.append(noise)
            snr = signal/((width*np.sqrt(np.sum(noise**2)/(10**self.wavelength_template[-1]-10**self.wavelength_template[0]))))
            
            #store snr
            self.snrs.append(snr)

            #inject noise
            if dust_no_noise == False:
                noisy_spectra[i] += noise
            
        return noisy_spectra


def calzetti_law(wavelength, EBV, R_V=4.05):
    """
    Calculate the Calzetti law attenuation factor k(lambda) for a given wavelength.
    
    Parameters:
    - wavelength: Wavelength in Angstroms
    - R_V: Total-to-selective extinction ratio (default is 4.05)
    
    Returns:
    - k_lambda: Attenuation factor at the given wavelength
    """
    
    k_lam = np.zeros(len(wavelength))
    lambda_um = wavelength / 10000.0  # Convert to micrometers
    closest_idx = np.abs(lambda_um-0.63).argmin()

    k_lam[closest_idx:] = (2.659 * (-1.857 + 1.040 / lambda_um[closest_idx:]) + R_V)
    k_lam[:closest_idx] = (2.659 * (-2.156 + (1.509 / lambda_um[:closest_idx]) - (0.198 / lambda_um[:closest_idx]**2) + (0.011 / lambda_um[:closest_idx]**3)) + R_V)
    


    ESBV = EBV
    
    tau = 10 ** (0.4 * ESBV * k_lam)


    return tau

def calzetti_law_single(wavelength, EBV, R_V=4.05):
    """
    Calculate the Calzetti law attenuation factor k(lambda) for a given wavelength.
    
    Parameters:
    - wavelength: Wavelength in Angstroms
    - R_V: Total-to-selective extinction ratio (default is 4.05)
    
    Returns:
    - k_lambda: Attenuation factor at the given wavelength
    """
    
    k_lam = 0
    lambda_um = wavelength / 10000.0  # Convert to micrometers

    if lambda_um >= 0.63:
        k_lam = (2.659 * (-1.857 + 1.040 / lambda_um) + R_V)    
    else:    
        k_lam = (2.659 * (-2.156 + (1.509 / lambda_um) - (0.198 / lambda_um**2) + (0.011 / lambda_um**3)) + R_V)    
    


    ESBV = EBV
    
    tau = 10 ** (0.4 * ESBV * k_lam)


    return tau



def sum_gaussian_areas(amplitudes, sigma, EBV):
    """
    Calculate the sum of the areas under multiple Gaussian curves with the same width.

    Parameters:
    amplitudes (list of float): The amplitudes of the Gaussian functions.
    sigma (float): The standard deviation (width) of each Gaussian.

    Returns:
    float: The total area under all the Gaussian curves.
    """

    #initialize means
    lines = tf.constant(10**means, dtype=tf.float32)

    #Compute sigma*sqrt(2*pi) once since it's common for all
    sqrt_2pi_sigma = np.sqrt(2 * np.pi*sigma)  

    #use second highest amplitude
    amplitude = np.sort(amplitudes.values[:-1])[-2]

    #get index of second highest amplitude
    index = np.argsort(amplitudes.values)[-2]

    #calculate total area
    attenuated_signal = amplitude/calzetti_law_single(lines[index], EBV)
    total_area = 0.68*attenuated_signal * sqrt_2pi_sigma


    return total_area




def inverted_relu(x):
    return -tf.nn.relu(x)  # Negate the output of the standard ReLU




class ScaledSigmoid(Layer):
    """
    Custom Keras Layer that applies a scaled sigmoid activation function.
    The output is scaled between `min_val` and `max_val`, with an optional steepness parameter 
    to control the sharpness of the sigmoid curve.

    Parameters:
        min_val (float): The minimum value of the scaled output range.
        max_val (float): The maximum value of the scaled output range.
        steepness (float): A scaling factor to control the steepness of the sigmoid curve. Default is 0.1.
    """

    def __init__(self, min_val, max_val, steepness=0.1, **kwargs):
        """
        Initializes the ScaledSigmoid layer with the specified scaling parameters.

        Args:
            min_val (float): Minimum value of the scaled output.
            max_val (float): Maximum value of the scaled output.
            steepness (float): Factor to control sigmoid curve steepness. Default is 0.1.
            **kwargs: Additional arguments for the Layer base class.
        """

        super(ScaledSigmoid, self).__init__(**kwargs)
        self.min_val = min_val  # Store the minimum value of the output range
        self.max_val = max_val  # Store the maximum value of the output range
        self.steepness = steepness  # Store the steepness parameter for the sigmoid function

    def call(self, inputs, **kwargs):
        """
        Applies the scaled sigmoid transformation to the input tensor.

        Args:
            inputs (Tensor): The input tensor to which the scaled sigmoid is applied.
            **kwargs: Additional arguments for the call method.

        Returns:
            Tensor: The transformed tensor where each value is scaled between `min_val` and `max_val`.
        """

        # Scale inputs by the steepness factor before applying the sigmoid function
        sigmoid = tf.nn.sigmoid(inputs * self.steepness)
        
        # Scale the output of the sigmoid function to the range [min_val, max_val]
        return self.min_val + (self.max_val - self.min_val) * sigmoid

    def get_config(self):
        """
        Returns the configuration of the layer for serialization.

        Returns:
            dict: A dictionary containing the layer's configuration parameters.
        """

        config = super(ScaledSigmoid, self).get_config()  # Get the base class config
        config.update({
            'min_val': self.min_val,  # Add min_val to the configuration
            'max_val': self.max_val,  # Add max_val to the configuration
            'steepness': self.steepness  # Add steepness to the configuration
        })
        return config
    
    
class GlobalSumPooling1D(Layer):
    """
    Custom Keras Layer that performs global sum pooling along the temporal dimension (axis 1) 
    of a 3D input tensor. This operation computes the sum of all values along the specified axis, 
    resulting in a 2D tensor as output.

    Use Case:
        - Useful for aggregating features in 1D data such as time-series or sequences, 
          where the overall sum of features along the sequence is meaningful.

    """
    
    def __init__(self):
        """
        Initializes the GlobalSumPooling1D layer.
        """

        super(GlobalSumPooling1D, self).__init__()  # Call the initializer of the base Layer class

    def call(self, inputs):
        """
        Applies global sum pooling to the input tensor along the temporal dimension (axis 1).

        Args:
            inputs (Tensor): A 3D tensor of shape (batch_size, timesteps, features),
                             where timesteps is the size of the temporal dimension.

        Returns:
            Tensor: A 2D tensor of shape (batch_size, features), where the temporal dimension
                    has been reduced by summing all values.
        """

        # Sum the values along axis 1 (temporal dimension)
        return tf.reduce_sum(inputs, axis=1)

#Melchior function
def SpectrumEncoder(input_shape=(len_data, 1), n_latent=10, n_hidden=[128, 64, 32], dropout=0.5):
    """

    Parameters:
    input_shape (tuple): The shape of the input data.
    n_latent (int): The number of latent features to learn.
    n_hidden (list): A list of integers specifying the number of hidden units in each layer.
    dropout (float): The dropout rate to apply after each hidden layer.

    Returns:
    Model: A Keras Model representing the SpectrumEncoder.
    """

    # Input layer
    input_layer = layers.Input(shape=input_shape)

    ##Convolutional blocks
    # First block
    x = layers.Conv1D(64, kernel_size=5, padding='same', activation='relu')(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = Dropout(0.2)(x)

    # Second block
    x = layers.Conv1D(128, kernel_size=11, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = Dropout(0.2)(x)

    # Third block
    y = layers.Conv1D(49, kernel_size=64, padding='same', activation='relu')(x)
    y = Dropout(0.2)(y)

    z = layers.Conv1D(128, kernel_size=64, padding='same', activation='relu')(x)
    z = Dropout(0.2)(z)

    #Multi-head attention mechanism
    x = MultiHeadAttention(num_heads=4, key_dim=32 // 4)(y, z)
    x = Dropout(0.1)(x)

    #Flatten the output
    x = GlobalSumPooling1D()(x)
  

    # Different specialized outputs

    #velocity dispersion
    sigmoid_part = Dense(1, activation='linear')(x)#, bias_initializer=tf.keras.initializers.Constant(np.log10(1.01)))(x)  # Use linear here because ScaledSigmoid applies the sigmoid
    sigmoid_part = ScaledSigmoid(min_val=1.38, max_val=20)(sigmoid_part)

    #emission and absorption line strengths
    both0 = Dense(3, activation='linear')(x)#, bias_initializer=tf.keras.initializers.Constant(1.0)
    emission0 = Dense(4, activation='relu')(x)
    both1 = Dense(2, activation='linear')(x)
    emission1 = Dense(1, activation='relu')(x)
    both2 = Dense(1, activation='linear')(x)
    emission2 = Dense(2, activation='relu')(x)
    both3 = Dense(4, activation='linear')(x)
    emission3 = Dense(2, activation='relu')(x)
    both4 = Dense(1, activation='linear')(x)
    emission4 = Dense(1, activation='relu')(x)
    both5 = Dense(2, activation='linear')(x)
    emission5 = Dense(1, activation='relu')(x)
    both6 = Dense(1, activation='linear')(x)
    emission6 = Dense(7, activation='relu')(x)
    both7 = Dense(1, activation='linear')(x)
    emission7 = Dense(3, activation='relu')(x)

    absorption = tf.keras.layers.Dense(units=8, activation=inverted_relu)(x)



    # Concatenating line strengths and velocity dispersion
    decoded = Concatenate()([both0, emission0, both1, emission1, both2, emission2, both3, emission3, both4, emission4, both5,
                             emission5, both6, emission6, both7, emission7, absorption, sigmoid_part])

    # Create the model
    model = Model(inputs=input_layer, outputs=decoded)


    return model


##LEGACY MODEL
def build_model():
    inputs = tf.keras.Input(shape=(len_data,))

    # Expand dimensions for convolutional layers
    x = tf.expand_dims(inputs, axis=-1)

    # Convolutional layers with dropout
    x = layers.Conv1D(16, 6, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Conv1D(64, 12, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Conv1D(128, 32, activation='relu')(x)
    x = layers.Dropout(0.3)(x)

    # Multiheaded attention mechanism with dropout
    x = layers.MultiHeadAttention(num_heads=3, key_dim=128, dropout=0.3)(x, x)

    # Flatten the output
    x = layers.Flatten()(x)


    sigmoid_part = Dense(1, activation='linear')(x)#, bias_initializer=tf.keras.initializers.Constant(np.log10(1.01)))(x)  # Use linear here because ScaledSigmoid applies the sigmoid
    sigmoid_part = ScaledSigmoid(min_val=1.38, max_val=9.2)(sigmoid_part)


    both0 = Dense(3, activation='linear')(x)#, bias_initializer=tf.keras.initializers.Constant(1.0)
    emission0 = Dense(4, activation='relu')(x)
    both1 = Dense(2, activation='linear')(x)
    emission1 = Dense(1, activation='relu')(x)
    both2 = Dense(1, activation='linear')(x)
    emission2 = Dense(2, activation='relu')(x)
    both3 = Dense(4, activation='linear')(x)
    emission3 = Dense(2, activation='relu')(x)
    both4 = Dense(1, activation='linear')(x)
    emission4 = Dense(1, activation='relu')(x)
    both5 = Dense(2, activation='linear')(x)
    emission5 = Dense(1, activation='relu')(x)
    both6 = Dense(1, activation='linear')(x)
    emission6 = Dense(7, activation='relu')(x)
    both7 = Dense(1, activation='linear')(x)
    emission7 = Dense(3, activation='relu')(x)

    absorption = tf.keras.layers.Dense(units=8, activation=inverted_relu)(x)



    # Concatenating the two parts back together
    decoded = Concatenate()([both0, emission0, both1, emission1, both2, emission2, both3, emission3, both4, emission4, both5,
                             emission5, both6, emission6, both7, emission7, absorption, sigmoid_part])

    


    autoencoder = Model(inputs, decoded)
    

    return autoencoder


#function to build pretrained model for fine tuning
def build_finetune_model():

    """
    Builds a pre-trained model for fine-tuning.
    
    Returns:
    Model: A pre-trained model for fine-tuning.
    
    """
    
    #Initialize custom functions for tensorflow

    #inverted relu function
    def inverted_relu(x):
        return -tf.nn.relu(x)  # Negate the output of the standard ReLU

    #scaled sigmoid function for velocity dispersion
    class ScaledSigmoid(Layer):
        def __init__(self, min_val, max_val, steepness=0.1, **kwargs):
            super(ScaledSigmoid, self).__init__(**kwargs)
            self.min_val = min_val
            self.max_val = max_val
            self.steepness = steepness  # Introduce steepness parameter

        def call(self, inputs, **kwargs):
            # Scale inputs by steepness factor before applying the sigmoid
            sigmoid = tf.nn.sigmoid(inputs * self.steepness)
            # Scale the output of the sigmoid from min_val to max_val
            return self.min_val + (self.max_val - self.min_val) * sigmoid

        def get_config(self):
            config = super(ScaledSigmoid, self).get_config()
            config.update({
                'min_val': self.min_val,
                'max_val': self.max_val,
                'steepness': self.steepness  # Make sure to include steepness in the config
            })
            return config
        
    #global sum pooling for attention mechanism dimensionality reduction
    class GlobalSumPooling1D(Layer):
        def __init__(self, **kwargs):
            super(GlobalSumPooling1D, self).__init__()

        def call(self, inputs):
            return tf.reduce_sum(inputs, axis=1)

    #load pre-trained model
    model = load_model('all_object_model.keras', custom_objects = {'ScaledSigmoid': ScaledSigmoid,
                                                                    'GlobalSumPooling1D': GlobalSumPooling1D,
                                                                    'inverted_relu': inverted_relu})
    
    #print model summary
    model.summary()


    return model


def build_model():
    #create spectrumEncoder model
    model = SpectrumEncoder()

    #print model summary
    model.summary()


    return model

def index_from_end(array, index):
    """
    Given an array and an index, return the index starting from the end of the array.
    
    Parameters:
    array (list or np.ndarray): The input array.
    index (int): The index from the beginning of the array.
    
    Returns:
    int: The index starting from the end of the array.
    """
    array_length = len(array)
    return array_length - 1 - index

def index_from_beginning(array, index_from_end):
    """
    Given an array and an index from the end, return the index starting from the beginning of the array.
    
    Parameters:
    array (list or np.ndarray): The input array.
    index_from_end (int): The index from the end of the array.
    
    Returns:
    int: The index starting from the beginning of the array.
    """
    array_length = len(array)
    return array_length - 1 - index_from_end


def redshift_to_shift(z, wavelength_template):

    """
    Convert redshift to pixel shift based on the wavelength template.
    
    Parameters:
    z (float): The redshift value.
    wavelength_template (np.ndarray): The wavelength template.
    
    Returns:
    int: The pixel shift corresponding to the redshift.
    """

    #initialize wavelength template and find maximimum observed wavelength
    x = wavelength_template  # Assume x is in log10(wavelength)
    obs_wavelength_log = x[-1]
    
    # Calculate delta_log from the redshift
    delta_log = np.log10(1 + z)
    
    # Compute the expected emission wavelength in log scale
    em_wavelength_log = obs_wavelength_log - delta_log
    
    # Find the index in the wavelength template closest to em_wavelength_log
    shift = np.argmin(np.abs(x - em_wavelength_log))
    
    return shift - len_data + 1

def shift_to_redshift(shift, wavelength_template):

    """
    Convert pixel shift to redshift based on the wavelength template.
    
    Parameters:
    shift (int): The pixel shift value.
    wavelength_template (np.ndarray): The wavelength template.
    
    Returns:
    float: The redshift corresponding to the pixel shift.
    """

    #initialize wavelength template and find maximimum observed wavelength
    x = wavelength_template  # Assume x is in log10(wavelength)
    obs_wavelength_log = x[-1]

    # Find the emission wavelength in log scale
    em_wavelength_log = x[shift + len_data - 1]
    
    # Calculate delta_log
    delta_log = obs_wavelength_log - em_wavelength_log
    
    # Convert delta_log back to redshift
    z = 10 ** delta_log - 1
    
    return z



def compute_batch_gaussians_tf(template, batch_amplitudes, batch_std_devs):
    """
    Compute Gaussian distributions for a batch of spectra using TensorFlow operations.
    
    Parameters:
    template (tf.Tensor): The wavelength template for the spectra.
    batch_amplitudes (tf.Tensor): The amplitudes of the Gaussian distributions.
    batch_std_devs (tf.Tensor): The standard deviations of the Gaussian distributions. 
    
    Returns:
    tf.Tensor: The summed Gaussian distributions for the batch of spectra.
    """

    # Cast the inputs to float32 and convert wavelength template to angstroms
    template = tf.cast(10**template, dtype = tf.float32)   # Units = log10(Anstroms)
    batch_amplitudes = tf.cast(batch_amplitudes, dtype = tf.float32)    #Units = arbitrary flux units
    batch_std_devs = tf.cast(batch_std_devs+5e-7, dtype = tf.float32)    #Units = Anstroms (assuming center at 6500 angstroms)
    
    # Initialize line centers 
    means = tf.constant(np.array([
        1033.82, 1215.24, 1240.81, 1305.53, 1335.31,
        1397.61, 1399.8, 1549.48, 1640.4, 1665.85,
        1857.4, 1908.734, 2326.0, 2439.5, 2799.117,
        3346.79, 3426.85, 3727.092, 3729.875, 3889.0,
        4072.3, 4102.89, 4341.68, 4364.436, 4862.68,
        4932.603, 4960.295, 5008.240, 6302.046, 6365.536,
        6529.03, 6549.86, 6564.61, 6585.27, 6718.29,
        6732.67, 3934.777, 3969.588, 4305.61, 5176.7,
        5895.6, 8500.36, 8544.44, 8664.52
    ]), dtype=tf.float32)  #Units = log10(Anstroms)

    # Ensure batch_std_devs is a 1D array
    if len(batch_std_devs.shape) != 1:
        raise ValueError("batch_std_devs must be a 1D array")

    # Expand batch_std_devs to match the dimensions needed for broadcasting
    std_dev_expanded = tf.reshape(batch_std_devs, (-1, 1, 1))  # [B, 1, 1]

    # Compute Gaussian distributions
    expanded_template = tf.expand_dims(template, 1)  # [N, 1]
    expanded_means = tf.expand_dims(means, 0)  # [1, M]

    # Apply broadcasting to calculate the Gaussians
    gaussians = (1/(std_dev_expanded*tf.math.sqrt(2*math.pi)))*tf.exp(-0.5 * tf.square((expanded_template - expanded_means) / std_dev_expanded))  # [B, N, M]

    # Transpose and expand gaussians for correct broadcasting
    gaussians = tf.transpose(gaussians, perm=[0, 2, 1])  # [B, M, N]

    # Expand batch amplitudes
    batch_amplitudes_expanded = tf.expand_dims(batch_amplitudes, 2)  # [B, M, 1]

    # Multiply Gaussians by batch amplitudes
    gaussians_scaled = gaussians * batch_amplitudes_expanded  # [B, M, N]

    # Sum along the means axis
    summed_gaussians = tf.reduce_sum(gaussians_scaled, axis=1)  # [B, N]

    return summed_gaussians


##LEGACY FUNCTION
def slice_2d_tensor_by_1d_indices(data_2d, indices_1d, data_length = len_data):
    
    """
    Slice a 2D tensor along the second dimension using 1D indices.
    
    Parameters:
    data_2d (tf.Tensor): The 2D tensor to slice.
    indices_1d (tf.Tensor): The 1D tensor of indices to use for slicing.
    data_length (int): The length of the data along the second dimension.
    
    Returns:
    tf.Tensor: The sliced 2D tensor based on the 1D indices.
    """

    # Calculate continuous indices within allowed bounds
    idx_min = indices_1d 
    idx_max = idx_min + data_length

    max_len = tf.cast(tf.shape(data_2d)[1], tf.float32)
    idx_max = tf.minimum(idx_max, max_len)

    # Create a meshgrid for the batch and indices
    idx_range = tf.linspace(0.0, 1.0, len_data)  # Create len_data points between 0 and 1
    idx_range = tf.expand_dims(idx_range, 0)  # Shape: [1, len_data]

    # Interpolate between idx_min and idx_max
    idxs = idx_min[:, None] + idx_range * (idx_max - idx_min)[:, None]
    idxs = tf.clip_by_value(idxs, 0.0, max_len - 1.0)  # Ensure indices are within valid range

    # Perform bilinear interpolation
    idx_floor = tf.floor(idxs)
    idx_ceil = idx_floor + 1
    idx_ceil = tf.minimum(idx_ceil, max_len - 1.0)  # Ensure idx_ceil does not exceed data length

    idx_floor = tf.cast(idx_floor, tf.int32)
    idx_ceil = tf.cast(idx_ceil, tf.int32)

    # Get values at idx_floor and idx_ceil
    def gather_vals(data, indices):
        batch_indices = tf.tile(tf.range(tf.shape(data)[0])[:, None], [1, tf.shape(indices)[1]])
        gather_indices = tf.stack([batch_indices, indices], axis=-1)
        return tf.gather_nd(data, gather_indices)

    values_floor = gather_vals(data_2d, idx_floor)
    values_ceil = gather_vals(data_2d, idx_ceil)

    # Calculate the weights for interpolation
    weights = idxs - tf.cast(idx_floor, tf.float32)

    # Interpolate between floor and ceil values
    result_tensor = values_floor * (1.0 - weights) + values_ceil * weights

    return result_tensor



def find_min_euclidean_distance_index(large_arrays, tiny_arrays, alpha=1, k=250, radius=1.0):
    """
    Find the indices of the best matches between two sets of arrays based on the mean-squared-error 
    and cosine similarity.
    
    Parameters:
    large_arrays (tf.Tensor): UNEZ templates outputted by compute_batch_gaussians_tf.
    tiny_arrays (tf.Tensor): The preprocessed input spectra.
    alpha (float): The weight to assign to the Mean Squared Error (MSE) loss. Default is 1.
    k (int): The number of top matches to consider. Default is 5.
    radius (float): The radius for the exponential decay function. Default is 1.0.
    
    Returns:
    tf.Tensor: The indices of the best matches.
    tf.Tensor: The loss value for the best matches.
    tf.Tensor: The hybrid loss values for all matches.
    """
    # Ensure data types are consistent
    large_arrays = tf.cast(large_arrays, dtype=tf.float32)
    tiny_arrays = tf.cast(tiny_arrays, dtype=tf.float32)

    # Dimensions of the inputs
    batch_size = tf.shape(large_arrays)[0]
    large_length = tf.shape(large_arrays)[1]
    tiny_length = tf.shape(tiny_arrays)[1]
    # Determine the number of sliding windows possible
    num_windows = large_length - tiny_length + 1

    # Create indices for all windows
    indices = tf.expand_dims(tf.range(num_windows), 0) + tf.expand_dims(tf.range(tiny_length), 1)

    # Batch and tile indices to gather windows across the batch
    indices = tf.tile(indices[None, :, :], [batch_size, 1, 1])

    # Gather windows from the large arrays
    large_windows = tf.gather(large_arrays, indices, batch_dims=1)

    # Compute squared differences and mean over the tiny_length dimension to get the MSE
    squared_diff = tf.square(large_windows - tiny_arrays[:, :, None])
    mse = tf.reduce_mean(squared_diff, axis=1)

    # Compute dot products and cosine similarities
    dot_products = tf.reduce_sum(tf.multiply(large_windows, tiny_arrays[:, :, None]), axis=1)
    norm_large = tf.norm(large_windows, axis=1)
    norm_tiny = tf.norm(tiny_arrays, axis=1, keepdims=True)
    cosine_similarities = dot_products / (norm_large * norm_tiny)

    # Hybrid loss calculation
    hybrid_loss = alpha * mse + (1 - alpha) * -cosine_similarities  # Maximizing cosine similarity is minimizing its negative
    
    # Find the indices of the top k smallest hybrid loss values in each batch
    values, indices = tf.nn.top_k(-hybrid_loss, k, sorted=True)
    values = -values  # Convert back to positive values

    # Generate exponentially decaying weights based on the radius
    weights = tf.exp(-tf.range(k, dtype=tf.float32) / radius)
    weights /= tf.reduce_sum(weights)  # Normalize weights to sum to 1

    # Calculate the weighted average of these top k values
    weighted_top_k_avg = tf.reduce_sum(values * weights, axis=1)

    # Calculate the average of these weighted top k values
    loss = tf.reduce_mean(weighted_top_k_avg)

    # Return the indices corresponding to the smallest hybrid loss (i.e., best matches)
    best_match_indices = tf.argmin(hybrid_loss, axis=1)

    return best_match_indices, loss, hybrid_loss


def exponential_decay_radius(epoch, initial_radius, min_radius, total_epochs):
    """
    Compute the exponential decay radius for the UNEZ algorithm based on the current epoch.
    
    Parameters:
    epoch (int): The current epoch number.
    initial_radius (float): The initial radius value.
    min_radius (float): The minimum radius value.
    total_epochs (int): The total number of epochs.
    """
    
    #uncomment to stop exponential decay at total_epochs
    """if epoch >= total_epochs:
        return min_radius"""
    
    # Compute the decay rate based on the initial and minimum radius
    decay_rate = (initial_radius / min_radius) ** (1 / total_epochs)


    return initial_radius * (decay_rate ** -epoch)



class unez():
    """
    UNEZ (UNsupervised  Emission and Z) algorithm for redshift estimation.
    """


    def __init__(self, learning_rate, wavelength_template, top_k_values = 1900):
        """
        Initialize the UNEZ algorithm with the specified learning rate and wavelength template.

        Parameters:
        learning_rate (float): The learning rate for the UNEZ algorithm.
        wavelength_template (np.ndarray): The wavelength template for the spectra.
        top_k_values (int): The number of top k values to consider for matching. Default is 1900.
        """

        # Initialize the learning rate and optimizer
        self.learning_rate = learning_rate
        self.opt = Adam(learning_rate=self.learning_rate)#, clipnorm=1)

        # Build the autoencoder model
        self.autoencoder = build_model()

        # Initialize emission line centers
        self.emissions = 10**means
        
        # Initialize the wavelength template and top_k values
        self.wavelength_template = tf.constant(wavelength_template, dtype = tf.float32)
        self.top_k = top_k_values

        # Initialize the alpha values for the hybrid loss
        self.alpha0 = 0.9
        self.alpha1 = 0.95

    ##LEGACY
    def train_step(self, autoencoder, batch_data, lammies, wavelength_template, alpha):
        with tf.GradientTape() as tape:
            #alpha = tf.exp(log_alpha)
            decoded = autoencoder(batch_data, training=True)  # [batch_size, len(numbers)]
            gaussians_batch = compute_batch_gaussians_tf(wavelength_template, decoded[:, :-1])

            best_starting_lambdas = find_min_euclidean_distance_index(
                gaussians_batch, batch_data
            )
            lammies = tf.cast(lammies, dtype=tf.float32)
            delta_lam = tf.sqrt(tf.reduce_mean(tf.square(tf.cast(lammies, dtype=tf.float32)- tf.cast(best_starting_lambdas, dtype=tf.float32))))
            gaussians_batch = slice_2d_tensor_by_1d_indices(gaussians_batch, decoded[:, -1])

            true_lammy_loss = tf.reduce_mean(tf.square(lammies-(decoded[:, -1]*4076+1000)))
            reconstruction_loss = tf.reduce_mean(tf.square(tf.cast(batch_data, tf.float32) - tf.cast(gaussians_batch, tf.float32)))
            lammy_loss = tf.reduce_mean(tf.square(tf.cast(best_starting_lambdas, tf.float32) - tf.cast((decoded[:, -1] * 4076) + 1000, tf.float32)))
            loss =  lammy_loss + alpha * reconstruction_loss

        #trainable_variables = autoencoder.trainable_variables# + [log_alpha]
        autoencoder_gradients = tape.gradient(loss, autoencoder.trainable_variables)
        self.opt.apply_gradients(zip(autoencoder_gradients, autoencoder.trainable_variables))
        del best_starting_lambdas, lammies, decoded
        #gc.collect()()

        return reconstruction_loss, lammy_loss, loss, gaussians_batch, true_lammy_loss, delta_lam
    

    #single training step
    def pretrain_step(self, batch_data, lammies, alpha, lambdas, r):
        """
        Perform a single training step for the UNEZ algorithm.

        Parameters:
        batch_data (tf.Tensor): The batch of input spectra.
        lammies (tf.Tensor): The true shift values for the input spectra.
        alpha (float): The weight to assign to the reconstruction loss.
        lambdas (tf.Tensor): The true shift values for the input spectra.
        r (float): The radius for the exponential decay function.

        Returns:
        loss (tf.Tensor): The total loss value for the training step.
        gaussians_batch (tf.Tensor): The Gaussian distributions for the batch of spectra.
        delta_lam (tf.Tensor): The average difference between true shifts and best starting shifts.
        hybrid_loss (tf.Tensor): The hybrid loss values for the batch of spectra.
        best_starting_lambdas (tf.Tensor): The best starting shift values for the batch of spectra.
        true_lammy_loss (tf.Tensor):## LEGACY ##The loss value for the true shift values.
        """

        #begin gradient recording
        with tf.GradientTape(persistent=True) as tape:

            #get the decoded values from the autoencoder
            decoded = self.autoencoder(batch_data, training=True)  # [batch_size, len(numbers)]
            
            #cast the shifts to float32
            lammies = tf.cast(lammies, dtype=tf.float32)
            
            #compute full range gaussian template
            gaussians_batch_full = compute_batch_gaussians_tf(wavelength_template, decoded[:, :-1], decoded[:, -1])
            
            #compute normalization factor
            gaussians_batch_full_norm = tf.norm(gaussians_batch_full, ord='euclidean', axis=1, keepdims=True)
            
            #normalize full range gaussian template to sum to 1
            gaussians_batch_full = gaussians_batch_full/gaussians_batch_full_norm

            #Calculate best shift values, top-k loss, and full range hybrid loss
            best_starting_lambdas, test_loss, hybrid_loss = find_min_euclidean_distance_index(gaussians_batch_full, batch_data, alpha = alpha, radius = r)

            #slice the full range gaussian template to the best starting shifts
            gaussians_batch = slice_2d_tensor_by_1d_indices(gaussians_batch_full, tf.cast(best_starting_lambdas, dtype = tf.float32))   
            
            #causing errors, come back and fix this later
            true_lammy_loss = 0#tf.reduce_mean(tf.square(tf.cast(lammies, dtype = tf.float32)-tf.cast(best_starting_lambdas, dtype = tf.float32)))
            
            #calculate average difference between true shifts and best starting shifts
            delta_lam = tf.sqrt(tf.reduce_mean(tf.square(tf.cast(lammies, dtype=tf.float32)- tf.cast(best_starting_lambdas, dtype=tf.float32))))

            loss = (tf.reduce_mean(test_loss))

        #calculate gradients
        autoencoder_gradients = tape.gradient(loss, self.autoencoder.trainable_variables)

        #zip gradients and their corresponding trainable variables
        grads_and_vars = zip(autoencoder_gradients, self.autoencoder.trainable_variables)

        #apply the gradients to trainable variables
        self.opt.apply_gradients(grads_and_vars)

        #delete variables to free up memory
        del lammies, decoded
        gc.collect()


        return loss, gaussians_batch, delta_lam, hybrid_loss, best_starting_lambdas, true_lammy_loss


    #calculate validation step
    def validation_step(self, batch_data, shifts, r, batch_size=16):

        """
        Perform a single validation step for the UNEZ algorithm.
        
        Parameters:
        batch_data (tf.Tensor): The batch of input spectra.
        shifts (tf.Tensor): The true shift values for the input spectra.
        r (float): The radius for the exponential decay function.
        batch_size (int): The batch size for the validation step.
        
        Returns:
        gaussians_batch_accumulated (list): The Gaussian distributions for the batch of spectra.
        best_shifts_accumulated (list): The best starting shift values for the batch of spectra.
        gaussians_batch_full_accumulated (list): The full range Gaussian distributions for the batch of spectra.
        true_decoded_accumulated (np.ndarray): The decoded values for the batch of spectra.
        delta_lam (float): The average difference between true shifts and best starting shifts.
        accumulated_loss (list): The loss values for the batch of spectra.
        accumulated_hybrid_loss (list): The hybrid loss values for the batch of spectra.
        """

        #transfer data to gpu
        dataset = tf.data.Dataset.from_tensor_slices((batch_data, shifts))

        #split data into batches
        dataset = dataset.batch(batch_size)

        #initialize lists to store data
        gaussians_batch_accumulated = []
        gaussians_batch_full_accumulated = []
        true_shift_loss_accumulated = []
        best_shifts_accumulated = []
        true_decoded_accumulated = []
        delta_lam_accumulated = []
        accumulated_loss = []
        accumulated_hybrid_loss = []

        #iterate through batches of data
        for batch_data_segment, shift_segment in dataset:

            #get the decoded values from the autoencoder
            decoded = self.autoencoder.predict(batch_data_segment, verbose = 0)  # [batch_size, len(numbers)]

            #cast the shifts to float32
            lammies_tmp = tf.cast(shift_segment, dtype=tf.float32)
            
            #compupte full range gaussian template
            gaussians_batch_full = compute_batch_gaussians_tf(wavelength_template, decoded[:, :-1], decoded[:, -1])

            #normalize full range gaussian template to sum to 1
            gaussians_batch_full_norm = tf.norm(gaussians_batch_full, ord='euclidean', axis=1, keepdims=True)
            gaussians_batch_full = gaussians_batch_full/gaussians_batch_full_norm

            #calculate best shift values, top-k loss, and full range hybrid loss
            best_shifts, test_loss, hybrid_loss = find_min_euclidean_distance_index(gaussians_batch_full, batch_data_segment, radius = r)

            #slice the full range gaussian template by the best starting shifts
            gaussians_batch = slice_2d_tensor_by_1d_indices(gaussians_batch_full, tf.cast((best_shifts), dtype = tf.float32))

            #calculate average difference between true shifts and best starting shifts
            delta_lam = tf.sqrt(tf.reduce_mean(tf.square(tf.cast(lammies_tmp, dtype=tf.float32)- tf.cast(best_shifts, dtype=tf.float32))))

            
            #sum the loss values for the batch
            loss = tf.reduce_mean(test_loss)
            
            #accumulate the batch output values
            accumulated_loss.append(loss.numpy())
            delta_lam_accumulated.append(delta_lam.numpy())
            for i in range(len(decoded[:, -1])):
                accumulated_hybrid_loss.append(hybrid_loss[i].numpy())
                gaussians_batch_accumulated.append(gaussians_batch[i].numpy())
                true_decoded_accumulated.append(decoded[i])
                best_shifts_accumulated.append(best_shifts.numpy()[i])
                gaussians_batch_full_accumulated.append(gaussians_batch_full.numpy()[i])

            delta_lam = tf.reduce_mean(delta_lam_accumulated).numpy()

        return gaussians_batch_accumulated,  best_shifts_accumulated, gaussians_batch_full_accumulated, np.array(true_decoded_accumulated), delta_lam, accumulated_loss, accumulated_hybrid_loss



    def train_and_evaluate(self, data, validation, shifts, validation_shifts, train_dataframe, validation_dataframe, widths, validation_snrs, output_dir, epochs=200, batch_size=16):
        
        """
        Trains an autoencoder on the provided data, evaluates its training loss, and returns the encoder model.
        """
        
        print(self.autoencoder.summary())

        #transfer data to gpu
        dataset = tf.data.Dataset.from_tensor_slices((data, shifts))
        dataset = dataset.shuffle(buffer_size=len(data), reshuffle_each_iteration=True).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

        #initialize alpha and loss lists for plotting
        alpha = self.alpha0
        collect_all_loss = []
        collect_all_validation_loss = []
        collect_all_outliers = []

        #initialize training radius
        r_0 = 30#0.005

        #begin training loop
        for epoch in range(epochs):
            r = exponential_decay_radius(epoch, r_0, 0.05, 5)#0.0005, 5)
            

            #initialize time for measuring epoch runtime
            start_time = time.time()


            #legacy code
            """if epoch > 150:
                alpha = self.alpha1"""

            #initialize lists to store evaluation data for the batch
            collect_shift_loss = []
            collect_reconstruction_loss = []
            collect_true_shift_loss = []
            collect_delta_lam = []
            collect_loss = []

            #iterate through batches of data
            for step, (batch_data, batch_shifts) in enumerate(dataset):
                
                #run training step on batch
                loss, gaussians_batch, delta_lam, hybrid_loss, best_starting_shifts, true_shift_loss = self.pretrain_step(batch_data, batch_shifts, alpha, self.top_k, r)   

                #accumulate the batch evaluation values
                collect_delta_lam.append(delta_lam)
                collect_true_shift_loss.append(true_shift_loss)
                collect_loss.append(loss)
            
            #calculate performance values for the epoch
            true_shift_loss = np.mean(collect_true_shift_loss)
            loss = np.mean(collect_loss)
            collect_all_loss.append(loss)
            delta_lam = np.mean(collect_delta_lam)

            #print epoch evaluation values
            print(f'\nEPOCH:{epoch}\n TRUE LAMMY LOSS: {true_shift_loss}\n DELTA_LAM: {delta_lam}\n LOSS: {loss}\n')
            
            #flush output print buffer
            sys.stdout.flush()

            #run validation step
            valid_gaussians_batch, valid_best_shifts, valid_full_gaussians, true_valid_decoded, valid_delta_lam, validation_loss, validation_hybrid_loss = self.validation_step(validation, validation_shifts, r)

            #accumulate validation loss 
            collect_all_validation_loss.append(np.mean(validation_hybrid_loss))

            #print validation evaluation values
            lr = self.opt.learning_rate.numpy()
            print(f'\nVALIDATION:\n VALID DELTA LAM: {valid_delta_lam}\n ADAM LEARNING RATE: {lr}\n ')
            
            #save model
            self.autoencoder.save(f'auto_ez_prototype_{epoch}.keras')
            print(f"Autoencoder Model saved to auto_ez_prototype.keras")













            #this part is a mess. i need to attend to it.

            #round the best starting shifts to integers
            valid_best_shifts_plot = np.array(valid_best_shifts)
            validation_shifts_plot = np.round(np.array(validation_shifts, dtype = np.int32))

            #convert the wavelength template to numpy
            wavelength_template_temp = self.wavelength_template.numpy()

            #convert the best starting shifts to redshift
            z_pred = np.array([shift_to_redshift(i, wavelength_template) for i in valid_best_shifts_plot])
            z_true = np.array([shift_to_redshift(i, wavelength_template) for i in validation_shifts_plot])

            #calculate residuals
            z_slope, z_intercept = np.polyfit(z_true, z_pred, 1)
            y_predicted = z_slope * z_true + z_intercept
            residuals = z_pred - y_predicted

            #calculate outliers
            z_outlier = np.abs([i for i in (z_pred-z_true)/(1+z_true)])

            #calculate NMAD
            z_nmad = np.mean(z_outlier[z_outlier<0.15])

            #calculate number of outliers
            outlier_idxs = [i for i, n in enumerate(z_outlier) if n>0.15]
            num_outliers = len(outlier_idxs)

            print(f'Number of outliers: {len(outlier_idxs)}')

            #initialize results dictionary
            results = {
                'learning_rate': [lr], 'batch_size': [batch_size], 'epochs': [epoch], 'alpha0': [self.alpha0],
                'outliers': [len(outlier_idxs)], 'NMAD': [z_nmad]}
            
            collect_all_outliers.append(num_outliers)   

            # Save hyperparameters and key performance metrics to CSV
            df = pd.DataFrame(results)
            df.to_csv('hyperparameter_tuning_results.csv', index=False)
            del df

            #plot loss
            plt.figure()
            plt.plot(collect_all_validation_loss, label = 'validation loss')
            plt.plot(collect_all_loss, label = 'training loss')
            plt.legend()
            plt.savefig('loss_plot.pdf')

            #plot outlier count
            plt.figure()
            plt.plot(collect_all_outliers, label = 'outliers')
            plt.xlabel('epoch')
            plt.ylabel('number of outliers')
            plt.savefig('outliers.pdf')

            #plot all results
            plot_epoch(epoch, wavelength_template_temp, batch_shifts, best_starting_shifts, batch_data,
                    validation, outlier_idxs, z_true, z_pred, true_valid_decoded, self.emissions, valid_best_shifts_plot,
                    validation_shifts, valid_best_shifts, z_outlier, z_nmad, residuals, validation_hybrid_loss, gaussians_batch, valid_gaussians_batch,
                    valid_full_gaussians, widths)

                    

                
            #initialize time for measuring epoch runtime
            end_time = time.time()
            execution_time = end_time - start_time

            print(f"EPOCH RUNTIME: {execution_time} s")

            #attempt at constraining memory leaks (still not fully working, memory leaks still present)
            del loss, gaussians_batch, true_shift_loss, valid_gaussians_batch

        return self.autoencoder
        



#convert redshift to velocity
#THIS NEEDS TO BE FIXED. I DO NOT TRUST THIS CALCULATION. VALUES SEEM ACCURATE INTUITIVELY BUT I NEED TO VERIFY THIS.   
def width_to_velocity(width):
    """
    Convert the width of an emission line to velocity.

    Parameters:
    width (float): The width of the emission line.

    Returns:
    float: The velocity of the emission line in km/s.
    """
    fwhm = 2.355*width
    center = 6500
    velocity = ((width)/center)*300000

    return velocity



def plot_epoch(epoch, wavelength_template_temp, shifts, best_starting_shifts, batch_data, validation, outlier_idxs, 
         z_true, z_pred, true_valid_decoded, emissions, valid_best_shifts_plot, validation_shifts, valid_best_shifts, 
         z_outlier, z_nmad, residuals, validation_hybrid_loss, gaussians_batch, valid_gaussians_batch, valid_full_gaussians, widths):
            
            """
            Plot the results of an epoch for the UNEZ algorithm.
            
            Parameters:
            epoch (int): The epoch number.
            wavelength_template_temp (np.ndarray): The wavelength template for the spectra.
            shifts (tf.Tensor): The true shift values for the input spectra.
            best_starting_shifts (tf.Tensor): The best starting shift values for the input spectra.
            batch_data (tf.Tensor): The batch of input spectra.
            validation (tf.Tensor): The validation dataset.
            outlier_idxs (list): The indices of the outliers.
            z_true (np.ndarray): The true redshift values.
            z_pred (np.ndarray): The predicted redshift values.
            true_valid_decoded (np.ndarray): The decoded values for the validation dataset.
            emissions (np.ndarray): The emission line centers.
            valid_best_shifts_plot (np.ndarray): The best starting shift values for the validation dataset.
            validation_shifts (tf.Tensor): The true shift values for the validation dataset.
            valid_best_shifts (np.ndarray): The best starting shift values for the validation dataset.
            z_outlier (np.ndarray): The redshift outliers.
            z_nmad (float): The normalized median absolute deviation of the redshift values.
            residuals (np.ndarray): The residuals between true and predicted redshift values.
            validation_hybrid_loss (list): The hybrid loss values for the validation dataset.
            gaussians_batch (np.ndarray): The Gaussian distributions for the batch of spectra.
            valid_gaussians_batch (np.ndarray): The Gaussian distributions for the validation dataset.
            valid_full_gaussians (np.ndarray): The full range Gaussian distributions for the validation dataset.
            widths (tf.Tensor): The true width values for the input spectra.
            
            Returns:
            None
            
            """

            #INITIALIZE FOLDERS
            if os.path.exists(f'outliers/{epoch}'):
                shutil.rmtree(f'outliers/{epoch}')
                
            if os.path.exists(f'outliers/{epoch}/fits'):
                shutil.rmtree(f'outliers/{epoch}/fits')

            if os.path.exists(f'outliers/{epoch}/confidence'):
                shutil.rmtree(f'outliers/{epoch}/confidence')
            
            os.makedirs(f'outliers/{epoch}')
            os.makedirs(f'outliers/{epoch}/confidence')
            os.makedirs(f'outliers/{epoch}/fits')

            




            #TRAINING PLOT
            plt.figure()
            plt.plot(10**(wavelength_template_temp[np.array(shifts[0], dtype = np.int32): np.array(shifts[0], dtype = np.int32)+len(batch_data[0])]), batch_data[0])
            plt.plot(10**(wavelength_template_temp[np.array(best_starting_shifts[0], dtype = np.int32): np.array(best_starting_shifts[0], dtype = np.int32)+len(batch_data[0])]), gaussians_batch[0], alpha = 0.8)
            plt.xlabel('Wavelength [$\AA$]')
            plt.ylabel('Flux')
            plt.savefig(f'output_plots/emission_line{epoch}.pdf')
            





            #SNR PLOT
            plt.figure()
            x = validation_snrs[outlier_idxs]
            y = abs(z_pred[outlier_idxs]-z_true[outlier_idxs])
            indices = np.argsort(x)

            x = x[indices]
            y = y[indices]

            plt.hist(x, bins = 20)
            plt.xlabel('SNR')
            plt.ylabel('Outlier Count')
            plt.savefig(f'snr_plots/snr_plot{epoch}.pdf')






            








            
            


            






            
            #TRUE VS PREDICTED REDSHIFT
            test1 = [i for i, n in enumerate(z_outlier) if n>0.15]

            fig = plt.figure(figsize=(8, 6))  # Create a figure object

            # Create a GridSpec with 3 rows and 1 column, but the first row takes up 2/3 of the space
            gs = fig.add_gridspec(3, 1, height_ratios=[2, 1, 1])

            # First subplot (2/3 of the figure height)
            ax1 = fig.add_subplot(gs[0:2, 0])
            ax1.set_ylabel('Predicted Redshift')
            ax1.set_xlabel('True Redshift')
            ax1.plot(z_true, z_pred, 'o', markersize=0.5)
            textstr = f'Mean NMAD: {z_nmad}'
            #'\n'.join(())
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            plt.text(0.015, 0.985, textstr, transform=plt.gca().transAxes, fontsize=6,
                    verticalalignment='top', horizontalalignment='left', bbox=props)



            # Second subplot (1/3 of the figure height)
            ax2 = fig.add_subplot(gs[2, 0])
            ax2.scatter(z_true[residuals<0.15], residuals[residuals<0.15], color='red', marker='o', s=0.5)  # 's' is the marker size
            ax2.axhline(y=0, color='black', linestyle='--')  # Adds a horizontal line at zero for reference
            ax2.set_xlabel('True Redshift')
            ax2.set_ylabel('Residuals')

            # Adjust layout
            plt.tight_layout()

            # Save the figure
            plt.savefig(f'z_plots/z_plot{epoch}.pdf')

            plt.figure()
            plt.hist(z_outlier, bins = 70)
            plt.xlabel('Redshift')
            plt.ylabel('Outlier Count')
            plt.savefig('redshift_outlier_hist/redshift_outlier.pdf')



            
          


            #WIDTHS PLOT
            plt.figure()
            ratios = abs(np.array([width_to_velocity(true_valid_decoded[i, -1])-width_to_velocity(widths[i]) for i in range(len(widths))]))

            plt.hist(ratios, bins = 100)
            plt.xlabel('abs(True Velocity - Predicted Velocity)')
            plt.ylabel('Count')
            plt.savefig(f'widths_plots/width_plot{epoch}.pdf')

            plt.figure()
            plt.hist(np.array([width_to_velocity(true_valid_decoded[i, -1]) for i in range(len(widths))]), bins = 100)
            plt.xlabel('Predicted Velocity')
            plt.ylabel('Count')
            plt.savefig(f'widths_plots/predicted_width_plot{epoch}.pdf')

            plt.figure()
            plt.hist(np.array([width_to_velocity(widths[i]) for i in range(len(widths))]), bins = 100)
            plt.xlabel('True Velocity')
            plt.ylabel('Count')
            plt.savefig(f'widths_plots/true_width_plot{epoch}.pdf')


            #WIDTHS SNR PLOT
            plt.figure()
            ratios = true_valid_decoded[:, -1]/widths
            plt.plot(validation_snrs, ratios, 'o', markersize = 0.5)
            plt.xlabel('SNR')
            plt.ylabel('True Velocity/Predicted Velocity')
            plt.savefig(f'widths_snr/width_snr_{epoch}.pdf')


#LEGACY CODE
def plot_outlier(args):
    outlier_idx, wavelength_template_temp, validation_lammy, valid_best_lambdas, valid_best_lambdas_plot, emissions, validation, valid_full_gaussians, z_true, z_pred, valid_gaussians_batch, epoch, validation_hybrid_loss, noiseless_valid, decoded, validation_dataframe, widths = args
    plt.figure()
    plt.ylabel('Flux')
    plt.xlabel(r'Wavelength ($\AA$)')  
    
    first_validation = validation_dataframe.iloc[outlier_idx].values

    closest_index_true = (np.abs(wavelength_template_temp - (((wavelength_template_temp[1]-wavelength_template_temp[0])*(validation_lammy[outlier_idx]))+wavelength_template_temp[outlier_idx]))).argmin()
    closest_index_pred = (np.abs(wavelength_template_temp - (((wavelength_template_temp[1]-wavelength_template_temp[0])*(valid_best_lambdas[outlier_idx]))+wavelength_template_temp[outlier_idx]))).argmin()

    locs = [loc for loc, val in enumerate(first_validation[:-1]) if val>0]
    
    #test_template = (wavelength_template_temp[np.array(valid_best_lambdas[0], dtype = np.int32): np.array(valid_best_lambdas[0], dtype = np.int32)+len_data])
    
    displacement = 0.05
    for loc in locs :
        lam_shift_pred = wavelength_template_temp[valid_best_lambdas_plot[outlier_idx]]+emissions[loc]-wavelength_template_temp[int(validation_lammy[outlier_idx])]
        closest_index = (np.abs(wavelength_template_temp - (lam_shift_pred))).argmin()
        closest_true_index = (np.abs(wavelength_template_temp - emissions[loc])).argmin()

        plt.axvline(x=(emissions[loc]), color='blue', linestyle='solid', linewidth = 0.2)
        plt.text((emissions[loc]), max(validation[outlier_idx])+displacement, validation_dataframe.columns[loc], color='blue', rotation=0, verticalalignment='bottom', horizontalalignment='center', fontsize = 6)

        #if self.emissions[loc] < (((wavelength_template_temp[1]-wavelength_template_temp[0])*(validation_lammy[0]))+wavelength_template_temp[0])+len_data and self.emissions[loc] > (((wavelength_template_temp[1]-wavelength_template_temp[0])*(validation_lammy[0]))+wavelength_template_temp[0]):
        plt.axvline(x=(emissions[loc]-wavelength_template_temp[0]+wavelength_template_temp[valid_best_lambdas_plot[outlier_idx]]), color='red', linestyle=':', linewidth = 0.5)
        plt.text((emissions[loc]-wavelength_template_temp[0]+wavelength_template_temp[valid_best_lambdas_plot[outlier_idx]]), -0.01-displacement, validation_dataframe.columns[loc], color='red', rotation=0, verticalalignment='bottom', horizontalalignment='center', fontsize = 6)

        if displacement >= -0.05:
            displacement = displacement - 0.01

        elif displacement <= 0.05:
            displacement = displacement + 0.01
    
    plt.plot(10**(wavelength_template_temp[int(validation_lammy[outlier_idx]): int(validation_lammy[outlier_idx]+len_data)]), noiseless_valid, color = 'blue', linewidth = 0.7, label = 'Noiseless Data')
    plt.plot(10**(wavelength_template_temp[int(validation_lammy[outlier_idx]): int(validation_lammy[outlier_idx]+len_data)]), validation[outlier_idx], color = 'gray', linewidth = 0.5, alpha = 0.875, label = 'Noisy Data')
    plt.plot(10**(wavelength_template_temp-wavelength_template_temp[0]+wavelength_template_temp[valid_best_lambdas_plot[outlier_idx]]), valid_full_gaussians[outlier_idx], linewidth = 0.2, color = 'red', alpha = 0.7, label = 'Neural Network Output')

    velocity = width_to_velocity(widths[outlier_idx])

    textstr = '\n'.join((
        f'True Redshift: {z_true[outlier_idx]:.2f}',
        f'Predicted Redshift: {z_pred[outlier_idx]:.2f}',
        f'$\Delta$z: {abs(z_true[outlier_idx]-z_pred[outlier_idx]):.2f}',
        f'$\Delta$v: {velocity:.2f} [km/s]',
        f'Predited $\Delta$v: {width_to_velocity(decoded[outlier_idx][-1])}'

    ))



    plt.ylim(min([min(valid_gaussians_batch[outlier_idx]), min(validation[outlier_idx]), min(noiseless_valid)])-0.05, max([max(valid_gaussians_batch[outlier_idx]), max(validation[outlier_idx]), max(noiseless_valid)])+0.05)
    # Add a text box in the top right corner
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    
    plt.text(0.015, 0.985, textstr, transform=plt.gca().transAxes, fontsize=6,
            verticalalignment='top', horizontalalignment='left', bbox=props)
    plt.legend(fontsize = 6)

    
    plt.xlabel('Wavelength [$\AA$]')
    plt.ylabel('Flux')


    plt.savefig(f'outliers/{epoch}/fits/emission_line{outlier_idx}.pdf')


    plt.figure()
    plt.ylabel('$L_{z}$')
    plt.xlabel('Redshift')
    # Compute x-values for the plot
    x_values = (10**((np.array(range(len(validation_hybrid_loss[outlier_idx])))*(wavelength_template_temp[1]-wavelength_template_temp[0]))))-1
    # Plot the lines
    plt.plot(x_values, validation_hybrid_loss[outlier_idx])
    # Compute the x-value for the orange dot
    orange_dot_x = (10**(((int(validation_lammy[outlier_idx]))*(wavelength_template_temp[1]-wavelength_template_temp[0]))))-1
    # Plot the orange dot
    #plt.plot(orange_dot_x, validation_hybrid_loss[outlier_idx][tf.cast(tf.floor(validation_lammy[outlier_idx]), tf.int32).numpy()-1000].numpy(), 'o')
    # Plot vertical dotted line at the x-value of the orange dot
    plt.axvline(x=orange_dot_x, color='orange', linestyle='dotted')
    # Find the x-value at the minimum of the first line
    min_x_value = x_values[np.argmin(validation_hybrid_loss[outlier_idx])]
    # Plot vertical dotted line at the x-value of the minimum of the first line
    plt.axvline(x=min_x_value, color='blue', linestyle='dotted')
    # Calculate the difference between the x-values
    x_difference = abs(orange_dot_x - min_x_value)
    # Create a text box with the x-values and the difference
    textstr = '\n'.join((
        f'True Redshift: {orange_dot_x:.2f}',
        f'Predicted Redshift: {min_x_value:.2f}',
        f'$\Delta$z: {x_difference:.2f}',
    ))

    # Add a text box in the top right corner
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.gcf().text(0.95, 0.95, textstr, fontsize=10, verticalalignment='top', horizontalalignment='right', bbox=props)
    # Save the figure
    plt.savefig(f'outliers/{epoch}/confidence/emission_line{outlier_idx}.pdf')
    plt.close('all')


#print entire script for archiving purposes
def main():
    # Print the current script
    with open(__file__, 'r') as file:
        # Read the content of the file
        content = file.read()
        # Print the content
        print(content)













if __name__ == '__main__':
    try:
        multiprocessing.set_start_method('fork')  # or 'forkserver'
    except:
        True == True
    #main()

    #load redshifts to extract number of data points
    z = np.loadtxt('starburst_unez_z_true.txt')

    #load simulated data
    data_points = len(z)#35000
    min_amplitude = 1
    max_amplitude = 10
    min_lambda = 3000
    max_lambda = 11000

    data_producer = data_producer(data_points, min_amplitude, max_amplitude, min_lambda, max_lambda)
    
    #initialize data
    pure_data, lambdas, dataframe, widths = data_producer.initialize_data(wavelength_template)

    #inject noise
    noisy_data = data_producer.noise_injector()

    #calculate snrs
    snrs = data_producer.snrs

    """dataframe = pd.read_csv('nda_dataframe.csv').iloc[:data_points]
    lambdas = np.loadtxt('nda_lambdas.txt')[:data_points]
    widths = np.loadtxt('nda_widths.txt')[:data_points]
    pure_data = np.loadtxt('nda_pure_data.txt')[:data_points]
    noisy_data = np.loadtxt('nda_noisy_data.txt')[:data_points]
    snrs = np.loadtxt('nda_snrs.txt')[:data_points]"""


    #LEGACY CODE THIS IS USED ONLY TO INITIALIZE UNEZ BUT IS NOT USED IN TRAINING. IF NO REAL SPECTRA ARE PROVIDED, 
    # THESE SIMULATED SPECTRA WILL BE USED TO TRAIN UNEZ
    #uncomment to reshuffle and save the data
    indices = np.arange(lambdas.shape[0])
    np.random.shuffle(indices)
    indices = np.argsort(snrs)[::-1]


    lambdas = lambdas[indices]
    pure_data = pure_data[indices]
    noisy_data = noisy_data[indices]
    dataframe = dataframe.iloc[indices].reset_index(drop = True)
    widths = np.array(widths).flatten()[indices]
    snrs = np.array(snrs)[indices] 

    #load real data
    flux = np.loadtxt('flux_all.txt')[indices]

    #normalize data
    noisy_norms = np.linalg.norm(flux, axis=1, keepdims=True)
    #pure_norms = np.linalg.norm(pure_data, axis=1, keepdims=True)

    noisy_data = flux/noisy_norms
    #pure_data = pure_data/pure_norms

    #load redshifts
    z = np.loadtxt('starburst_unez_z_true.txt')[indices]

    #convert redshifts to pixel shifts
    lambdas = np.array([redshift_to_shift(i, wavelength_template) for i in z])

    #split data into training and validation sets
    train_dataframe = dataframe.iloc[:int(data_points*5/6)]
    train_lambdas = lambdas[:int(data_points*5/6)]
    train_data = noisy_data[:int(data_points*5/6)]
    train_snrs = snrs[:int(data_points*5/6)]

    validation_dataframe = dataframe.iloc[int(data_points*5/6):]
    validation_lambdas = lambdas[int(data_points*5/6):]
    validation_data = noisy_data[int(data_points*5/6):]
    validation_widths = widths[int(data_points*5/6):]
    validation_snrs = snrs[int(data_points*5/6):]

    #plot a sample spectrum
    plt.figure()
    plt.plot(noisy_data[0])
    plt.savefig('test.pdf')

    #initialize UNEZ model
    unez_model = unez(0.0002, wavelength_template, top_k_values = 1900)

    #train UNEZ model
    trained_unez = unez_model.train_and_evaluate(train_data, validation_data, train_lambdas, validation_lambdas,
                                           train_dataframe, validation_dataframe, validation_widths, validation_snrs, 'ai_test/pseudo_amplitudes/autoencoder_model.keras',
                                           epochs = 75, batch_size = 64)
    