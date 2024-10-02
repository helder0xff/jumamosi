import torch
import argparse
import torch.nn as nn
import os
import re

def snn2mcu_value_cuantization(number_of_bits, 
								float_value, 
								negatives_allowed = False):
	"""
	Converts a float value to a fixed-point representation for 
	CPU implementation.

	Args:
	  number_of_bits: The number of bits to use for the representation.
	  float_value: The float value to convert.
	  negatives_allowed: Whether negative values are allowed. Defaults to False.

	Returns:
	  The integer representation of the float value in fixed-point.
	"""	
	bits = number_of_bits
	if False == negatives_allowed:
		if 0 > float_value:
			float_value = 0.0
	else:
		bits -= 1

	return int(2**bits * float_value)

def snn2mcu_param_dict_extractor(model):
	"""
	Extracts a dictionary of parameters from a given SNN model for 
	CPU implementation.

	Args:
	  model: The SNN model object to extract parameters from.

	Returns:
	  A dictionary containing extracted parameters in a format suitable for 
	  CPU.
	"""	

	# Go through the layers and extract the weights. They will be saved as a 
	# list for each neuron as follow:
	# 'l1_n1_w': [0.322, 0.901, -0.123, ...]
	# It will extract as well the model architecture , that is, the number of 
	# neurons per layer in the following fashion:
	# 'model_architecture': [16, 64, 4] -> 3 layers of 16, 64 and 4 neurons each
	# for instance.
	param_dict = {}
	for name, param in model.named_parameters():
		layer 		= int(re.findall(r'\d+', name)[0])
		key 		= 'l' + str(layer)
		neuron_id 	= 0
		if "weight" in name:
			for neuron in param:
				key = 'l' + str(layer) + '_n' + str(neuron_id) + '_w'
				# Initialize the model_architecture in the case of the first
				# layer. Because the first layer is the hidden layer, not 
				# the input layer, the model architecture is initialized
				# with the number of weights of one of the neurons (the first
				# one) that are coming from the input layer.
				if 1 == layer and 0 == neuron_id:
					param_dict['model_architecture'] = [len(neuron)]				
				for weight in neuron:
					try:
						param_dict[key].append(weight.item())
					except:
						param_dict[key] = [weight.item()]
				neuron_id += 1
			param_dict['model_architecture'].append(len(param))
				
	# Include the number of layers.
	param_dict['number_of_layers'] = len(param_dict['model_architecture'])

	# So far we have weights of the model connections. Lets innclude input 
	# neurons weights as well.
	neuron_id 	= 0
	layer 		= 0
	for n in range(param_dict['model_architecture'][0]):
		key = 'l' + str(layer) + '_n' + str(neuron_id) + '_w'
		try:
			param_dict[key].append(0.9)
		except:
			param_dict[key] = [0.9]
		neuron_id += 1

	return param_dict


def snn2mcu_weights(	file_path, 
						param_dict, 
						precision, 
						negatives_allowed = False):
	"""
	Writes SNN model weights to a file in a format suitable for 
	CPU implementation.

	Args:
	  file_path: The path to the file where weights will be written.
	  param_dict: A dictionary containing extracted SNN model parameters.
	  precision: The number of bits to use for weight quantization.
	  negatives_allowed: Whether negative weights are allowed. Defaults to False.
	"""	

	with open(file_path, 'w') as file:
		#file.write('memory_initialization_radix=10;\n')
		#file.write('memory_initialization_vector=')

		string_memory = '{'
		for l in range(param_dict['number_of_layers']):
			for n in range(param_dict['model_architecture'][l]):
				key = 'l' + str(l) + '_n' + str(n) + '_w' 
				for w in param_dict[key]:
					string_memory += str(snn2mcu_value_cuantization(
						precision, 
						w, 
						negatives_allowed))
					string_memory += ','

		string_memory = string_memory[:-1]
		string_memory += '}'

		file.write(string_memory)

def snn2mcu_thresholds(file_path, param_dict, precision):
	"""
	Writes SNN model neuron thresholds to a file in a format suitable for 
	CPU implementation.

	Args:
	  file_path: The path to the file where thresholds will be written.
	  param_dict: A dictionary containing extracted SNN model parameters.
	  precision: The number of bits to use for threshold quantization.
	"""

	with open(file_path, 'w') as file:
		file.write('memory_initialization_radix=10;\n')
		file.write('memory_initialization_vector=')

		string_memory = ''
		for l in range(param_dict['number_of_layers']):
			for n in range(param_dict['model_architecture'][l]):
				string_memory += str(int((2**precision) - 1))
				string_memory += ' '

		string_memory = string_memory[:-1]
		string_memory += ';'

		file.write(string_memory)

def snn2mcu_leakages(file_path, param_dict, precision):
	"""
	Writes SNN model neuron leakages to a file in a format suitable for 
	CPU implementation.

	Args:
	  file_path: The path to the file where leakages will be written.
	  param_dict: A dictionary containing extracted SNN model parameters.
	  precision: The number of bits to use for leakage quantization.
	"""

	with open(file_path, 'w') as file:
		file.write('memory_initialization_radix=10;\n')
		file.write('memory_initialization_vector=')

		string_memory = ''
		for l in range(param_dict['number_of_layers']):
			for n in range(param_dict['model_architecture'][l]):
				string_memory += str(int(2**precision / 100))
				string_memory += ' '

		string_memory = string_memory[:-1]
		string_memory += ';'

		file.write(string_memory)

# Define Network
class Net(nn.Module):
	"""
	Defines a simple Spiking Neural Network (SNN) architecture with 
	Leaky Integrate-and-Fire (LIF) neurons.

	This network consists of two fully connected layers with 
	LIF non-linearities.

	Attributes:
	  num_inputs: The number of input neurons.
	  num_hidden: The number of hidden neurons in the first layer.
	  num_outputs: The number of output neurons.
	  beta: The decay parameter for the LIF neurons.
	"""	
	def __init__(self):
	    super().__init__()

	    # Initialize layers
	    self.fc1 = nn.Linear(num_inputs, num_hidden)
	    self.lif1 = snn.Leaky(beta=beta)
	    self.fc2 = nn.Linear(num_hidden, num_outputs)
	    self.lif2 = snn.Leaky(beta=beta)

	def forward(self, x):

	    # Initialize hidden states at t=0
	    mem1 = self.lif1.init_leaky()
	    mem2 = self.lif2.init_leaky()
	    
	    # Record the final layer
	    spk2_rec = []
	    mem2_rec = []

	    for step in range(num_steps):
	        cur1 = self.fc1(x)
	        spk1, mem1 = self.lif1(cur1, mem1)
	        cur2 = self.fc2(spk1)
	        spk2, mem2 = self.lif2(cur2, mem2)
	        spk2_rec.append(spk2)
	        mem2_rec.append(mem2)

	    return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)

parser = argparse.ArgumentParser(
	description='*.txt files generation for CPU initialization from SNN model.')
parser.add_argument('--model_path',
                    '-mp',
                    help = 'Path to model.',
                    default = None,
                    type = str,
                    required = True)
parser.add_argument('--out_dir',
                    '-od',
                    help = 'Output directory.',
                    default = None,
                    type = str,
                    required = True)
parser.add_argument('--precision',
                    '-p',
                    help = 'Quantitazion number of bits.',
                    default = None,
                    type = int,
                    required = True)
args = parser.parse_args()

def main():
	model_path 					= args.model_path
	out_dir						= args.out_dir
	weights_file_path 			= out_dir + '/' + "init_weigthts.coe"
	weights_negatives_file_path = out_dir + '/' + "init_weigthts_negatives.coe"
	leakages_file_path 			= out_dir + '/' + "init_leakages.coe"
	thresholds_file_path 		= out_dir + '/' + "init_thresholds.coe"
	precision 					= args.precision

	if not os.path.exists(out_dir):
		os.makedirs(out_dir)



	model = torch.load(model_path, map_location=torch.device('cpu'))
	
	param_dict = snn2mcu_param_dict_extractor(model)
	snn2mcu_weights(weights_file_path, param_dict, precision)
	snn2mcu_weights(weights_negatives_file_path, param_dict, precision, True)
	#snn2mcu_thresholds(thresholds_file_path, param_dict, precision)
	#snn2mcu_leakages(leakages_file_path, param_dict, precision)

main()

# end of file #
