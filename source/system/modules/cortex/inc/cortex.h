/**
 * @file    cortex.h
 * @author  helder
 * @brief   Header file of the cortex.
 */

/* STRUCTS BEGIN */
/**
 * @brief Neuron sturct..
 */
typedef struct cortex_neuron {
    /** Leak.   */
    uint8_t leak;
    /** Spiking threshold.  */
    uint8_t threshold;
    /** Time of no activiy after spike. */
    uint8_t refractory_period_ticks;
    /** Membrame potential. */
    int16_t membrane_potential;
    /** Did the neuron spiked.  */
    uint8_t spike;
    /** Number of input dendrites.  */
    uint8_t connection_number;
    /** Neuron ID.  */
    uint8_t id;
    /** Pointer to the beginning of input weights.  */
    int8_t *weights;
} cortex_neuron_t;
/* STRUCTS END */


/* FUNCTION DECLARATIONS BEGIN */
/**
 * @brief Initialize cortex. So far it does nothing.
 * 
 * @return void.
 */
void cortex_init(void);

/**
 * @brief Main task for freeRTOS.
 * 
 * @param param Pointer to parameters (if any).
 * @return void.
 */
void cortex_main_task_RTOS(void* param);

/**
 * @brief Initialize neuron with the given values..
 * 
 * @param neuron Pointer to the neuron to be initialized.
 * @param leak Leak for the LIF neuron model.
 * @param threshold Threshold above the neuron shal spike.
 * @param refractory_period_ticks Refractory 'time' after spiking.
 * @param connection_number Number of connections of the neuron.
 * @param id Identification Number.
 * @param weights Pointer to the weights of the connections.
 * @return void.
 */
void cortex_neuron_init(   cortex_neuron_t *neuron,
                            uint8_t leak,
                            uint8_t threshold,
                            uint8_t refractory_period_ticks,
                            uint8_t connection_number,
                            uint8_t id,
                            int8_t *weights);

/**
 * @brief Update neuron internal state.
 * 
 * The update will be done following the neuron model and the spiking state
 * of its connections.
 * 
 * @param neuron Pointer to the neuron to be updated.
 * @param previous_layer Pointer to the previous layer (initial neuron)
 * @return void.
 */
void cortex_neuron_update(  cortex_neuron_t *neuron, 
                            cortex_neuron_t *previous_layer);

/**
 * @brief Charge the neuron.
 * 
 * @param neuron Pointer to the neuron to be updated.
 * @param charge Charge to be applied to the membrane potential.
 * @return void.
 */
void cortex_neuron_charge(  cortex_neuron_t* neuron, 
                            int8_t charge);

/**
 * @brief Leak (discharge) the neuron.
 * 
 * @param neuron Pointer to the neuron to be updated.
 * @return void.
 */
void cortex_neuron_leake(cortex_neuron_t* neuron);

/**
 * @brief Update all neurons of a given layer.
 * 
 * @param layer Pointer to the first neuron of the layer.
 * @param neuron_number Number of neurons.
 * @param input Input layer, from where the spikes might be comming.
 * @return void.
 */
void cortex_layer_update(   cortex_neuron_t *layer, 
                            uint8_t neuron_number, 
						    cortex_neuron_t *input);

/**
 * @brief Update all neurons of a nerve (input) layer.
 * 
 * @param layer Pointer to the first neuron of the layer.
 * @param neuron_number Number of neurons.
 * @param input Input, from where the spikes might be comming.
 * @return void.
 */
void cortex_nerve_layer_update(cortex_neuron_t *layer,
                                uint8_t neuron_number,
                                cortex_neuron_t *input);

/**
 * @brief Reset the spikes of a given layer.
 * 
 * @param layer Pointer to the first neuron of the layer.
 * @param neuron_number Number of neurons.
 * @return void.
 */
void cortex_layer_spike_reset(  cortex_neuron_t *layer, 
                                uint8_t neuron_number);

/**
 * @brief Reset the spikes of a given neuron.
 * 
 * @param neuron Pointer to the neuron to be reset.
 * @return void.
 */
void cortex_neuron_spike_reset(cortex_neuron_t *neuron);

/* end file */
