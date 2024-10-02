# Non-Structured SNN -> NSSNN -> N-SN2 (temporal name)

This project aims to develop brain-like computing capabilities on the CPU of an embedded system. The core of the project is an **unstructured Spiking Neural Network (SNN)** that will consider the **sparsity** of synaptic connections to avoid unnecessary computations on connections with very low weight. This approach mimics how biological neural networks optimize resource usage. The final goal of this project is to research and develop a **cortical column model**.

## Key Features
- **Unstructured SNN Core**: The main component of the project is an unstructured SNN that models the complexity of brain-like networks.
- **Sparsity Handling**: Low-weight connections are not computed to improve computational efficiency, following a sparse connectivity paradigm.
- **Layer Connectivity**: This version supports inter-layer connectivity across different depths, extending beyond typical layer-by-layer structured SNNs.
- **Embeddable**: Designed for deployment on embedded systems with resource constraints.

## V1
The current implementation is a **structured, layer-by-layer SNN**. This version computes all weights and neuron connections without taking sparsity into account. 

## Future Goals
- Transition from the structured model to the **unstructured SNN** architecture.
- Refine the SNN to incorporate sparsity-based optimizations.
- Explore the behavior of a simulated **cortical column**, leveraging the unstructured network architecture.

# License

This project is licensed under the GNU GPLv3 License. See the https://www.gnu.org/licenses/gpl-3.0.en.html file for details.
