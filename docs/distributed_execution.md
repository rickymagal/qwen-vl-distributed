# Distributed Execution Model

The model runs a single logical inference instance across multiple machines
using pipeline parallelism.

Each machine runs exactly one pipeline stage and is responsible for:

- A contiguous range of transformer blocks
- Its local KV cache for those blocks
- Receiving activations from the previous stage
- Producing activations for the next stage

There is no expert-centric sharding, no dynamic load balancing, and no
fault tolerance in this version. Execution is synchronous and deterministic
by design, prioritizing correctness and debuggability.

Network transport is explicit and implemented at the activation boundary.
