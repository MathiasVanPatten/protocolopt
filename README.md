# Roadmap
Not binding, the further out the less binding as I find issues through the process.

## MVPs
- [x] MVP Bitflip Training, showing the ability to replicate the basic operation on 1 bit
- [X] First QoL and Cleanup
    - [X] Add callbacks to the simulation class
    - [X] Change plotting to be number of spatial dimensions agnostic and moved into callbacks
    - [X] Add ability to constrain MCMC chains to specific bits of the potential to guaruntee a good starting state
    - [X] Switch Experiment Tracking to Aim
- [X] MVP Bit Erasure, showing the ability to replicate the basic operation on 1 bit
- [ ] Replication Revisit
    - [ ] Ensure the work is as expected compared to original code for bitflip and erasure
        - [ ] Bitflip
        - [ ] Bit Erasure
    - [ ] ensure the shape of the potential in time is as expected compared to original code for bitflip and erasure
        - [ ] Bitflip
        - [ ] Bit Erasure
- [ ] Second QoL and Cleanup
    - [ ] Generalize training script, moving the more basic training tasks into a yaml-led process
    - [ ] Do an api pass, making sure everything makes sense and is as easy as possible with error checking to make sure the user is doing the right thing
    - [ ] Optimization pass
    - [ ] Do a documentation pass, docstrings, typehints, comments, etc.
    - [ ] Add ability to save and load model
    - [ ] Add ability to generate, simulate, and plot a generated ensemble given the bits you want to infer on to make it easy to examine a trained model

## Some stretch goals
- [ ] MVP NAND, showing 2 bits
    - [ ] Loose (01 or 10 are 1)
    - [ ] Strict (11 is only 1)
- [ ] MVP 2 bit adder, showing 4 bits (01 + 01 = 10 00 as example)
- [ ] MVP 4 bit adder, showing 8 bits (0101 + 0101 = 1010 0000 as example)
- [ ] MVP 8 bit adder, showing 16 bits (01010101 + 01010101 = 10101010 00000000 as example)
- [ ] Show different potential model
    - [ ] Spline
    - [ ] Multi-layer Perceptron

    

