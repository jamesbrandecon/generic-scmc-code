# Sequentially Constrained Monte Carlo

This repo includes some simple functions to implement Sequentially Constrained Monte Carlo (SCMC). SCMC, as implemented here, is meant to enforce generic constraints when estimating flexible functions via  MCMC. The general idea is to produce an unconstrained (or imperfectly constrained) posterior using standard MCMC, and then to run an iterative procedure that slowly pushes the posterior draws toward the space satisfying constraints. 

## Code contents/Usage
There are two main constructions here. One is the `Particle` structure, which stores the parameter vector and (optionally) a transformed value which can be an input to the likelihood. This was useful for my own testing but may not be for you, in which case you can always set `theta == gtheta`.
```julia
struct Particle
    theta::Vector{Float64}    # Raw parameter vector.
    gtheta::Vector{Float64}   # Transformed parameter vector (here, same as theta).
end
```

The other is `SCMCConfig`, which stores the details necessary to run SCMC. Most are standard/self-explanatory, but the two example files should clarify anything else.
```julia
    struct SCMCConfig
        g::Function              # Reparameterization (set this to x->x as default).
        prior::Function          # Prior density evaluator.
        likelihood::Function     # Likelihood density evaluator.
        distance::Function       # Distance function (computes constraint violations).
        proposal_sd::Float64     # Standard deviation for the MH proposal.
        ESS_min::Float64         # Minimum effective sample size threshold 
        lambda_schedule::Vector{Float64}  # Sequence of penalty values.
    end
```

To run SCMC, my hope is that you only have to define this config, generate an unconstrained posterior using off-the-shelf tools like Turing.jl, and then use the main algorithm function to enforce constraints:
```julia
scmc_algorithm(chain::Chains, config::SCMCConfig; mh_steps::Int=1)
```