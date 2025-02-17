using Pkg
Pkg.activate(temp=true)
# installing packages some might not have installed
Pkg.add([
    "Random",
    "Distributions",
    "Statistics",
    "LinearAlgebra",
    "StatsPlots",
    "MCMCChains",
    "Turing",
    "Printf"
])
using Random, Distributions, Statistics, LinearAlgebra
using Turing, MCMCChains, StatsPlots
using Printf

# ----------------------------
# Structs
# ----------------------------

# Define a Particle type.
struct Particle
    theta::Vector{Float64}    # Raw parameter vector.
    gtheta::Vector{Float64}   # Transformed parameter vector (here, same as theta).
end

# Define a configuration type for SCMC.
struct SCMCConfig
    g::Function              # Reparameterization (here identity).
    prior::Function          # Prior density evaluator.
    likelihood::Function     # Likelihood density evaluator.
    distance::Function       # Distance function (computes violations).
    proposal_sd::Float64     # Standard deviation for the MH proposal.
    ESS_min::Float64         # Minimum effective sample size threshold.
    lambda_schedule::Vector{Float64}  # Sequence of lambda values.
end

# ----------------------------
# Extracting and Initializing Particles
# ----------------------------

"""
    extract_particles(chain::Chains)

Extracts a matrix of samples from a Turing chain. Each row is a sample.
Assumes parameters in param_names are present in the chain.
"""
function extract_particles(chain::Chains)
    # names = ["a[1]", "a[2]", "a[3]", "a[4]", "a[5]"]
    extra_names = ["lp",
    "n_steps",
    "is_accept",
    "acceptance_rate",
    "log_density",
    "hamiltonian_energy",
    "hamiltonian_energy_error",
    "max_hamiltonian_energy_error",
    "tree_depth",
    "numerical_error",
    "step_size",
    "nom_step_size"]
    return hcat([chain[x] for x in names(chain) if string(x) ∉ extra_names]...)
end

"""
    initialize_particles_from_chain(chain, config)

Extracts particles from the Turing chain and applies the reparameterization.
Returns a vector of Particle and uniform weights.
"""
function initialize_particles_from_chain(chain::Chains, config::Union{SCMCConfig, Nothing} = nothing)
    if config == nothing
        config = SCMCConfig(x->x, x->1.0, x->1.0, x->0.0, 0.1, 1000.0, [0.0, 1.0])
    end
    raw_particles = extract_particles(chain)
    N, _ = size(raw_particles)
    particles = [Particle(vec(raw_particles[i, :]), config.g(vec(raw_particles[i, :]))) for i in 1:N]
    weights = fill(1.0 / N, N)
    return particles, weights
end


# ----------------------------
# SCMC Algorithm Components
# ----------------------------

"""
    penalty_smc(theta, penalty, problem)

Computes the penalty multiplier using the normal CDF.
For each component d of the distance vector, compute
    cdf(Normal(0,1), -2 * penalty * d)
and return the product.
If the violation is negligible, return 1.
"""
function penalty_smc(theta::Vector{Float64}, penalty::Real, distance_function::Function)::Real
    d = distance_function(theta)
    multiplier = prod(cdf.(Normal(0,1), -2 * penalty .* d))
    return multiplier
end

"""
    resample_particles(particles, weights)

Resamples the particles using systematic resampling and resets weights to uniform.
"""
function resample_particles(particles::Vector{Particle}, weights::Vector{Float64})
    N = length(particles)
    positions = ((0:N-1) .+ rand()) ./ N
    cumulative_sum = cumsum(weights)
    new_particles = Vector{Particle}(undef, N)
    i, j = 1, 1
    while i <= N && j <= N
        if positions[i] < cumulative_sum[j]
            new_particles[i] = particles[j]
            i += 1
        else
            j += 1
        end
    end
    new_weights = fill(1.0 / N, N)
    return new_particles, new_weights
end

"""
    mh_transition(particles, config, target_density)

Performs one Metropolis–Hastings move per particle using a Gaussian random walk proposal.
"""
function mh_transition(particles::Vector{Particle}, config::SCMCConfig, target_density::Function)
    N = length(particles)
    new_particles = similar(particles)
    acceptances = zeros(N)
    for i in 1:N
        current_theta = particles[i].theta
        proposal = current_theta .+ randn(length(current_theta)) .* config.proposal_sd
        current_density = target_density(current_theta)
        proposal_density = target_density(proposal)
        α = min(1.0, proposal_density / (current_density))
        if rand() < α
            new_particles[i] = Particle(proposal, config.g(proposal))
            acceptances[i] = 1
        else
            new_particles[i] = particles[i]
            acceptances[i] = 0
        end
    end
    @info @sprintf("MH Acceptance Rate: %.2f", mean(acceptances))
    return new_particles
end

"""
    compute_weights!(particles, weights, config, lambda_m, lambda_prev, problem)

Updates the weights in-place. We use the ratio of penalty multipliers computed
by penalty_smc at lambda_m and lambda_prev.
"""
function compute_weights!(particles::Vector{Particle}, weights::Vector{Float64},
                          config::SCMCConfig, lambda_m::Float64, lambda_prev::Float64)::Nothing
    N = length(particles)
    for i in 1:N
        multiplier_current = penalty_smc(particles[i].gtheta, lambda_m, config.distance)
        multiplier_prev    = penalty_smc(particles[i].gtheta, lambda_prev, config.distance)
        factor = multiplier_current / multiplier_prev
        weights[i] *= factor
    end
    total = sum(weights)
    # weights ./= total
    for i in 1:N
        weights[i] /= total
    end
    return nothing
end

# ----------------------------
# Functions for adaptive lambda selection
# ----------------------------

# Helper function to compute ESS from a given candidate lambda without modifying the current weights
function candidate_ess(particles::Vector{Particle}, orig_weights::Vector{Float64},
    config::SCMCConfig, lambda_candidate::Float64, lambda_prev::Float64)
    # Make a copy of the weights so we don't disturb the original
    new_weights = copy(orig_weights)
    compute_weights!(particles, new_weights, config, lambda_candidate, lambda_prev)
    return 1 / sum(w^2 for w in new_weights)
end

# Helper function that uses bisection to solve for the next lambda such that ESS equals config.ESS_min.
# If even at the maximum allowed lambda the ESS is above the target, then the maximum is returned.
function find_lambda(particles::Vector{Particle}, weights::Vector{Float64},
  config::SCMCConfig, lambda_prev::Float64)
    # Define the target function: f(lambda) = candidate_ess(...) - ESS_min
    f(λ) = candidate_ess(particles, weights, config, λ, lambda_prev) - config.ESS_min

    # Set search bounds: lower bound is the current lambda, and upper bound is the maximum allowed.
    lower = lambda_prev
    upper = config.lambda_schedule[end]  # assume the schedule's final value is our maximum

    # Check if even the maximum lambda does not drop ESS to or below ESS_min.
    if f(upper) > 0
        # Cannot lower ESS enough; return the maximum lambda.
        @warn  "Maximum lambda reached; ESS is still above the threshold."
        return upper
    end

    # Bisection parameters
    tol = 1e-3
    max_iter = 100
    iter = 0
    while (upper - lower > tol) && (iter < max_iter)
        mid = (lower + upper) / 2
        if f(mid) > 0
            lower = mid
        else
            upper = mid
        end
        iter += 1
    end
    return (lower + upper) / 2
end

# ----------------------------
# Full SCMC algorithm that uses the adaptive lambda selection.
# ----------------------------

function scmc_algorithm(chain::Chains, config::SCMCConfig; mh_steps::Int=1)
    # Extract particles and set uniform weights.
    particles, weights = initialize_particles_from_chain(chain, config)
    lambda_prev = 0.0
    M = length(config.lambda_schedule)  # maximum iterations (or maximum lambda)

    for m in 1:M
    # Solve for the next lambda that makes ESS equal to the threshold.
    # lambda_prev + 0.11; #
    lambda_m = find_lambda(particles, weights, config, lambda_prev)

    # Update weights using the penalty ratio.
    compute_weights!(particles, weights, config, lambda_m, lambda_prev)
    
    ess = 1 / sum(w^2 for w in weights)
    @info @sprintf("SCMC Step %d: lambda = %.4f, ESS = %.2f", m, lambda_m, ess)

    # Always resample.
    particles, weights = resample_particles(particles, weights)

    # Define the target density at the current lambda.
    target_density(theta) = config.likelihood(theta) * config.prior(theta) * penalty_smc(config.g(theta), lambda_m, config.distance)

    # Apply multiple MH moves.
    for _ in 1:mh_steps
        particles .= mh_transition(particles, config, target_density);
    end

    lambda_prev = lambda_m  # update for the next iteration

    # Exit early if lambda has reached the maximum value.
    if lambda_m == config.lambda_schedule[end]
        break
    end
    end

    return particles, weights
end