# -------------------------------------------------------------------
# Example: Polynomial regression
# This example demonstrates how to apply the generic SCMC code to a polynomial regression problem.
# The goal is to ensure that the learned polynomial has nonnegative derivative over [0,1].
# -------------------------------------------------------------------
Random.seed!(123929)

# Params
proposal_sd_value = 0.05
n_samples = 5000;
ESS_min_value = 4900.0;
lambda_schedule_value = collect(range(0, stop=50, length=50))

# True coefficients (chosen so that the function has nonnegative derivative over [0,1])
true_coeffs = [0.5, 1.0, 0.5, 0.2, 0.1]

# Define reparameterization function -- here the identity function
g_identity(theta) = theta

# Define a 4th-order polynomial.
function poly(x, coeffs)
    return coeffs[1] .+ coeffs[2]*x .+ coeffs[3]*x.^2 .+ coeffs[4]*x.^3 .+ coeffs[5]*x.^4
end

# Define a Turing model for 4th order polynomial regression.
@model function poly_model(x, y) 
    a ~ filldist(Normal(0, 1), 5)
    σ ~ InverseGamma(2, 3)
    for i in eachindex(x)
        mu = poly(x[i], a); #a[1] + a[2]*x[i] + a[3]*x[i]^2 + a[4]*x[i]^3 + a[5]*x[i]^4
        y[i] ~ Normal(mu, σ)
    end
end

# Define prior 
theta_prior_dist = Normal(0, 1)
function prior_func(theta::Vector{Float64})
    return prod(pdf(theta_prior_dist, x) for x in theta[1:end-1]) * # polynomial coefficients
              pdf(InverseGamma(2, 3), theta[end])             # noise variance
end

# Define likelihood
function likelihood(theta::Vector{Float64})
    preds = poly(x_data, theta[1:5])
    return sum(logpdf.(Normal.(preds, theta[end]), y_data))
end

function distance_function(theta::Vector{Float64})
    grid = range(0, 1, length=1000)
    deriv = theta[2] .+ 2 * theta[3] .* grid .+ 3 * theta[4] .* grid.^2 .+ 4 * theta[5] .* grid.^3
    violations = abs.(min.(deriv, 0.0))
    return violations
end

# -------------------------------------------------------------------
# Applying SCMC to polynomial regression
# -------------------------------------------------------------------
## ----------------------------
# Generate data
## ----------------------------
N = 1000
x_data = sort(rand(N))
σ_noise = 3.1
y_data = poly(x_data, true_coeffs) .+ randn(N) .* σ_noise

## ----------------------------
# Unconstrained NUTS
## ----------------------------
model = poly_model(x_data, y_data)
chain = sample(model, NUTS(), n_samples)

## ----------------------------
# Constrained via SCMC
## ----------------------------
# Construct the configuration.
config = SCMCConfig(
    g_identity,
    prior_func,
    likelihood,
    distance_function,
    proposal_sd_value,
    ESS_min_value,
    lambda_schedule_value
)

@info "Running SCMC on Turing chain..."
final_particles, final_weights = scmc_algorithm(chain, config, mh_steps=5);

# -------------------------------------------------------------------
# Summary and Plotting
# -------------------------------------------------------------------
function satisfies_constraint(p::Particle; tol=1e-3)
    grid = range(0, 1, length=1000)
    deriv = p.theta[2] .+ 2*p.theta[3] .* grid .+ 3*p.theta[4] .* grid.^2 .+ 4*p.theta[5] .* grid.^3
    return minimum(deriv) ≥ -tol
end

function raw_derivatives(p::Particle)
    grid = range(0, 1, length=1000)
    deriv = p.theta[2] .+ 2*p.theta[3] .* grid .+ 3*p.theta[4] .* grid.^2 .+ 4*p.theta[5] .* grid.^3
    return deriv
end

init_particles = initialize_particles_from_chain(chain, config)[1]

n_sat_init = count(satisfies_constraint, init_particles)
n_sat = count(satisfies_constraint, final_particles)

@info "Initial particles satisfying the constraint: $n_sat_init out of $(length(init_particles))"
@info "Particles satisfying the nonnegative derivative constraint: $n_sat out of $(length(final_particles))"

# Posterior mean polynomial vs. truth
true_derivatives = raw_derivatives(Particle(true_coeffs, true_coeffs))
initial_derivatives = mean([raw_derivatives(init_particles[i]) for i in 1:length(init_particles)])
derivatives = mean([raw_derivatives(final_particles[i]) for i in 1:length(final_particles)])
histogram(true_derivatives, label="True", alpha = 0.3)
histogram!(initial_derivatives, label="Before", alpha = 0.3)
histogram!(derivatives, label="After", alpha = 0.3)
xlabel!("Posterior Mean Derivatives")
ylabel!("Frequency")

# Plot implied derivatives
plot(true_derivatives, initial_derivatives, label="True vs Initial", ylabel="Initial dy/dx", xlabel="True dy/dx", title="True vs Initial Derivatives")
plot!(true_derivatives, derivatives, label="True vs Final")
plot!(x->x, minimum(true_derivatives), maximum(true_derivatives), label="y=x", lw=3)

# Plot the worst particles from before and after SMC
# measured by distance function
worst_init_particle = argmax([maximum(distance_function(p.theta)) for p in init_particles])
worst_particle = argmax([maximum(distance_function(p.theta)) for p in final_particles])
x_grid = range(0, 1, length=1000)
y_worst_init = poly(x_grid, init_particles[worst_init_particle].theta)
y_worst = poly(x_grid, final_particles[worst_particle].theta)
y_true = poly(x_grid, true_coeffs)
plot(x_grid, y_true, label="True Polynomial", lw=3)
plot!(x_grid, y_worst_init, label="Worst Unconstrained Particle", lw=3)
plot!(x_grid, y_worst, label="Worst Constrained Particle", lw=3)
xlabel!("x")
ylabel!("y")

