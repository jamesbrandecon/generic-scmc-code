# -------------------------------------------------------
# Example 2: Monotonicity Constraint
# This is another example of a polynomial regression problem.
# The goal is again to ensure that the learned polynomial has nonnegative derivative over [0,1].
# In this case, the polynomial is multi-dimensional 
# and part of the polynomial is known (Q) and part is unknown (β).
# -------------------------------------------------------

Random.seed!(1234)

# Params
α_true = 1.0
β_true = [0.2, 0.1, 0.1]
Q_true = [1.0  0.1  0.1;
          0.1  1.0  0.1;
          0.1  0.1  1.0]

chain_samples = 1000;
σ_noise = 0.5
N = 1000;
ESS_min_value = chain_samples * 0.5;
proposal_sd_value = 0.1;
lambda_schedule_value = range(0.1, 1e10, length=50);

# Define the outcome function f(x) = α + βᵀ x + ½ xᵀ Q x.
function f(x::Vector{Float64}, α, β, Q)
    return α + dot(β, x) + 0.5 * dot(x, Q * x)
end

# -------------------------------------------------------------------
# Apply SCMC to polynomial regression
# -------------------------------------------------------------------
# Generate data
x_data = [rand(length(β_true)) for i in 1:N]
y_data = [f(x, α_true, β_true, Q_true) + randn()*σ_noise for x in x_data]

# We estimate α and β, while fixing Q = Q_true.
@model function poly_model(x, y, Q)
    α ~ Normal(0, 10)
    β ~ MvNormal(zeros(3), 10 * Matrix{Float64}(I, 3, 3))
    σ ~ InverseGamma(2, 3)
    for i in eachindex(x)
        μ = α + dot(β, x[i]) + 0.5 * dot(x[i], Q * x[i])
        y[i] ~ Normal(μ, σ)
    end
end

function distance_function(theta::Vector{Float64})
    β = theta[2:4]
    xs = range(0, 1, length=10)
    grid = collect(Iterators.product(xs, xs, xs))  
    violations = Float64[]
    for pt in grid
        x = [pt[1], pt[2], pt[3]]
        grad = β .+ Q_true * x
        push!(violations, max(0.0, -minimum(grad)))
    end
    return violations
end

g_identity(theta) = theta

function likelihood(theta::Vector{Float64})
    # Compute predicted y for all x (here you may want to vectorize or pre-store x_data)
    # and then compute the likelihood, e.g.,
    preds = [f(x, theta[1], theta[2:4], Q_true) for x in x_data]
    
    return prod(pdf.(Normal.(preds, theta[end]), y_data))
end

function prior_func(theta::Vector{Float64})
    return pdf(Normal(0,10), theta[1]) * 
        prod(pdf(Normal(0,1), x) for x in theta[2:end-1]) * 
        pdf(InverseGamma(2, 3), theta[end])
end


# Implement the unconstrained model in Turing
model = poly_model(x_data, y_data, Q_true)
chain = sample(model, NUTS(0.65), 2000)

# Implement the constrained model
config = SCMCConfig(
    g_identity,
    prior_func,
    likelihood, 
    distance_function,
    0.01,
    1700,
    lambda_schedule_value
)
@info "Running SCMC on original Turing chain..."
final_particles, final_weights = scmc_algorithm(chain, config; mh_steps=10)

# -------------------------------------------------------
# Compare unconstrained and constrained results
# -------------------------------------------------------
function violation_measure(β::Vector{Float64})
    # For each grid point, compute the gradient: β + Q_true * x.
    # Record the minimum component of that gradient.
    worst = Inf
    ggrid = [ collect(x) for x in Iterators.product(range(0, 1, length=10),
                                                  range(0, 1, length=10),
                                                  range(0, 1, length=10)) ]
    for x in ggrid
        grad = β + Q_true * x
        worst = min(worst, minimum(grad))
    end
    return worst
end

posterior_β = hcat([chain["β[$i]"] for i in 1:3]...)
β_samples = reduce(vcat, [posterior_β[:, :, i] for i in axes(posterior_β, 3)])
violations = [violation_measure(β_samples[i, :]) for i in axes(β_samples, 1)]
println("---------------------------------")
println("Unconstrained Sampling")
println("Share of samples with worst-case derivative < 0 = ", count(v -> v < 0, violations)/length(violations))
println("---------------------------------")

# Now calculate the same for the constrained results 
posterior_β = vcat([final_particles[i].theta[2:4]' for i in 1:length(final_particles)]...)
violations = [violation_measure(posterior_β[i, :]) for i in axes(posterior_β, 1)]
println("---------------------------------")
println("Constrained Sampling")
println("Share of samples with worst-case derivative < 0 = ", count(v -> v < 0, violations)/length(violations))
println("---------------------------------")