"""
Research question. Suppose the only information we had about
a charging station with 1 plug was when it is plugged into and
unplugged from EVs, over (say) 1 year. Can we infer the number
of EVs that missed out on getting charged because it was busy?
"""

using Optim
using QuadGK
using Random
using Convex, SCS
using Plots; pyplot()

#### Global Constants

T = 24   # Period (hours). Weekdays assumed equivalent.
N = 365  # Number of periods to simulate (one year).
D = 3.0  # Charger dead-time, i.e. EV hours-to-charge.
X = [6, 4, 1, -2, 0] / 24
# Each term above integrates on [0,T] to T, so we can sum the x[i]
# to get the average EV arrival rate at the charger over each
# day. Constraint on x: need model(x) to always be positive.
# Estimate collisions via simulation. TODO: check using integration.
MODEL_PERIODS = 3650

#### Functions

function simulate_inhomogeneous_poisson(λ, T)
    """Implements inhomogeneous poisson process
       sampling algorithm from 'Thinning algorithm.pdf'.
       In using an inhomogeneous poisson process to
       model EV arrival density, the model can represent
       variations in arrival frequency with time of day,
       but it assumes there are no 'clusters' of linked
       arrivals (e.g. sharing a common cause, such as a 
       rugby game or a slow truck on the highway.) 
       """
    λmax = -optimize((t)->-λ(t[1]), zeros(1), BFGS()).minimum       
    s = 0.
    sample = Float64[]
    while s < T 
        u = rand()
        w = -log(u) / λmax
        s = s + w
        D = rand()
        if D <= λ(s) / λmax
            push!(sample, s)
        end
    end
    if sample[end] > T
        pop!(sample)
    end
    return sample
end

function filter_blocked_arrivals(arrivals, deadtime)
    """Assume any arrival less than deadtime hours after the last 
       successful arrival is blocked (i.e. charger occupied).
       Modelling assumptions are: 1) One charger per site;
       2) EVs are always plugged in for exactly D hours; 3) EV
       drivers who arrive hoping to use the charger but see that
       it is occupied simply drive away.
    """
    i = 1
    r = Int[]
    while i <= length(arrivals)
        push!(r, i)
        ii = i + 1
        while ii <= length(arrivals) && arrivals[ii] < arrivals[i] + deadtime
            ii += 1
        end
        i = ii
    end
    return arrivals[r]
end

# Arbitrary periodic intensity function λ on (0, T).
model(x) = (t) -> x[1] + x[2] * (1 + sin(1 * 2 * pi * t / T)) +
                         x[3] * (1 + sin(2 * 2 * pi * t / T)) +
                         x[4] * (1 + cos(1 * 2 * pi * t / T)) +
                         x[5] * (1 + cos(2 * 2 * pi * t / T))

function scaled_ignorance(scaling, plug_ins)
    """
    Negative log-likelihood of an inhomogeneous poisson process
    λ(x) is integral(λ(x)) - sum(log(λ(x_i))). We scale each 
    observation to correct for censoring of blocked arrivals.
    """
    return (x) -> N * T * sum(x) - sum(
        s * log(x[1] + x[2] * (1 + sin(1 * 2 * pi * t / T)) +
                       x[3] * (1 + sin(2 * 2 * pi * t / T)) +
                       x[4] * (1 + cos(1 * 2 * pi * t / T)) +
                       x[5] * (1 + cos(2 * 2 * pi * t / T))) 
        for (s, t) in zip(scaling, plug_ins)
        )
end

function s1(x)
    """
    Correction factor that is applied to correct Geiger-Mueller count rate
    to account for dead time (see geiger1.pdf)
    """
    return (t) -> 1 / (1 - D * model(x)(t))
end

function s2(x)
    """
    Improved correction accounting for inhomogeneity of Poisson process.
    """
    return (t) -> begin
        integral, err = quadgk(t -> model(x)(t), t-D, t, rtol=1e-8)
        return 1 / (1 - integral)
    end
end

function visualize(arrivals, plug_ins, λ; λ1=nothing, λ2=nothing)
    """Make a plot"""
    xs = range(0, stop=5*T, length=1000)
    p = scatter(arrivals, λ.(arrivals), label="arrivals", marker=:+)
    scatter!(p, plug_ins, λ.(plug_ins), label="plug_ins", markerstyle="+")
    xs = range(0, stop=5*T, length=10000)
    plot!(p, xs, λ.(xs), label="true density")
    if λ1 != nothing
        plot!(p, xs, λ1.(xs), label="unscaled fit 1")
    end
    if λ2 != nothing
        plot!(p, xs, λ2.(xs), label="rescaled fit 2")
    end
    plot!(p, ylim=(0, ylims(p)[2]), xlim=(0,5*T))
    return p
end

#### Main

λ = model(X)
arrivals = simulate_inhomogeneous_poisson(λ, T*N)
plug_ins = filter_blocked_arrivals(arrivals, D)
n_arrivals = length(arrivals) 
n_plug_ins = length(plug_ins)

p = visualize(arrivals, plug_ins, λ)

println("n_arrivals : ", n_arrivals)
println("n_plug_ins : ", n_plug_ins)
println("arrivals per day : ", n_arrivals/N)
println("plug_ins per day : ", n_plug_ins/N)

# Fit model λ1(t) to data by minimizing ignorance.

x1 = Variable(6)
ign1 = scaled_ignorance(ones(n_plug_ins), plug_ins)
problem1 = minimize(ign1(x1), [x1 >= -10])
solve!(problem1, SCS.Optimizer)
λ1 = model(x1.value)

# Due to blocked events we want to solve a scaled version of the
# problem of minimizing ignorance, incorporating the correction
# formula for Geiger counter dead time: w = 1 / (1 - D * λs(t_i))
# Fit model λ2(t) to data by minimizing scaled ignorance.

rescale1 = [s1(x1.value)(t) for t in plug_ins] # Geiger counter correction
rescale2 = [s2(x1.value)(t) for t in plug_ins] # Retrospective integral of λ
ign2 = scaled_ignorance(rescale2, plug_ins)
x2 = Variable(6)
problem2 = minimize(ign2(x2), [x2 >= -10])
solve!(problem2, SCS.Optimizer)
λ2 = model(x2.value)

# Estimate collisions via simulation. TODO: check using integration.

true_collision_daily = (n_arrivals - n_plug_ins) / N

arrivals_from_model = simulate_inhomogeneous_poisson(λ2, MODEL_PERIODS * T)
plug_ins_from_model = filter_blocked_arrivals(arrivals_from_model, D)
n_arrivals_from_model = length(arrivals_from_model)
n_plug_ins_from_model = length(plug_ins_from_model)
modeled_collision_daily = (n_arrivals_from_model - n_plug_ins_from_model) / MODEL_PERIODS

# Report findings

p = visualize(arrivals, plug_ins, λ; λ1=λ1, λ2=λ2)
plot(p)
savefig(p, "collision_plot.png")

println("----------------------------")
println("true arrivals per day : ", n_arrivals/N)
println("true plug_ins per day : ", n_plug_ins/N)
println("true blocked plug_ins daily: ", true_collision_daily)
println("modelled arrivals per day : ", n_arrivals_from_model / MODEL_PERIODS)
println("modelled plug_ins per day : ", n_plug_ins_from_model / MODEL_PERIODS)
println("modelled blocked plug_ins daily: ", modeled_collision_daily)