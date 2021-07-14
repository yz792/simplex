# kernel_induction.jl

using Statistics, LinearAlgebra, Random
import SpecialFunctions

#=======================

Gaussians

=======================#

"""
γ  (x, σ) is the gaussian measure at x
"""
γ(x::Number, σ) = exp(-x^2 / (2 * σ^2)) / (sqrt(2*π) * σ)
γ(x::Number) = γ(x, 1.0)

"""
cdf(x, σ) is the integral of the Gaussian of standard deviation σ from 0 to x
"""
cdf(x, σ) = SpecialFunctions.erf(x / (σ * sqrt(2))) / 2
cdf(x) = cdf(x, 1.0)

"""
ccdf(x, σ) is the integral of the Gaussian of standard deviation σ from x to ∞
"""
ccdf(x, σ) = 1/2 - cdf(x, σ)
ccdf(x) = 1/2 - cdf(x)

gauss_integral(a, b, σ) = cdf(b, σ) - cdf(a, σ)

#=============================

Linear Constraints / Forms

=============================#

"""
    LinCon(vec, rhs)

Represents rhs - dot(x, vec), and evaluates to this on a vector x.
Corresponding constraint is rhs - dot(x, vec) >= 0, 
or dot(x,vec) <= rhs.   
"""
struct LinCon
    vec::Vector
    rhs::Number
end

function (l::LinCon)(x::AbstractArray)
    @assert length(l.vec) == length(x)
    l.rhs - dot(l.vec,x)
end

#===========================

The basic kernels

Will assume that input xs has symmetry enforced.
Input is always xs, followed by the parameters as a number or tuple.

Note: sometimes we might need to scale these. A natural scaling is to make the rhs 1.
rhs1 does this.

===========================#

rhs1(l::LinCon) = LinCon(l.vec ./ abs(l.rhs), sign(l.rhs))    
rhsup(l::LinCon, x) = LinCon(l.vec, x*l.rhs)


# Monotone Kernels

function right_tail(xs, x::Number)
    @assert x >= 0
    LinCon(xs .> x, ccdf(x))
end

function mgf(xs, λ::Number)
    @assert λ > 0
    LinCon(exp.(λ.*xs), exp(λ^2 / 2) )
end

function moment(xs, m::Number)
    @assert m > 0 && iseven(m)
    LinCon((xs.^m), prod(1:2:m) )
end

# Windowed Kernels

function closed_interval(xs, center, width)
    @assert center >= 0 && width > 0
    lower = center - width
    upper = center + width
    LinCon(lower .<= xs .<= upper, gauss_integral(lower,upper,1.0))
end

function interval(xs, center, width)
    @assert center >= 0 && width > 0
    lower = center - width
    upper = center + width
    LinCon(lower .< xs .< upper, gauss_integral(lower,upper,1.0))
end

"""
    gauss_conv(xs, x, τ)

Convolve with gaussian of variance τ^2, and eval at x.
"""
function gauss_conv(xs, x, τ, fac=1)
    LinCon(γ.(xs .- x, τ), γ(x, fac*sqrt(1+τ^2)) )
end

"""
Convolve with gaussian of variance 1, and eval at t.
Compute the derivative in t, which should be negative for positive t.
"""
function gauss_deriv(xs, t::Number)
    LinCon(γ.(xs .- t) .* (xs .- t), -γ(t, sqrt(2)) * t / 2)
end

function gauss_rat_deriv(xs, t::Number)
    LinCon((2*xs .- t) .* exp.(t^2/4 .- (xs .- t).^2 / 2) / sqrt(2), 0.0)
end

#====================

Decreasing Kernels

=====================#

ratio_con(bigcon, smallcon) =
    LinCon( (smallcon.vec / smallcon.rhs) .- (bigcon.vec / bigcon.rhs), 0)

function decrease_right_tail(xs, xl::Number, xu::Number)
    @assert xl < xu 
    ratio_con( right_tail(xs, xl), right_tail(xs, xu) )
end

function decrease_mgf(xs, λs::Number, λb::Number)
    @assert λs < λb
    ratio_con( mgf(xs, λs), mgf(xs, λb) )
end

function decrease_moment(xs, m1::Number, m2::Number)
    @assert m1 < m2
    ratio_con( moment(xs, m1), moment(xs, m2) )
end

function decrease_interval(xs, x1::Number, x2::Number, width::Number)
    @assert 0.0 <= x1 < x2
    ratio_con( interval(xs, x1, width), interval(xs, x2, width) )
end

function decrease_closed_interval(xs, x1::Number, x2::Number, width::Number)
    @assert 0.0 <= x1 < x2
    ratio_con( closed_interval(xs, x1, width), closed_interval(xs, x2, width) )
end

function decrease_gauss_conv(xs, x1::Number, x2::Number, τ::Number)
    @assert 0.0 <= x1 < x2
    ratio_con( gauss_conv(xs, x1, τ), gauss_conv(xs, x2, τ) )
end

#=====================

Generating decreasing kernel vectors:
given a vector of parameters, generate a vector of the kernel SpecialFunctions

======================#

right_tail(xs, ps::AbstractArray) = 
    [right_tail(xs, p) for p in ps]

mgf(xs, ps::AbstractArray) = 
    [mgf(xs, p) for p in ps]

moment(xs, ps::AbstractArray) = 
    [moment(xs, p) for p in ps]
interval(xs, ps::AbstractArray, width)  = 
    [interval(xs, p, width) for p in ps]

closed_interval(xs, ps::AbstractArray, width) = 
    [closed_interval(xs, p, width) for p in ps]

gauss_conv(xs, ps::AbstractArray, τ, fac=1) = 
    [gauss_conv(xs, p, τ, fac) for p in ps]
gauss_deriv(xs, ts::AbstractArray) = 
    [gauss_deriv(xs, t) for t in ts]
gauss_rat_deriv(xs, ts::AbstractArray) = 
    [gauss_rat_deriv(xs, t) for t in ts]

decrease_right_tail(xs, ps::AbstractArray) =
    [decrease_right_tail(xs, ps[i-1], ps[i]) for i in 2:length(ps)]

decrease_mgf(xs, ps::AbstractArray) =
    [decrease_mgf(xs, ps[i-1], ps[i]) for i in 2:length(ps)]

decrease_moment(xs, ps::AbstractArray) =
    [decrease_moment(xs, ps[i-1], ps[i]) for i in 2:length(ps)]        

decrease_interval(xs, ps::AbstractArray, width::Number) =
    [decrease_interval(xs, ps[i-1], ps[i], width) for i in 2:length(ps)]  

decrease_closed_interval(xs, ps::AbstractArray, width::Number) =
    [decrease_closed_interval(xs, ps[i-1], ps[i], width) for i in 2:length(ps)]  

decrease_gauss_conv(xs, ps::AbstractArray, τ::Number) =
    [decrease_gauss_conv(xs, ps[i-1], ps[i], τ) for i in 2:length(ps)]  

#======================

Symmetrizing

======================#

function sym_pairs(xs)
    xs = round.(xs; sigdigits=8)

    di = Dict()
    pairs = []
    for i in 1:length(xs)
        x = Float64(xs[i])
        if !haskey(di,x)
            di[x] = i
        end
        if haskey(di,-x)
            push!(pairs, (i, di[-x]))
        end
    end
    return pairs
end


#=================================

Optimization Code 

=================================#

xs_plus_v(xs, v) = vcat(xs .+ v, xs .- v)
apply_rule(dist, rule) = vcat(dist .* rule, dist .* (1 .- rule) )














function find_universal_rule(xs, v, dist_cons, rule_cons, maxits; 
    opt_crit = :feas, 
    regularize=2, optimizer=:gurobi, tol=1e-6, distpairs = [])
    iter = 0

    r = []

    if isempty(distpairs)
        d0 = max_variance(xs, dist_cons)
        distpairs = [(d0, k) for k in rule_cons]
    end 

    obj = -1

    while obj < -tol && iter < maxits
        iter += 1
        println("Best...")
        r, robj = best_rule(xs, distpairs; opt_crit, regularize, optimizer)
        if robj < -tol
            println("No good rule")
            break
        end
        println("Worst...")
            dps, obj = worst_dists_for(xs, v, r, dist_cons, rule_cons, 
                verbosity = 1)
        append!(distpairs, dps)

    end

    return r, distpairs
end

function find_universal_rule0(xs, v, dist_cons, rule_cons, maxits; 
    regularize=2, optimizer=:gurobi, tol=1e-6, dists = [])
    iter = 0

    r = []

    if isempty(dists)
        d0 = max_variance(xs, dist_cons)
        dists = [d0]
    end 

    while obj < -tol && iter < maxits
        iter += 1
        println("Best...")
        r, robj = best_rule(xs, v, dists, rule_cons; regularize=regularize, optimizer=:tulip)
        if robj < -tols
            println("No good rule")
            break
        end
        println("Worst...")
        dist, obj = worst_dist(xs, v, r, dist_cons, rule_cons, 
            verbosity = 1, optimizer=:tulip)
        push!(dists, dist)

        @show 1, sum(abs.(diff(r)))
    end

    return r, dists
end