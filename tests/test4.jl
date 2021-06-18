include("../IterRef.jl")
include("inv_dist_lp.jl")
using ..IterRef
using Convex
using LinearAlgebra
using Tulip
using Clp
using Gurobi
using JuMP


A, b, c = lp2(20)
T = BigFloat

n = length(c)


# Convex as wrapper, dual of constraint is available under mode 0.
T = BigFloat
X = Variable(length(c), Positive())
problem = minimize(dot(c,X), numeric_type = T)
problem.constraints += A*X <= b
@time solve!(problem, () -> IterRef.Optimizer{T}(verbose=true,mode=0))
@show problem.status, problem.optval
@show evaluate(X), length(evaluate(X))
@show problem.constraints[1].dual,length(problem.constraints[1].dual)



# JuMP seem only work for Float64 precision, which might not be useful under our scenario, but anyway it can run.
opt_primal =  Model()
set_optimizer(opt_primal, IterRef.Optimizer{Float64})
@variable(opt_primal, x[1:n])
@objective(opt_primal, Min, dot(c,x) )
@constraint(opt_primal, constraint, A * x .<= b)
@constraint(opt_primal, lower_bound, x.>=zeros(n) )
JuMP.optimize!(opt_primal)
@show objective_value(opt_primal)
@show JuMP.value.(x)
@show JuMP.dual.(constraint)
