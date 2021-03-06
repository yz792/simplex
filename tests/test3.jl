include("../IterRef.jl")
include("inv_dist_lp.jl")
using ..IterRef
using Convex
using LinearAlgebra
using Tulip
using Clp
using Gurobi



A, b, c = lp0(3)

# A almost singular
# The following LPs should output same result (except sign)



# guessing index
T = BigFloat
X = Variable(length(c), Positive())
problem = minimize(dot(c,X), numeric_type = T)
problem.constraints += A*X >= b
@time solve!(problem, () -> IterRef.Optimizer{T}(verbose=true,mode=1))
@show problem.status, problem.optval

# iterative LP
T = BigFloat
X = Variable(length(c), Positive())
problem = minimize(dot(c,X), numeric_type = T)
problem.constraints += A*X >= b
@time solve!(problem, () -> IterRef.Optimizer{T}(verbose=true,mode=0))
@show problem.status, problem.optval

# unbounded
T = BigFloat
X = Variable(length(c), Positive())
problem = maximize(dot(c,X), numeric_type = T)
problem.constraints += A*X >= b
@time solve!(problem, () -> IterRef.Optimizer{T}(verbose=true,mode=1))
@show problem.status, problem.optval

# would output feasible, which is an error, should be unbounded. (Due to use -1e5 to replace -Inf. Will fix this bug in the future)
A,b,c = -A,-b,-c
T = BigFloat
X = Variable(length(c), Positive())
problem = minimize(dot(c,X), numeric_type = T)
problem.constraints += A*X <= b
@time solve!(problem, () -> IterRef.Optimizer{T}(verbose=true,mode=1))
@show problem.status, problem.optval
