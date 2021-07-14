include("../IterRef.jl")
include("inv_dist_lp.jl")
using ..IterRef
using Convex
using LinearAlgebra
using Tulip
using Clp
using Gurobi


A, b, c = lp2(2)
T = BigFloat

@show A
@show b
@show c

# A not sigular
# The following LPs should output same result (except sign)



# default optimizer = GLPK
# T = BigFloat
# X = Variable(length(c), Positive())
# problem = minimize(dot(c,X)+10, numeric_type = T)
# problem.constraints += A*X <= b
# @time solve!(problem, () -> IterRef.Optimizer{T}(verbose=true,mode=0))
# @show problem.status, problem.optval
#
# T = BigFloat
# X = Variable(length(c), Positive())
# problem = minimize(dot(c,X)+10, numeric_type = T)
# problem.constraints += A*X <= b
# @time solve!(problem, () -> Tulip.Optimizer{T}())
# @show problem.status, problem.optval


# upper bound variable
# X = Variable(length(c), Negative())
# problem = maximize(dot(c,X)+5, numeric_type = T)
# problem.constraints += A*X >= -b
# @time solve!(problem, () -> IterRef.Optimizer{T}(verbose=true,mode=1))
# @show problem.status, problem.optval
#
# X = Variable(length(c), Negative())
# problem = maximize(dot(c,X)+5, numeric_type = T)
# problem.constraints += A*X >= -b
# @time solve!(problem, () -> Tulip.Optimizer{T}())
# @show problem.status, problem.optval
# #
#
# # costumized optimizer = Gurobi
# T = BigFloat
# X = Variable(length(c), Positive())
# problem = minimize(dot(c,X), numeric_type = T)
# problem.constraints += A*X <= b
# @time solve!(problem, () -> IterRef.Optimizer{T}(verbose=true,mode=1,inner=Gurobi.Optimizer))
# @show problem.status, problem.optval
#
# # costumized optimizer = Gurobi
# T = BigFloat
# X = Variable(length(c), Positive())
# problem = minimize(dot(c,X), numeric_type = T)
# problem.constraints += A*X <= b
# @time solve!(problem, () -> IterRef.Optimizer{T}(verbose=true,mode=0))
# @show problem.status, problem.optval
#
# # Negative
X = Variable(length(c), Negative())
problem = maximize(dot(c,X)+sum(c)*1000, numeric_type = T)
problem.constraints += A*(X+1000) >= -b
problem.constraints += X <= -1000
@time solve!(problem, () -> IterRef.Optimizer{T}(verbose=true,mode=0))
@show problem.status, problem.optval
#
# # # Negative
# X = Variable(length(c), Negative())
# problem = maximize(dot(c,X)+sum(c)*1000, numeric_type = T)
# problem.constraints += A*(X+1000) >= -b
# problem.constraints += X <= -1000
# @time solve!(problem, () -> Tulip.Optimizer{T}())
# @show problem.status, problem.optval
#
# # we need X >> -1e5, otherwise would break. (Due to use -1e5 to replace -Inf. Will fix this bug in the future)
# X = Variable(length(c), Negative())
# problem = maximize(dot(c,X)+sum(c)*-1e7, numeric_type = T)
# problem.constraints += A*(X+1e7) >= -b
# problem.constraints += X <= -1e-7
# @time solve!(problem, () -> IterRef.Optimizer{T}(verbose=true,mode=1))
# @show problem.status, problem.optval
#
# X = Variable(length(c), Negative())
# problem = maximize(dot(c,X)+sum(c)*-1e7, numeric_type = T)
# problem.constraints += A*(X+1e7) >= -b
# problem.constraints += X <= -1e-7
# @time solve!(problem, () -> Tulip.Optimizer{T}())
# @show problem.status, problem.optval
