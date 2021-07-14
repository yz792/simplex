using Convex, DoubleFloats,Tulip,JuMP,GLPK
# include("../IterRef.jl")
# using ..IterRef
include("kernel_induction.jl")


# hard code

xs = -6:0.5:6

#cs = 0.0:0.1:9
cs = 0.0:0.5:8

dist_cons = gauss_conv(xs, collect(cs), 1.0)

NT = BigFloat

ONE = one(NT)
Z = zero(NT)

n = length(xs)
D1 = Variable(n)
T = Variable()

A1 = zeros(42,26)
A2 = zeros(13,26)
b1 = zeros(42)
b2 = zeros(13)

# cons = D1 >= T
# cons += sum(D1) == ONE


for i = 1:25
    tmp = zeros(26)
    tmp[i] = -1
    tmp[26] = 1
    A1[i,:] = tmp
    b1[i] = 0
end

cnt = 0
cnt = 26



for k in dist_cons
    global cnt
    tmp = [k.vec; 1]

    println(maximum(tmp))
    println(minimum(abs.(tmp)))
    #print(tmp)
    A1[cnt,:] = tmp
    b1[cnt] = k.rhs
    cnt += 1
    #cons += k.rhs - dot(k.vec, D1) >= T
end

pairs = sym_pairs(xs)
for i = 1:12
    tmp = zeros(26)
    tmp[i] = 1
    tmp[26-i] = -1
    #print(tmp)
    A2[i,:] = tmp
    b2[i] = 0
    #cons += D1[i] == D1[j]

end

A2[13,:] = [ones(25);0]
b2[13] = 1

m = 55
n = 26

# opt_primal =  Model()
# set_optimizer(opt_primal, GLPK.Optimizer)
# @variable(opt_primal, x[1:n])
# @objective(opt_primal, Max, x[26] )
# @constraint(opt_primal, constraint1, A1 * x .<= b1)
# @constraint(opt_primal, constraint2, A2 * x .== b2)
# JuMP.optimize!(opt_primal)
# x_prime = JuMP.value.(x)
#
# y_1 = JuMP.dual.(constraint1)
# y_2 = JuMP.dual.(constraint2)
#
# y_prime = [y_1;y_2]
# A = [A1; A2]
# b = [b1;b2]
#
#
# @show x_prime
# @show y_1
# @show y_2
# @show length(y_prime[y_prime .!= 0])
# @show length(x_prime[x_prime .!= 0])
# @show x_prime[26]

A = [A1; A2]
b = [b1;b2]
c = [zeros(25);1]


opt_dual =  Model()
set_optimizer(opt_dual, GLPK.Optimizer)
@variable(opt_dual, y[1:m])
@objective(opt_dual, Min, dot(b,y))
@constraint(opt_dual, constraint1, A' * y .== c)
@constraint(opt_dual, hehe, y[1:42].>=0)
JuMP.optimize!(opt_dual)
y_prime = JuMP.value.(y)

# y_1 = JuMP.dual.(constraint1)
# y_2 = JuMP.dual.(constraint2)
#
# y_prime = [y_1;y_2]

# @show A


#
# @show x_prime
# @show y_1
# @show y_2
# @show length(y_prime[y_prime .!= 0])
# @show length(x_prime[x_prime .!= 0])
# @show x_prime[26]




function findnonzeroidx(ls)
    rt = Int[]
    for i = 1:length(ls)
        if ls[i] != 0
            push!(rt,i)
        end
    end
    return rt
end


#idx1 = findnonzeroidx(x_prime)
# idx2 = findnonzeroidx(y_prime)

idx2 = sortperm(abs.(y_prime),rev=true)[1:24]


# @show length(idx2)

T = BigFloat

# idx1 = (1:26)[x_prime .!= 0]
# idx2 = (1:55)[y_prime .!= 0]
b_prime = b[idx2]
b_prime = T.(b_prime)

A_prime = A[idx2,:]
A_prime = T.(A_prime)


z = A_prime\b_prime

@show (sum(z[1:25]))

@show maximum(abs.(A2*z-b2))
@show maximum(A1*z-b1)
# @show minimum(abs.(y_prime[idx2]))
#
# @show z[26]
