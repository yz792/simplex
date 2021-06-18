using LinearAlgebra
using SparseArrays
#=

inv_dist_lp.jl

Generates an lp to test Dan's invariant distributions conjecture.
The main routines will return A, b, c so that the LP is
min dot(x,c)
s.t. Ax = b
x >= 0

=#

function lp2(k,T=Float64)
    A = T.(randn(4*k,k))
    b = ones(T,4*k)*(k)
    c = -ones(T,k)
    return A,b,c
end



function make_line(x, y, r)
    @assert y^2 < r^2-1
    i_min = -floor(Int, sqrt(r^2 - y^2) + x)
    i_max = floor(Int, sqrt(r^2 - y^2) - x)

    z = [sqrt(y^2 + (i+x)^2) for i in i_min:i_max]
    return z
end


function make_lines(xs::Vector, ys::Vector, r)
    return vec([make_line(x,y,r) for x in xs, y in ys])
end


hashfloat(x) = round(Int, Float64(x)*1e8)




"""
    A, b, c = lp0(r::Real; T=Float64)

"""
function lp0(r::Real; T=Float64)
    ys = sqrt.(collect(0:0.5:(r^2.0 - 1.5)))
    xs = unique(mod.(ys,1))
    lines = make_lines(xs, ys, r)

    return lp0(lines; T)
end

"""
    A, b, c = lp0(r::Real, k; T=Float64)

"""
function lp0(r::Real, k; T=Float64)
    ys = sqrt.(collect(0:0.5:(r^2.0 - 1.5)))
    k = min(k,length(ys))
    xs = unique(mod.(ys[1:k],1))
    lines = make_lines(xs, ys, r)

    return lp0(lines; T)
end



"""
    A, b, c = lp0(lines::Vector)

Return program min c'x st. Ax >= b x >= 0.
"""
function lp0(lines::Vector; T=Float64)

    z_flat = vcat(lines...)
    z = unique(sort(z_flat))
    nvars = length(z)

    z_dict = Dict()
    for i in 1:nvars
        z_dict[hashfloat(z[i])] = i
    end
    z_to_ind(x) = z_dict[hashfloat(x)]

    c = exp.(T.(z).^2 ./ 4) .* [0;diff(z)]

    # begin by making constraints as tuples: vector a_i and real b_i
    cons = []
    push!(cons, ([0;diff(z)], 1/T(2)))

    # force function to be decreasing
    for i in 2:nvars
        v = zeros(nvars)
        v[i-1] = one(T)
        v[i] = -one(T)
        push!(cons, (v, zero(T)))
    end

    # finish by putting in the constraints for the lines.

    for line in lines
        v = zeros(nvars)
        v[z_to_ind(line[1])] = one(T)
        #push!(cons, (copy(v),0))
        for i in 2:length(line)
            v = -v
            v[z_to_ind(line[i])] += one(T)
            push!(cons, (copy(v), zero(T)))
        end
    end

    # now construct A and b
    # we will make them dense for now, because that is simplest.
    # for big programs, they should be made sparse

    ncons = length(cons)

    A = zeros(T, ncons, nvars)
    b = zeros(T, ncons)

    for i in 1:ncons
        A[i,:] = cons[i][1]
        b[i] = cons[i][2]
    end

    A = sparse(A)

    return A, b, c

end

"""
    As, bs, cs = standard_form(A, b, c)

Given A, b, c output from lp0, construct the lp of form
min cs'x s.t. As*x = bs
"""
function standard_form(A, b, c)
    ncons, nvars = size(A)
    As = [A I]
    cs = vcat(c, zeros(ncons))
    ls = zeros(length(cs))
    return As, b, cs,ls
end

function standard_form2(A, b, c)
    ncons, nvars = size(A)
    As = [A -I]
    cs = vcat(c, zeros(ncons))
    ls = zeros(length(cs))
    return As, b, cs,ls
end
