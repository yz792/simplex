include("../IterRef.jl")


using ..IterRef

γ(x) = exp(-x^2/2) / sqrt(2*pi)

using Convex,  GLPK, Tulip
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

function from_gaussian(xs, kmax, v = 1;
    ipmits = 100, bits=0)
    n = length(xs)
    P = Variable(n)
    add_constraint!(P, P >= 0)
    add_constraint!(P, P <= 1)

    bits_needed = 2*ceil(Int,(maximum(xs)^2 / 2)/log(2) + kmax*log(maximum(xs))/log(2))

    bits_requested = max(bits,max(256,bits_needed))

    println("bits needed: $(bits_needed), requested : $(bits_requested)")

    setprecision(bits_requested)

    oureps = sqrt(eps(BigFloat))

    xs = BigFloat.(xs)

    # symmetry conditions
    pairs = sym_pairs(xs)
    for (i,j) in pairs
        add_constraint!(P, P[i] == 1-P[j])
    end
    zind = findall(xs .== 0)
    if !isempty(zind)
        add_constraint!(P, P[zind[1]] == 1/2)
    end

    f = γ.(xs)
    f = f / sum(f)

    xsm = xs .- v
    xsp = xs .+ v
    Fp = f .* P
    Fm = f .* (1 .- P)

    t = Variable()
    problem = minimize(t, numeric_type=BigFloat)
    for k in 2:2:kmax
        xspk = xsp.^k
        xsmk = xsm.^k
        momk = sum(xsmk .* Fm) + sum(xspk .* Fp)

        orig_momk = sum(f .* xs.^k)
        #@show typeof(orig_momk)
        problem.constraints += momk >= (1-t)*orig_momk
        problem.constraints += momk <= (1+t)*orig_momk
    end

    solve!(problem, () -> IterRef.Optimizer{BigFloat}(mode=2))
    println(problem.status, " ", Float64.(problem.optval))

    if problem.optval > oureps
        println("Seems Bad: wanted $(Float64.(oureps))")
    else
        println("Is better than eps: $(Float64.(oureps))")
    end

    p = P.value
    fp = f .* p
    fm = f .* (1 .- p)


    old_moms = [sum(f .* xs.^k) for k in 2:2:kmax]
    new_moms = [sum(fm .* xsm.^k) + sum(fp .* xsp.^k) for k in 2:2:kmax]

    return t.value, p, old_moms, new_moms

end

xs = -6:6
kmax = 6
t, p, old, new = from_gaussian(xs, kmax)

# xs = -12:12
# kmax = 6
# t, p, old, new = from_gaussian(xs, kmax)
