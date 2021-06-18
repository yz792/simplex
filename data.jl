"""
    IRData{T, Tv, Ta}

Holds data about an interior point method.

The problem is represented as
```
min   c'x + c0
s.t.  A x = b
      l ≤ x
```
where `l`, `u` may take infinite values.
"""
struct IRData{T, Tv, Tb, Ta}

    # Problem size
    nrow::Int
    ncol::Int

    # Objective
    objsense::Bool  # min (true) or max (false)
    c0::T
    c::Tv

    # Constraint matrix
    A::Ta

    # RHS
    b::Tv

    # Variable bounds (may contain infinite values)
    l::Tv
    # Variable bound flags (we template with `Tb` to ease GPU support)
    # These should be vectors of the same type as `l`, `u`, but `Bool` eltype.
    # They should not be passed as arguments, but computed at instantiation as
    # `lflag = isfinite.(l)` and `uflag = isfinite.(u)`
    lflag::Tb

    function IRData(
        A::Ta, b::Tv, objsense::Bool, c::Tv, c0::T, l::Tv,
    ) where{T, Tv<:AbstractVector{T}, Ta<:AbstractMatrix{T}}
        nrow, ncol = size(A)

        lflag = isfinite.(l)
        Tb = typeof(lflag)

        return new{T, Tv, Tb, Ta}(
            nrow, ncol,
            objsense, c0, c,
            A, b, l,  lflag
        )
    end
end

# TODO: extract IPM data from presolved problem
"""
    IRData(pb::ProblemData, options::MatrixOptions)

Extract problem data to standard form.
"""
function IRData(pb::ProblemData{T}) where{T}

    # Problem size
    m, n = pb.ncon, pb.nvar

    # Extract right-hand side and slack variables
    nzA = 0          # Number of non-zeros in A
    b = zeros(T, m)  # RHS
    sind = Int[]     # Slack row index
    sval = T[]       # Slack coefficient
    lslack = T[]     # Slack lower bound


    counter = 1
    for (i, ub) in enumerate(pb.ucon)

        push!(sind, i)
        push!(sval, one(T))
        push!(lslack, zero(T))
        b[i] = ub


        # This line assumes that there are no dupplicate coefficients in Arows
        # Numerical zeros will also be counted as non-zeros
        nzA += length(pb.arows[i].nzind)
    end

    nslack = length(sind)

    # Objective
    c = [pb.obj; zeros(T, nslack)]
    c0 = pb.obj0
    if !pb.objsense
        # Flip objective for maximization problem
        c .= -c
        c0 = -c0
    end

    # Instantiate A
    aI = Vector{Int}(undef, nzA + nslack)
    aJ = Vector{Int}(undef, nzA + nslack)
    aV = Vector{T}(undef, nzA + nslack)

    # populate non-zero coefficients by column
    nz_ = 0
    for (j, col) in enumerate(pb.acols)
        for (i, aij) in zip(col.nzind, col.nzval)
            nz_ += 1

            aI[nz_] = i
            aJ[nz_] = j
            aV[nz_] = aij
        end
    end
    # populate slack coefficients
    for (j, (i, a)) in enumerate(zip(sind, sval))
        nz_ += 1
        aI[nz_] = i
        aJ[nz_] = n + j
        aV[nz_] = a
    end

    # At this point, we should have nz_ == nzA + nslack
    # If not, this means the data between rows and columns in `pb`
    # do not match each other
    nz_ == (nzA + nslack) || error("Found $(nz_) non-zero coeffs (expected $(nzA + nslack))")

    A = zeros(T, m, n+nslack)

    for(i, j, v) in zip(aI, aJ, aV)
        A[i, j] = v
    end


    # Variable bounds
    l = [pb.lvar; lslack]

    return IRData(A, b, pb.objsense, c, c0, l)
end
