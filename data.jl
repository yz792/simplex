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
    u::Tv

    zidx::Int # col index of obj


    # Variable bound flags (we template with `Tb` to ease GPU support)
    # These should be vectors of the same type as `l`, `u`, but `Bool` eltype.
    # They should not be passed as arguments, but computed at instantiation as
    # `lflag = isfinite.(l)` and `uflag = isfinite.(u)`
    lflag::Vector{Bool}
    uflag::Vector{Bool}
    svar::Vector{Bool}

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

    # extract true objective function, get auxilary z position, then find corresponding row
    zidx = (1:n)[pb.obj.!=0]
    zidx = zidx[1]
    pb.zidx = zidx

    # Extract right-hand side and slack variables
    nzA = 0          # Number of non-zeros in A
    b = zeros(T, m)  # RHS

    counter = 1
    for (i, ub) in enumerate(pb.ucon)
        b[i] = ub


        # This line assumes that there are no dupplicate coefficients in Arows
        # Numerical zeros will also be counted as non-zeros
        nzA += length(pb.arows[i].nzind)
    end

    # Instantiate A
    aI = Vector{Int}(undef, nzA)
    aJ = Vector{Int}(undef, nzA )
    aV = Vector{T}(undef, nzA )

    # populate non-zero coefficients by column, include z col
    nz_ = 0
    for (j, col) in enumerate(pb.acols)
        if !pb.svar[j]
            col.nzval = -col.nzval
        end
        for (i, aij) in zip(col.nzind, col.nzval)
            nz_ += 1
            aI[nz_] = i
            aJ[nz_] = j
            aV[nz_] = aij
        end
    end

    A = zeros(T, m, n)

    for(i, j, v) in zip(aI, aJ, aV)
        A[i, j] = v
    end

    c = vec(A[1,:])
    c = c[setdiff(1:n, zidx)]
    # default obj0 is zero, need to change
    pb.obj0 = -b[1]
    c0 = pb.obj0
    if !pb.objsense
        # Flip objective for maximization problem
        c .= -c
        c0 = -c0
    end

    A = A[2:end,setdiff(1:n,zidx)]
    #
    # println("before slack")
    # @show size(A)

    b = b[2:end]
    l = copy(pb.lvar)
    l = l[setdiff(1:n,zidx)]

    # upper bound of variable
    for i = 1:n
        if isfinite(pb.uvar[i])
            tmp = T.(zeros(n))
            tmp[i] = T(1)
            A = [A;tmp]
            b = [b;uvar[i]]
        end
    end

    m,n = size(A)
    aux = T.(I(m))


    # reduce the obj row
    noslack = pb.con_noslack.-1
    aux_idx = setdiff(1:m,noslack)
    aux = aux[:,aux_idx]

    A = [A aux]
    c = [c; zeros(length(aux_idx))]
    l = [l;zeros(length(aux_idx))]

    #@show size(A)


    return IRData(A, b, pb.objsense, c, c0, l)
end
