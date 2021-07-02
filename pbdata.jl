using SparseArrays

mutable struct RowOrCol{T}
    nzind::Vector{Int}
    nzval::Vector{T}
end

const Row = RowOrCol
const Col = RowOrCol

"""
    ProblemData{T}

Data structure for storing problem data in precision `T`.

The LP is represented in canonical form

```math
\\begin{array}{rl}
    \\displaystyle \\min_{x} \\ \\ \\ & c^{T} x + c_{0} \\\\
    s.t. \\ \\ \\ & l_{r} \\leq A x \\leq u_{r} \\\\
    & l_{c} \\leq x \\leq u_{c}
\\end{array}
```
"""
mutable struct ProblemData{T}

    name::String

    # Dimensions
    ncon::Int  # Number of rows
    nvar::Int  # Number of columns (i.e. variables)
    nsincon::Int # Number of single constraint (i.e. so that every variable will be treated as free variable)

    # Objective
    # TODO: objective sense
    objsense::Bool  # true is min, false is max
    obj::Vector{T}
    obj0::T  # Constant objective offset

    # Constraint matrix
    # We store both rows and columns. It is redundant but simplifies access.
    # TODO: put this in its own data structure? (would allow more flexibility in modelling)
    arows::Vector{Row{T}}
    acols::Vector{Col{T}}

    # TODO: Data structures for QP
    # qrows
    # qcols

    # Bounds
    # only store in the form Ax <= b and x >= l
    ucon::Vector{T}
    lvar::Vector{T}

    # Names
    con_names::Vector{String}
    var_names::Vector{String}

    # Only allow empty problems to be instantiated for now
    ProblemData{T}(pbname::String="") where {T} = new{T}(
        pbname, 0, 0,0,
        true, T[], zero(T),
        Row{T}[], Col{T}[],
         T[], T[],
        String[], String[]
    )
end

import Base.empty!

function Base.empty!(pb::ProblemData{T}) where{T}

    pb.name = ""

    pb.ncon = 0
    pb.nvar = 0
    pb.nsincon = 0

    pb.objsense = true
    pb.obj = T[]
    pb.obj0 = zero(T)

    pb.arows = Row{T}[]
    pb.acols = Col{T}[]

    pb.ucon = T[]
    pb.lvar = T[]

    pb.con_names = String[]
    pb.var_names = String[]

    return pb
end


# =============================
#     Problem creation
# =============================

"""
    add_constraint!(pb, rind, rval, l, u; [name, issorted])

Add one linear constraint to the problem s.t. a^T x <= b

# Arguments
* `pb::ProblemData{T}`: the problem to which the new row is added
* `rind::Vector{Int}`: column indices in the new row
* `rval::Vector{T}`: non-zero values in the new row
* `l::T`
* `u::T`
* `name::String`: row name (defaults to `""`)
* `issorted::Bool`: indicates whether the row indices are already issorted.
"""
function add_constraint!(pb::ProblemData{T},
    rind::Vector{Int}, rval::Vector{T},
    l::T, u::T,
    name::String="";
    issorted::Bool=false
)::Int where{T}
    # Sanity checks
    nz = length(rind)
    nz == length(rval) || throw(DimensionMismatch(
        "Cannot add a row with $nz indices but $(length(rval)) non-zeros"
    ))

    # Go through through rval to check all coeffs are finite and remove zeros.
    _rind = Vector{Int}(undef, nz)
    _rval = Vector{T}(undef, nz)
    _nz = 0
    for (j, aij) in zip(rind, rval)
        if !iszero(aij)
            isfinite(aij) || error("Invalid row coefficient: $(aij)")
            _nz += 1
            _rind[_nz] = j
            _rval[_nz] = aij
        end
    end
    resize!(_rind, _nz)
    resize!(_rval, _nz)

    p = sortperm(_rind)

    # TODO: combine dupplicate indices

    # Increment row counter
    if isfinite(u)
        pb.ncon += 1
        push!(pb.ucon, u)
        push!(pb.con_names, name)
        # Create new row
        if issorted
            row = Row{T}(_rind, _rval)
        else
            # Sort indices first
            row = Row{T}(_rind[p], _rval[p])
        end

        push!(pb.arows, row)
        # Update column coefficients
        for (j, aij) in zip(_rind, _rval)
            push!(pb.acols[j].nzind, pb.ncon)
            push!(pb.acols[j].nzval, aij)
        end
    end

    if isfinite(l)
        pb.ncon += 1
        push!(pb.ucon, -l)
        # if already named then don't add again
        if isfinite(u)
            push!(pb.con_names, "")
        else
            push!(pb.con_names, name)
        end
        _rval2 = -_rval
        if issorted
            row2 = Row{T}(_rind, _rval2)
        else
            # Sort indices first
            row2 = Row{T}(_rind[p], _rval2[p])
        end
        push!(pb.arows, row2)

        for (j, aij) in zip(_rind, _rval2)
            push!(pb.acols[j].nzind, pb.ncon)
            push!(pb.acols[j].nzval, aij)
        end

    end
    # Done
    return pb.ncon
end

"""
    add_variable!(pb, cind, cval, obj, l, u, [name])

Add one variable to the problem.

# Arguments
* `pb::ProblemData{T}`: the problem to which the new column is added
* `cind::Vector{Int}`: row indices in the new column
* `cval::Vector{T}`: non-zero values in the new column
* `obj::T`: objective coefficient
* `l::T`: column lower bound
* `u::T`: column upper bound
* `name::String`: column name (defaults to `""`)
* `issorted::Bool`: indicates whether the column indices are already issorted.
"""
function add_variable!(pb::ProblemData{T},
    cind::Vector{Int}, cval::Vector{T},
    obj::T, l::T, u::T,
    name::String="";
    issorted::Bool=false
)::Int where{T}
    # Sanity checks
    nz = length(cind)
    nz == length(cval) || throw(DimensionMismatch(
        "Cannot add a column with $nz indices but $(length(cval)) non-zeros"
    ))

    # Go through through cval to check all coeffs are finite and remove zeros.
    _cind = Vector{Int}(undef, nz)
    _cval = Vector{T}(undef, nz)
    _nz = 0
    for (j, aij) in zip(cind, cval)
        if !iszero(aij)
            isfinite(aij) || error("Invalid column coefficient: $(aij)")
            _nz += 1
            _cind[_nz] = j
            _cval[_nz] = aij
        end
    end
    resize!(_cind, _nz)
    resize!(_cval, _nz)

    # Increment column counter
    pb.nvar += 1
    if !isfinite(l)

        # TODO need change to handle infinity
        push!(pb.lvar,T(-1e5))
    else
        push!(pb.lvar, l)
    end
    #push!(pb.uvar, u)
    push!(pb.obj, obj)
    push!(pb.var_names, name)

    # TODO: combine dupplicate indices

    # Create a new column
    if issorted
        col = Col{T}(_cind, _cval)
    else
        # Sort indices
        p = sortperm(_cind)
        col = Col{T}(_cind[p], _cval[p])
    end
    push!(pb.acols, col)

    # Update row coefficients
    for (i, aij) in zip(_cind, _cval)
        push!(pb.arows[i].nzind, pb.nvar)
        push!(pb.arows[i].nzval, aij)
    end

    # Done
    return pb.nvar
end

"""
    load_problem!(pb, )

Load entire problem.
"""
function load_problem!(pb::ProblemData{T},
    name::String,
    objsense::Bool, obj::Vector{T}, obj0::T,
    A::SparseMatrixCSC,
    lcon::Vector{T}, ucon::Vector{T},
    lvar::Vector{T}, uvar::Vector{T},
    con_names::Vector{String}, var_names::Vector{String}
) where{T}
    empty!(pb)

    # Sanity checks
    ncon, nvar = size(A)
    ncon == length(lcon) || error("")
    ncon == length(ucon) || error("")
    ncon == length(con_names) || error("")
    nvar == length(obj)
    isfinite(obj0) || error("Objective offset $obj0 is not finite")
    nvar == length(lvar) || error("")
    nvar == length(uvar) || error("")

    # Copy data
    pb.name = name
    pb.ncon = ncon
    pb.nvar = nvar
    pb.objsense = objsense
    pb.obj = copy(obj)
    pb.obj0 = obj0
    pb.lcon = copy(lcon)
    pb.ucon = copy(ucon)
    pb.lvar = copy(lvar)
    pb.uvar = copy(uvar)
    pb.con_names = copy(con_names)
    pb.var_names = copy(var_names)

    # Load coefficients
    pb.acols = Vector{Col{T}}(undef, nvar)
    pb.arows = Vector{Row{T}}(undef, ncon)
    for j in 1:nvar
        col = A[:, j]
        pb.acols[j] = Col{T}(col.nzind, col.nzval)
    end

    At = sparse(A')
    for i in 1:ncon
        row = At[:, i]
        pb.arows[i] = Row{T}(row.nzind, row.nzval)
    end

    return pb
end
