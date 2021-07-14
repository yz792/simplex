# =============================================
#   1. Supported constraints and attributes
# =============================================

"""
    SUPPORTED_CONSTR_ATTR

List of supported MOI `ConstraintAttribute`.
"""
const SUPPORTED_CONSTR_ATTR = Union{
    MOI.ConstraintName,
    #MOI.ConstraintPrimal,
    #MOI.ConstraintDual,
    MOI.ConstraintFunction,
    MOI.ConstraintSet
}

MOI.supports(::Optimizer, ::A, ::Type{<:MOI.ConstraintIndex}) where{A<:SUPPORTED_CONSTR_ATTR} = true

# Variable bounds
function MOI.supports_constraint(
    ::Optimizer{T}, ::Type{MOI.SingleVariable}, ::Type{S}
) where {T, S<:SCALAR_SETS{T}}
    return true
end

# Linear constraints
function MOI.supports_constraint(
    ::Optimizer{T}, ::Type{MOI.ScalarAffineFunction{T}}, ::Type{S}
) where {T, S<:SCALAR_SETS{T}}
    return true
end


function MOI.is_valid(
    m::Optimizer{T},
    c::MOI.ConstraintIndex{MOI.SingleVariable, S}
) where{T, S <:SCALAR_SETS}
    v = MOI.VariableIndex(c.value)
    MOI.is_valid(m, v) || return false

    return haskey(m.con_indices, c)
end

function MOI.is_valid(
    m::Optimizer{T},
    c::MOI.ConstraintIndex{MOI.ScalarAffineFunction{T}, S}
) where{T, S<:SCALAR_SETS{T}}
    return haskey(m.con_indices, c)
end

# =============================================
#   2. Add constraints
# =============================================

# TODO: make it clear that only finite bounds can be given in input.
# To relax variable bounds, one should delete the associated bound constraint.
function MOI.add_constraint(
    m::Optimizer{T},
    f::MOI.SingleVariable,
    s::MOI.LessThan{T}
) where{T}

    # Check that variable exists
    v = f.variable
    MOI.throw_if_not_valid(m, v)
    j = m.var_indices[v]  # inner index

    1 <= j <= m.pbdata.nvar || error("Invalid variable index $j")

    m.pbdata.uvar[j] = s.upper
    m.pbdata.lvar[j] = -T(Inf)
    return MOI.ConstraintIndex{MOI.SingleVariable, MOI.LessThan{T}}(v.value)
end

function MOI.add_constraint(
    m::Optimizer{T},
    f::MOI.SingleVariable,
    s::MOI.GreaterThan{T}
) where{T}

    # Check that variable exists
    v = f.variable
    MOI.throw_if_not_valid(m, v)

    # Update inner model
    j = m.var_indices[v]  # inner index

    1 <= j <= m.pbdata.nvar || error("Invalid variable index $j")

    # Update bound
    m.pbdata.lvar[j] = s.lower
    m.pbdata.uvar[j] = T(Inf)
    return MOI.ConstraintIndex{MOI.SingleVariable, MOI.GreaterThan{T}}(v.value)
end

function MOI.add_constraint(
    m::Optimizer{T},
    f::MOI.SingleVariable,
    s::MOI.EqualTo{T}
) where{T}
    # Check that variable exists
    v = f.variable
    MOI.throw_if_not_valid(m, v)
    # Check if a bound already exists

    # Update inner model
    j = m.var_indices[v]  # inner index
    1 <= j <= m.pbdata.nvar || error("Invalid variable index $j")

    m.pbdata.lvar[j] = s.value
    m.pbdata.uvar[j] = s.value
    # Update bound tracking
    return MOI.ConstraintIndex{MOI.SingleVariable, MOI.EqualTo{T}}(v.value)
end

function MOI.add_constraint(
    m::Optimizer{T},
    f::MOI.SingleVariable,
    s::MOI.Interval{T}
) where{T}

    # Check that variable exists
    v = f.variable
    MOI.throw_if_not_valid(m, v)
    # Check if a bound already exists

    # Update variable bounds
    j = m.var_indices[v]  # inner index
    m.pbdata.uvar[j] = s.upper
    m.pbdata.lvar[j] = s.lower

    # Update bound tracking

    return MOI.ConstraintIndex{MOI.SingleVariable, MOI.Interval{T}}(v.value)
end

# General linear constraints
function MOI.add_constraint(
    m::Optimizer{T},
    f::MOI.ScalarAffineFunction{T},
    s::SCALAR_SETS{T}
) where{T}
    # Check that constant term is zero
    if !iszero(f.constant)
        throw(MOI.ScalarFunctionConstantNotZero{T, typeof(f), typeof(s)}(f.constant))
    end

    # Convert to canonical form
    fc = MOI.Utilities.canonical(f)

    # Extract row
    nz = length(fc.terms)
    rind = Vector{Int}(undef, nz)
    rval = Vector{T}(undef, nz)
    lb, ub = _bounds(s)
    for (k, t) in enumerate(fc.terms)
        rind[k] = m.var_indices[t.variable_index]
        rval[k] = t.coefficient
    end

    # Update inner model
    i = add_constraint!(m.pbdata, rind, rval, lb, ub)

    # Create MOI index
    m.con_counter += 1
    cidx = MOI.ConstraintIndex{typeof(f), typeof(s)}(m.con_counter)

    # Update constraint tracking
    m.con_indices[cidx] = i
    push!(m.con_indices_moi, cidx)

    return cidx
end


# =============================================
#   5. Get/set constraint attributes
# =============================================

#
#   ListOfConstraintIndices
#
function MOI.get(
    m::Optimizer{T},
    ::MOI.ListOfConstraintIndices{MOI.ScalarAffineFunction{T}, S}
) where{T, S<:SCALAR_SETS{T}}
    return [
        cidx
        for cidx in keys(m.con_indices) if isa(cidx,
            MOI.ConstraintIndex{MOI.ScalarAffineFunction{T}, S}
        )
    ]
end

#
#   NumberOfConstraints
#



#
#   ConstraintName
#



function MOI.get(
    m::Optimizer{T}, ::MOI.ConstraintName,
    c::MOI.ConstraintIndex{MOI.ScalarAffineFunction{T}, S}
) where{T, S<:SCALAR_SETS{T}}
    MOI.throw_if_not_valid(m, c)

    # Get name from inner model
    i = m.con_indices[c]
    return get_attribute(m.inner, ConstraintName(), i)
end



function MOI.set(
    m::Optimizer{T}, ::MOI.ConstraintName,
    c::MOI.ConstraintIndex{MOI.ScalarAffineFunction{T}, S},
    name::String
) where{T, S<:SCALAR_SETS{T}}
    MOI.throw_if_not_valid(m, c)

    # Check for dupplicate name
    c_ = get(m.name2con, name, nothing)
    c_ === nothing || c_ == c || error("Dupplicate constraint name $name")

    # Update inner model
    i = m.con_indices[c]
    old_name = get_attribute(m.inner, ConstraintName(), i)
    set_attribute(m.inner, ConstraintName(), i, name)

    # Update constraint name tracking
    delete!(m.name2con, old_name)
    if name != ""
        m.name2con[name] = c
    end
    return nothing
end

function MOI.get(m::Optimizer, CIType::Type{<:MOI.ConstraintIndex}, name::String)
    c = get(m.name2con, name, nothing)
    return isa(c, CIType) ? c : nothing
end

#
#   ConstraintFunction
#
function MOI.get(
    m::Optimizer{T}, ::MOI.ConstraintFunction,
    c::MOI.ConstraintIndex{MOI.SingleVariable, S}
) where{T, S<:SCALAR_SETS{T}}
    MOI.throw_if_not_valid(m, c)  # Sanity check

    return MOI.SingleVariable(MOI.VariableIndex(c.value))
end

function MOI.set(
    m::Optimizer{T}, ::MOI.ConstraintFunction,
    c::MOI.ConstraintIndex{MOI.SingleVariable, S},
    ::MOI.SingleVariable
) where{T, S<:SCALAR_SETS{T}}
    return throw(MOI.SettingSingleVariableFunctionNotAllowed())
end

function MOI.get(
    m::Optimizer{T}, ::MOI.ConstraintFunction,
    c::MOI.ConstraintIndex{MOI.ScalarAffineFunction{T}, S}
) where{T, S<:SCALAR_SETS{T}}
    MOI.throw_if_not_valid(m, c)  # Sanity check

    # Get row from inner model
    i = m.con_indices[c]
    row = m.pbdata.arows[i]
    nz = length(row.nzind)

    # Map inner indices to MOI indices
    terms = Vector{MOI.ScalarAffineTerm{T}}(undef, nz)
    for (k, (j, v)) in enumerate(zip(row.nzind, row.nzval))
        terms[k] = MOI.ScalarAffineTerm{T}(v, m.var_indices_moi[j])
    end

    return MOI.ScalarAffineFunction(terms, zero(T))
end

# TODO
function MOI.set(
    m::Optimizer{T}, ::MOI.ConstraintFunction,
    c::MOI.ConstraintIndex{MOI.ScalarAffineFunction{T}, S},
    f::MOI.ScalarAffineFunction{T}
) where{T, S<:SCALAR_SETS{T}}
    MOI.throw_if_not_valid(m, c)
    iszero(f.constant) || throw(MOI.ScalarFunctionConstantNotZero{T, typeof(f), S}(f.constant))

    fc = MOI.Utilities.canonical(f)

    # Update inner model
    # TODO: use inner query
    i = m.con_indices[c]
    # Set old row to zero
    f_old = MOI.get(m, MOI.ConstraintFunction(), c)
    for term in f_old.terms
        j = m.var_indices[term.variable_index]
        set_coefficient!(m.pbdata, i, j, zero(T))
    end
    # Set new row coefficients
    for term in fc.terms
        j = m.var_indices[term.variable_index]
        set_coefficient!(m.pbdata, i, j, term.coefficient)
    end

    # Done

    return nothing
end

#
#   ConstraintSet
#
function MOI.get(
    m::Optimizer{T}, ::MOI.ConstraintSet,
    c::MOI.ConstraintIndex{MOI.SingleVariable, MOI.LessThan{T}}
) where{T}
    # Sanity check
    MOI.throw_if_not_valid(m, c)
    v = MOI.VariableIndex(c.value)

    # Get inner bounds
    j  = m.var_indices[v]
    ub = m.pbdata.uvar[j]

    return MOI.LessThan(ub)
end

function MOI.get(
    m::Optimizer{T}, ::MOI.ConstraintSet,
    c::MOI.ConstraintIndex{MOI.SingleVariable, MOI.GreaterThan{T}}
) where{T}
    # Sanity check
    MOI.throw_if_not_valid(m, c)
    v = MOI.VariableIndex(c.value)

    # Get inner bounds
    j  = m.var_indices[v]
    lb = m.pbdata.lvar[j]

    return MOI.GreaterThan(lb)
end

function MOI.get(
    m::Optimizer{T}, ::MOI.ConstraintSet,
    c::MOI.ConstraintIndex{MOI.SingleVariable, MOI.EqualTo{T}}
) where{T}
    # Sanity check
    MOI.throw_if_not_valid(m, c)
    v = MOI.VariableIndex(c.value)

    # Get inner bounds
    j  = m.var_indices[v]
    ub = m.pbdata.uvar[j]

    return MOI.EqualTo(ub)
end

function MOI.get(
    m::Optimizer{T}, ::MOI.ConstraintSet,
    c::MOI.ConstraintIndex{MOI.SingleVariable, MOI.Interval{T}}
) where{T}
    # Sanity check
    MOI.throw_if_not_valid(m, c)
    v = MOI.VariableIndex(c.value)

    # Get inner bounds
    j  = m.var_indices[v]
    lb = m.pbdata.lvar[j]
    ub = m.pbdata.uvar[j]

    return MOI.Interval(lb, ub)
end

function MOI.get(
    m::Optimizer{T}, ::MOI.ConstraintSet,
    c::MOI.ConstraintIndex{MOI.ScalarAffineFunction{T}, S}
) where{T, S<:SCALAR_SETS{T}}
    MOI.throw_if_not_valid(m, c)  # Sanity check

    # Get inner bounds
    i = m.con_indices[c]
    lb = m.pbdata.lcon[i]
    ub = m.pbdata.ucon[i]

    if S == MOI.LessThan{T}
        return MOI.LessThan(ub)
    elseif S == MOI.GreaterThan{T}
        return MOI.GreaterThan(lb)
    elseif S == MOI.EqualTo{T}
        return MOI.EqualTo(lb)
    elseif S == MOI.Interval{T}
        return MOI.Interval(lb, ub)
    end
end

function MOI.set(
    m::Optimizer{T}, ::MOI.ConstraintSet,
    c::MOI.ConstraintIndex{MOI.SingleVariable, S},
    s::S
) where{T, S<:SCALAR_SETS{T}}
    # Sanity check
    MOI.throw_if_not_valid(m, c)
    v = MOI.VariableIndex(c.value)

    # Update inner bounds
    # Bound key does not need to be updated
    j = m.var_indices[v]
    if S == MOI.LessThan{T}
        set_attribute(m.inner, VariableUpperBound(), j, s.upper)
    elseif S == MOI.GreaterThan{T}
        set_attribute(m.inner, VariableLowerBound(), j, s.lower)
    elseif S == MOI.EqualTo{T}
        set_attribute(m.inner, VariableLowerBound(), j, s.value)
        set_attribute(m.inner, VariableUpperBound(), j, s.value)
    elseif S == MOI.Interval{T}
        set_attribute(m.inner, VariableLowerBound(), j, s.lower)
        set_attribute(m.inner, VariableUpperBound(), j, s.upper)
    else
        error("Unknown type for ConstraintSet: $S.")
    end

    return nothing
end

function MOI.set(
    m::Optimizer{T}, ::MOI.ConstraintSet,
    c::MOI.ConstraintIndex{MOI.ScalarAffineFunction{T}, S},
    s::S
) where{T, S<:SCALAR_SETS{T}}
    MOI.throw_if_not_valid(m, c)

    # Update inner bounds
    i = m.con_indices[c]
    if S == MOI.LessThan{T}
        set_attribute(m.inner, ConstraintUpperBound(), i, s.upper)
    elseif S == MOI.GreaterThan{T}
        set_attribute(m.inner, ConstraintLowerBound(), i, s.lower)
    elseif S == MOI.EqualTo{T}
        set_attribute(m.inner, ConstraintLowerBound(), i, s.value)
        set_attribute(m.inner, ConstraintUpperBound(), i, s.value)
    elseif S == MOI.Interval{T}
        set_attribute(m.inner, ConstraintLowerBound(), i, s.lower)
        set_attribute(m.inner, ConstraintUpperBound(), i, s.upper)
    else
        error("Unknown type for ConstraintSet: $S.")
    end

    return nothing
end

#
#   ConstraintPrimal
#
function MOI.get(
    m::Optimizer{T}, attr::MOI.ConstraintPrimal,
    c::MOI.ConstraintIndex{MOI.SingleVariable, S}
) where{T, S<:SCALAR_SETS{T}}
    MOI.throw_if_not_valid(m, c)
    MOI.check_result_index_bounds(m, attr)

    # Query row primal
    j = m.var_indices[MOI.VariableIndex(c.value)]
    return m.inner.x[j]
end

# function MOI.get(
#     m::Optimizer{T}, attr::MOI.ConstraintPrimal,
#     c::MOI.ConstraintIndex{MOI.ScalarAffineFunction{T}, S}
# ) where{T, S<:SCALAR_SETS{T}}
#     MOI.throw_if_not_valid(m, c)
#     MOI.check_result_index_bounds(m, attr)
#
#     # Query from inner model
#     i = m.con_indices[c]
#     return m.inner.solution.Ax[i]
# end

#
#   ConstraintDual
#
# function MOI.get(
#     m::Optimizer{T}, attr::MOI.ConstraintDual,
#     c::MOI.ConstraintIndex{MOI.SingleVariable, S}
# ) where{T, S<:SCALAR_SETS{T}}
#     MOI.throw_if_not_valid(m, c)
#     MOI.check_result_index_bounds(m, attr)
#
#     # Get variable index
#     j = m.var_indices[MOI.VariableIndex(c.value)]
#
#     # Extract reduced cost
#     if S == MOI.LessThan{T}
#         return -m.inner.solution.s_upper[j]
#     elseif S == MOI.GreaterThan{T}
#         return m.inner.solution.s_lower[j]
#     else
#         return m.inner.solution.s_lower[j] - m.inner.solution.s_upper[j]
#     end
# end
#
function MOI.get(
    m::Optimizer{T}, attr::MOI.ConstraintDual,
    c::MOI.ConstraintIndex{MOI.ScalarAffineFunction{T}, S}
) where{T, S<:SCALAR_SETS{T}}
    MOI.throw_if_not_valid(m, c)
    MOI.check_result_index_bounds(m, attr)

    # Get dual from inner model
    i = m.con_indices[c]
    return 0
    #return m.inner.y[i]
end
