# =============================================
#   1. Supported variable attributes
# =============================================
include("pbdata.jl")
"""
    SUPPORTED_VARIABLE_ATTR

List of supported `MOI.VariableAttribute`.
* `MOI.VariablePrimal`
"""
const SUPPORTED_VARIABLE_ATTR = Union{
    MOI.VariableName,
    MOI.VariablePrimal
}

MOI.supports(::Optimizer, ::MOI.VariableName, ::Type{MOI.VariableIndex}) = true


# =============================================
#   2. Add variables
# =============================================
function MOI.is_valid(m::Optimizer, x::MOI.VariableIndex)
    return haskey(m.var_indices, x)
end

function MOI.add_variable(m::Optimizer{T}) where {T}
    # TODO: dispatch a function call to m.inner instead of m.inner.pbdata
    m.var_counter += 1
    x = MOI.VariableIndex(m.var_counter)
    j = add_variable!(m.pbdata, Int[], T[], zero(T), T(-Inf), T(Inf))

    # Update tracking of variables
    m.var_indices[x] = j
    #m.var2bndtype[x] = Set{Type{<:MOI.AbstractScalarSet}}()
    push!(m.var_indices_moi, x)

    return x
end

# TODO: dispatch to inner model
function MOI.add_variables(m::Optimizer, N::Int)
    N >= 0 || error("Cannot add negative number of variables")

    N == 0 && return MOI.VariableIndex[]

    vars = Vector{MOI.VariableIndex}(undef, N)
    for j in 1:N
        x = MOI.add_variable(m)
        vars[j] = x
    end

    return vars
end


#
function MOI.get(m::Optimizer{T},
    attr::MOI.VariablePrimal,
    x::MOI.VariableIndex
) where{T}
    MOI.throw_if_not_valid(m, x)
    MOI.check_result_index_bounds(m, attr)
    # Query inner solution
    j = m.var_indices[x]
    if j>m.pbdata.zidx
        j -= 1
    end
    return m.inner.x[j]
end
