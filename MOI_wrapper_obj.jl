# =============================================
#   1. Supported objectives
# =============================================
function MOI.supports(
    #::Optimizer{T},
    ::Optimizer{T},
    ::MOI.ObjectiveFunction{F}
) where{T, F<:Union{MOI.SingleVariable, MOI.ScalarAffineFunction}}
    return true
end

# =============================================
#   2. Get/set objective function
# =============================================
function MOI.get(
    m::Optimizer{T},
    ::MOI.ObjectiveFunction{MOI.SingleVariable}
) where{T}
    obj = MOI.get(m, MOI.ObjectiveFunction{MOI.ScalarAffineFunction}())
    return convert(MOI.SingleVariable, obj)
end

function MOI.get(
    m::Optimizer{T},
    ::MOI.ObjectiveFunction{MOI.ScalarAffineFunction}
) where{T}
    # Objective coeffs
    terms = MOI.ScalarAffineTerm{T}[]
    for (j, cj) in enumerate(m.pbdata.obj)
        !iszero(cj) && push!(terms, MOI.ScalarAffineTerm(cj, m.var_indices_moi[j]))
    end

    # Constant term
    c0 = m.pbdata.obj0

    return MOI.ScalarAffineFunction(terms, c0)
end

# TODO: use inner API
function MOI.set(
    m::Optimizer{T},
    ::MOI.ObjectiveFunction{F},
    f::F
) where{T, F <: MOI.SingleVariable}

    MOI.set(
        m, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{T}}(),
        convert(MOI.ScalarAffineFunction{T}, f)
    )

    return nothing
end

function MOI.set(
    m::Optimizer{T},
    ::MOI.ObjectiveFunction{MOI.ScalarAffineFunction{T}},
    f::MOI.ScalarAffineFunction{T}
) where{T}

    # Sanity checks
    isfinite(f.constant) || error("Objective constant term must be finite")
    for t in f.terms
        MOI.throw_if_not_valid(m, t.variable_index)
    end

    # Update inner model
    m.pbdata.obj .= zero(T) # Reset inner objective to zero
    for t in f.terms
        j = m.var_indices[t.variable_index]
        m.pbdata.obj[j] += t.coefficient  # there may be dupplicates
    end
    #set_attribute(m.inner, ObjectiveConstant(), f.constant)  # objective offset

    m.pbdata.obj0 = f.constant

    return nothing
end
