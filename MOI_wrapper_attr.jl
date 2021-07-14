# =============================================
#   Supported attributes
# =============================================
const SUPPORTED_OPTIMIZER_ATTR = Union{
    MOI.RawParameter,
    MOI.SolverName,
    MOI.Silent,
}



const SUPPORTED_PARAMETERS = (
    "inner",
    "maxiter",
    "T",
    "verbose",
    "alpha_p",
    "alpha_d",
    "mode",
)





MOI.supports(::Optimizer, ::A) where{A<:SUPPORTED_OPTIMIZER_ATTR} = true

function MOI.supports(::Optimizer, param::MOI.RawParameter)
    return param.name in SUPPORTED_PARAMETERS
end



# =============================================
#   1. Optimizer attributes
# =============================================

include("Refiner.jl")

#
#   SolverName
#
MOI.get(::Optimizer, ::MOI.SolverName) = "IterRef"

#
#   Silent
#
MOI.get(m::Optimizer, ::MOI.Silent) = !m.inner.verbose

function MOI.set(m::Optimizer, ::MOI.Silent, flag::Bool)
    m.inner.verbose = 1 - flag
    return nothing
end




#
#   RawParameter
#
function MOI.set(model::Optimizer, param::MOI.RawParameter, value)
    name = String(param.name)
    tmp = model.inner
    push!(model.options_set, name)
    if name == "inner"
        ref_set_inner(tmp, value)
    elseif name == "maxiter"
        ref_set_maxiter(tmp, value)
    elseif name == "T"
        ref_set_T(tmp, value)
    elseif name == "verbose"
        ref_set_verbose(tmp, value)
    elseif name == "alpha_p"
        ref_set_alpha_p(tmp, value)
    elseif name == "alpha_d"
        ref_set_alpha_d(tmp, value)
    elseif name == "mode"
        ref_set_mode(tmp, value)
    else
        throw(MOI.UnsupportedAttribute(param))
    end
    return
end





# =============================================
#   2. Model attributes
# =============================================

#
#   ListOfVariableIndices
#
function MOI.get(m::Optimizer, ::MOI.ListOfVariableIndices)
    return copy(m.var_indices_moi)
end

#
#   Name
#
MOI.get(m::Optimizer, ::MOI.Name) = m.pbdata.name
MOI.set(m::Optimizer, ::MOI.Name, name) = (m.pbdata.name = name)

#
#   NumberOfVariables
#
MOI.get(m::Optimizer, ::MOI.NumberOfVariables) = m.pbdata.nvar


#
#   ObjectiveSense
#
function MOI.get(m::Optimizer, ::MOI.ObjectiveSense)
    return m.pbdata.objsense ? MOI.MIN_SENSE : MOI.MAX_SENSE
end

function MOI.set(m::Optimizer, ::MOI.ObjectiveSense, s::MOI.OptimizationSense)

    if s == MOI.MIN_SENSE || s == MOI.FEASIBILITY_SENSE
        m.pbdata.objsense = true
    elseif s == MOI.MAX_SENSE
        m.pbdata.objsense = false
    else
        error("Objetive sense not supported: $s")
    end

    return nothing
end

#
#   ObjectiveValue
#
function MOI.get(m::Optimizer{T}, attr::MOI.ObjectiveValue) where{T}
    #MOI.check_result_index_bounds(m, attr)
    c0 = m.pbdata.obj0
    res = ref_get_res(m.inner)
    if !m.pbdata.objsense
        res = -res
    end
    res = c0+res
    return res
end

#
#   DualObjectiveValue
#
# function MOI.get(m::Optimizer{T}, attr::MOI.DualObjectiveValue) where{T}
#     MOI.check_result_index_bounds(m, attr)
#     return get_attribute(m.inner, DualObjectiveValue())
# end

# #
#   RawSolver
#
# MOI.get(m::Optimizer, ::MOI.RawSolver) = m.inner

#
#   RelativeGap
# #
# function MOI.get(m::Optimizer{T}, ::MOI.RelativeGap) where{T}
#     # TODO: dispatch a function call on m.inner
#     zp = m.inner.solver.primal_objective
#     zd = m.inner.solver.dual_objective
#     return (abs(zp - zd) / (T(1 // 10^6)) + abs(zd))
# end

#
#   ResultCount
# #
function MOI.get(m::Optimizer, ::MOI.ResultCount)
    st = MOI.get(m, MOI.TerminationStatus())

    if (st == MOI.OPTIMIZE_NOT_CALLED
        || st == MOI.OTHER_ERROR
        || st == MOI.MEMORY_LIMIT
    )
        return 0
    end
    return 1
end


#
#   TerminationStatus
#
# TODO: use inner query
function MOI.get(m::Optimizer, ::MOI.TerminationStatus)
    return ref_get_status(m.inner)
end

#
#   PrimalStatus
#
# TODO: use inner query
function MOI.get(m::Optimizer, attr::MOI.PrimalStatus)
    return ref_get_p_status(m.inner)
end

# #
# #   DualStatus
# #
# # TODO: enable dual for ls mode
function MOI.get(m::Optimizer, attr::MOI.DualStatus)

        if ref_get_mode(m.inner) > 0
            return MOI.NO_SOLUTION
        end
        return ref_get_d_status(m.inner)

end
