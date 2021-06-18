import MathOptInterface
const MOI = MathOptInterface
using Revise

include("pbdata.jl")
include("data.jl")
include("Refiner.jl")

# ==============================================================================
#           HELPER FUNCTIONS
# ==============================================================================


#
#
# """
#     _bounds(s)
#
# """
_bounds(s::MOI.EqualTo{T}) where{T} = s.value, s.value
_bounds(s::MOI.LessThan{T}) where{T}  = T(-Inf), s.upper
_bounds(s::MOI.GreaterThan{T}) where{T}  = s.lower, T(Inf)
_bounds(s::MOI.Interval{T}) where{T}  = s.lower, s.upper

const SCALAR_SETS{T} = Union{
    MOI.LessThan{T},
    MOI.GreaterThan{T},
    MOI.EqualTo{T},
    MOI.Interval{T}
} where{T}

@enum(ObjType, _SINGLE_VARIABLE, _SCALAR_AFFINE)


# ==============================================================================
# ==============================================================================
#
#               S U P P O R T E D    M O I    F E A T U R E S
#
# ==============================================================================
# ==============================================================================

"""
    Optimizer{T}

Wrapper for MOI.
"""
mutable struct Optimizer{T} <: MOI.AbstractOptimizer
    inner::Refiner{T}



    # Map MOI Variable/Constraint indices to internal indices
    var_counter::Int  # Should never be reset
    con_counter::Int  # Should never be reset
    var_indices_moi::Vector{MOI.VariableIndex}
    var_indices::Dict{MOI.VariableIndex, Int}
    con_indices_moi::Vector{MOI.ConstraintIndex}
    con_indices::Dict{MOI.ConstraintIndex{MOI.ScalarAffineFunction{T}, <:SCALAR_SETS{T}}, Int}

    # Variable and constraint names
    name2var::Dict{String, MOI.VariableIndex}
    name2con::Dict{String, MOI.ConstraintIndex}
    # MOIIndex -> name mapping for SingleVariable constraints
    # Will be dropped with MOI 0.10
    #   => (https://github.com/jump-dev/MathOptInterface.jl/issues/832)
    pbdata::ProblemData{T}


    function Optimizer{T}(;kwargs...) where{T}
        m = new{T}(
            Refiner{T}(),
            # Variable and constraint counters
            0, 0,
            # Index mapping
            MOI.VariableIndex[], Dict{MOI.VariableIndex, Int}(),
            MOI.ConstraintIndex[], Dict{MOI.ConstraintIndex{MOI.ScalarAffineFunction, <:SCALAR_SETS{T}}, Int}(),
            # Name -> index mapping
            Dict{String, MOI.VariableIndex}(), Dict{String, MOI.ConstraintIndex}(),
             ProblemData{T}() # Variable bounds tracking
        )

        # if length(kwargs) > 0
        #     @warn("""Passing optimizer attributes as keyword arguments to
        #         Optimizer()
        #         ## Example
        #
        #             using JuMP, Clp
        #             model = JuMP.Model(IterRef.Optimizer)
        #
        #                         MOI.set(model, MOI.RawParameter("key"), value)
        #                     or
        #                         JuMP.set_optimizer_attribute(model, "key", value)
        #                     instead.
        #                     """)
        #     end

        for (k, v) in kwargs
            set_parameter(m.inner, string(k), v)
        end


        return m
    end
end

Optimizer(;kwargs...) = Optimizer{Float64}(;kwargs...)




function MOI.empty!(m::Optimizer)
    # Inner model
    empty!(m.inner)
    # Reset index mappings
    m.var_indices_moi = MOI.VariableIndex[]
    m.con_indices_moi = MOI.ConstraintIndex[]
    m.var_indices = Dict{MOI.VariableIndex, Int}()
    m.con_indices = Dict{MOI.ConstraintIndex, Int}()

    # Reset name mappings
    m.name2var = Dict{String, MOI.VariableIndex}()
    m.name2con = Dict{String, MOI.ConstraintIndex}()


end

function MOI.is_empty(m::Optimizer)
    m.pbdata.nvar == 0 || return false
    m.pbdata.ncon == 0 || return false

    length(m.var_indices) == 0 || return false
    length(m.var_indices_moi) == 0 || return false
    length(m.con_indices) == 0 || return false
    length(m.con_indices_moi) == 0 || return false

    length(m.name2var) == 0 || return false
    length(m.name2con) == 0 || return false


    return true
end


function MOI.optimize!(model::Optimizer)
    t = time()
    tmpdata = IRData(model.pbdata)
    #print(tmpdata.A)

    inner = model.inner
    ref_set_A(inner,tmpdata.A)
    ref_set_b(inner,tmpdata.b)
    ref_set_c(inner,tmpdata.c)
    ref_set_l(inner,tmpdata.l)


    # @show (inner.b)

    # @show (inner.c)
    # @show (inner.A)

    #@show size(inner.A)
    #@show size(inner.c)
    #@show size(inner.l)
    #@show size(inner.b)

    @show typeof(inner.l)


    optimize!(inner)
    return
end



# MOI.Utilities.supports_default_copy_to(::Optimizer, ::Bool) = true
#
# function MOI.copy_to(dest::Optimizer, src::MOI.ModelLike; kwargs...)
#     return MOI.Utilities.automatic_copy_to(dest, src; kwargs...)
# end


# ==============================================================================
#           I. Optimizer attributes
# ==============================================================================
# ==============================================================================
#           II. Model attributes
# ==============================================================================
include("MOI_wrapper_attr.jl")

# ==============================================================================
#           III. Variables
# ==============================================================================
include("MOI_wrapper_var.jl")

# ==============================================================================
#           IV. Constraints
# ==============================================================================
include("MOI_wrapper_constraint.jl")

# ==============================================================================
#           V. Objective
# ==============================================================================
include("MOI_wrapper_obj.jl")
