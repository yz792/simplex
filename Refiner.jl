import MathOptInterface
import SparseArrays
using JuMP
using GLPK
using LinearAlgebra

const MOI = MathOptInterface


# =============================================
#   1. Initialization
# =============================================
mutable struct Refiner{T}
    #inner <: MOI.AbstractOptimizer
    # TODO other verbose parameter e.g. #of x, size of A

    # setting parameter
    inner
    maxiter::Int
    verbose::Bool
    T::Type # solution return type
    alpha_p::Float64
    alpha_d::Float64
    mode::Int # 1 = solving iterative linear system, 2 = solve high precision linear system, 0 = solve iterative linear programming

    # inputs
    A::AbstractArray{T,2}
    b::Array{T,1}
    c::Array{T,1}
    l::Array{T,1} # default: min cx s.t. Ax =b, x >= l


    # outputs
    x::Array{T,1} # primal solution
    y::Array{T,1} # dual solution
    res::T
    status::MOI.TerminationStatusCode
    p_status::MOI.ResultStatusCode
    d_status::MOI.ResultStatusCode
    k::Int
    solve_time::Float64
    solve_time_lp::Float64
    solve_time_ls::Float64





    function Refiner{T}(; kwargs...) where{T}
        # default setting
        model =
            new{T}(GLPK.Optimizer, 10, false,T , 1e8, 1e8,0,T.(zeros(1,1)),T[],T[],T[],T[],T[],0,MOI.OPTIMIZE_NOT_CALLED,MOI.UNKNOWN_RESULT_STATUS,MOI.UNKNOWN_RESULT_STATUS,0,0,0,0)


        if length(kwargs) > 0
            @warn("""please set attribute later.
            ## Example
            m = Refiner()
            set_inner(m,costumized_inner)
            """)

        end
        return model

    end
end

# TODO need try catch to avoid type error
function set_parameter(tmp::Refiner, name::String, value)
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
        throw("IterRef doesn't support  parameter ", name)
    end
end


function empty!(opt::Refiner)
    T = opt.T
    opt.x = T[]
    opt.y = T[]
    opt.res = 0
    opt.status=MOI.OPTIMIZE_NOT_CALLED
    opt.p_status=MOI.UNKNOWN_RESULT_STATUS
    opt.d_status=MOI.UNKNOWN_RESULT_STATUS
    opt.k=0
    opt.solve_time=0
    opt.solve_time_lp=0
    opt.solve_time_ls=0
    opt.A = T.(zeros(1,1))
    opt.b = T[]
    opt.c = T[]
    opt.l = T[]
end

# =============================================
#   2. Getter and Setter
# =============================================

function ref_set_A(opt::Refiner,A::AbstractArray{<:AbstractFloat,2})
    opt.A = A
end

function ref_set_b(opt::Refiner,b::Array{<:AbstractFloat,1})
    opt.b = b
end

function ref_set_c(opt::Refiner,c::Array{<:AbstractFloat,1})
    opt.c = c
end

function ref_set_l(opt::Refiner,l::Array{<:AbstractFloat,1})
    opt.l = l
end

function ref_set_T(opt::Refiner,T)
    opt.T = T
end

function ref_set_alpha_p(opt::Refiner,alpha_p)
    opt.alpha_p = alpha_p
end

function ref_set_alpha_p(opt::Refiner,alpha_d)
    opt.alpha_d = alpha_d
end

function ref_set_inner(opt::Refiner,inner)
    opt.inner = inner
end

function ref_set_maxiter(opt::Refiner,maxiter::Int)
    opt.maxiter = maxiter
end

function ref_set_verbose(opt::Refiner,verbose::Bool)
    opt.verbose = verbose
end

function ref_set_mode(opt::Refiner,mode::Int)
    opt.mode = mode
end

function ref_get_mode(opt::Refiner)
    return opt.mode 
end



function ref_get_x(opt::Refiner)
    return opt.x
end

function ref_get_status(opt::Refiner)
    return opt.status
end

function ref_get_p_status(opt::Refiner)
    return opt.p_status
end

function ref_get_d_status(opt::Refiner)
    return opt.d_status
end

function ref_get_res(opt::Refiner)
    return opt.res
end

function ref_get_y(opt::Refiner)
    return opt.y
end

function ref_get_m(opt::Refiner)
    m,n = size(opt.A)
    return m
end

function ref_get_n(opt::Refiner)
    m,n = size(opt.A)
    return n
end


# =============================================
#  3. Linear Programming Algorithms
# =============================================
function optimize!(opt::Refiner)
    A = opt.A
    b = opt.b
    c = opt.c
    l = opt.l
    verbose = opt.verbose
    alpha_p = opt.alpha_p
    alpha_d = opt.alpha_d
    itermax = opt.maxiter
    T = opt.T
    mode = opt.mode
    inner = opt.inner

    t = time()

    m,n = size(A)


    mode_dict = ["0 = solve iterative linear programming","1 = solving iterative linear system", "2 = solve high precision linear system"]

    if verbose
        println("using mode ", mode_dict[mode+1])
    end


    # iterative linear programming
    if mode == 0
        status,res,x,y,tp,ts,p_status,d_status = _iter_ref_lp(inner,A,b,c,l,T,verbose,alpha_p,alpha_d,itermax,0.0,0.0)
    # optimize with linear system
    else
        status,res,x,y,tp,ts,p_status,d_status = _iter_ref_ls(inner,A,b,c,l,T,verbose,alpha_p,alpha_d,itermax,mode,0.0,0.0)
    end

    opt.status = status
    opt.x = x
    opt.y = y
    opt.res = res

    solve_time = time() - t
    opt.solve_time = solve_time
    opt.solve_time_lp = tp
    opt.solve_time_ls = ts


    if verbose
        @show opt.solve_time
        @show opt.solve_time_lp
        @show opt.solve_time_ls
    end
end



# solve iterative linear programming
function _iter_ref_lp(
        #inner<:MOI.AbstractOptimizer,
        inner,
        A::AbstractArray{<:AbstractFloat,2},
        b::Array{<:AbstractFloat,1},
        c::Array{<:AbstractFloat,1},
        l::Array{<:AbstractFloat,1},
        T,
        verbose,
        alpha_p,
        alpha_d,
        iter_max,
        tp::Float64,
        ts::Float64)


        t = time()

        # check dimension
        m,n = size(A)
        x_star = zeros(T,n)
        y_star = zeros(T,m)
        rt = 0



        # initialize variables
        A_bar = Float64.(A)
        b_bar = Float64.(b)
        c_bar = Float64.(c)
        l_bar = Float64.(l)

        A = T.(A)
        b = T.(b)
        c = T.(c)
        l = T.(l)

        eps_primal = sqrt(eps(T))
        eps_dual = sqrt(eps(T))
        eps_slack = eps_primal*n*eps_dual

        coeff_primal = T(1.0)
        coeff_dual = T(1.0)

        opt_primal =  Model()
        set_optimizer(opt_primal, inner)
        opt_dual =  Model()
        set_optimizer(opt_dual, inner)

        # core computation
        for k = 1:iter_max

                empty!(opt_primal)
                empty!(opt_dual)

                #=
                Primal LP
                        max   cx
                        s.t.  Ax = b
                               x >= l
                =#

                @variable(opt_primal, x[1:n])
                @objective(opt_primal, Min, dot(c_bar,x) )
                @constraint(opt_primal, constraint, A_bar * x .== b_bar)
                @constraint(opt_primal, lower_bound, x.>=l_bar )
                JuMP.optimize!(opt_primal)

                tmp_t = time()

                if termination_status(opt_primal)==MOI.OPTIMAL || termination_status(opt_primal)==MOI.ALMOST_OPTIMAL
                        # feasible and optimal
                        x_prime = JuMP.value.(x)
                        x_star = x_star + (1/coeff_primal)*x_prime

                elseif termination_status(opt_primal)==MOI.INFEASIBLE || termination_status(opt_primal)==MOI.ALMOST_INFEASIBLE
                        # test infeasibility
                        A_test = [A b-A*l]
                        l_test = T.(zeros(n+1))
                        l_test[n+1] = -1
                        l_test[1:n] = l
                        c_test = zeros(n+1)
                        c_test[n+1] = 1
                        #tp += (time()-t)
                        test_status,test_rt,test_x,test_y,tp,ts = _iter_ref_lp(inner,A_test,b,c_test,l_test,T,false,alpha_p,alpha_d,iter_max,tp,ts)


                        # original LP feasible
                        if test_rt == -1.0
                                x_star = test_x[1:n] + l
                        # original LP feasible with tolerance eps/k, k=-test_rt
                        elseif test_rt >= -1 - eps_primal && test_rt < 0
                                x_star = -1/test_rt*test_x[1:n] + l
                        # original LP infeasible
                        else
                                tp += tmp_t - t
                                return termination_status(opt_primal),Inf,x_star,y_star,tp,ts,primal_status(opt_primal),dual_status(opt_primal)
                        end

                        # original LP unbounded
                elseif termination_status(opt_primal)==MOI.DUAL_INFEASIBLE || termination_status(opt_primal)==MOI.ALMOST_DUAL_INFEASIBLE

                        # TODO might need change here

                        x_prime = JuMP.value.(x)
                        x_star = x_star + (1/coeff_primal)*x_prime
                        tp += tmp_t - t
                        return termination_status(opt_primal),-Inf,-x_star,y_star,tp,ts,primal_status(opt_primal),dual_status(opt_primal)

                else
                        # other issue (e.g. numerical issue)
                        tp += tmp_t - t
                        return termination_status(opt_primal),0,x_star, y_star,tp,ts,primal_status(opt_primal),dual_status(opt_primal)
                end

                #=
                Dual LP
                        max   by - yAl
                        s.t.  Ay <= c
                =#

                @variable(opt_dual, y[1:m])
                @objective(opt_dual, Max, dot(b_bar,y)-dot(y,A_bar*l_bar) )
                @constraint(opt_dual, dual_constraint, A_bar' * y .<= c_bar)
                JuMP.optimize!(opt_dual)



                if termination_status(opt_dual)==MOI.OPTIMAL || termination_status(opt_dual)==MOI.ALMOST_OPTIMAL
                        # feasible and optimal
                        y_prime = JuMP.value.(y)
                        y_star = y_star + (1/coeff_dual)*y_prime
                else
                        # other issue (e.g. numerical issue)
                        tmp_t = time()
                        tp += (tmp_t - t)
                        return termination_status(opt_dual),0,x_star, y_star,tp,ts,primal_status(opt_primal),dual_status(opt_primal)
                end

                # optimal value
                rt = dot(c,x_star)

                # compute residue
                b_hat = b - A*x_star
                c_hat = c - A'*y_star
                l_hat = l - x_star

                # compute tolerance
                delta_primal = max(maximum(abs.(b_hat)),maximum(l_hat))
                delta_dual = max(0,maximum(-c_hat))
                delta_slack = dot(b,y_star)-dot(c,x_star)

                if verbose
                        @show k
                        @show rt
                        @show delta_primal
                        @show delta_dual
                        @show delta_slack

                end



                # termination check
                if delta_primal<=eps_primal &&  delta_dual<=eps_dual && delta_slack <= eps_slack
                        tmp_t = time()
                        tp += (tmp_t - t)
                        return termination_status(opt_primal),rt,x_star, y_star,tp,ts,primal_status(opt_primal),dual_status(opt_primal)
                end

                # update LP
                coeff_primal = min(1/delta_primal,alpha_p*coeff_primal)
                coeff_dual = min(1/delta_dual,alpha_d*coeff_dual)

                b_hat = coeff_primal * b_hat
                c_hat = coeff_dual * c_hat
                l_hat = coeff_primal * l_hat

                b_bar = Float64.(b_hat)
                c_bar = Float64.(c_hat)
                l_bar = Float64.(l_hat)
        end

        solve_time = time() - t
        tp += solve_time

        # need more iteration to meet termination standard
        @warn("More iterations needed")
        return termination_status(opt_primal),rt,x_star, y_star,tp,ts,primal_status(opt_primal),dual_status(opt_primal)

end


# optimize by guessing index and solving linear system
function _iter_ref_ls(
        #inner<:MOI.AbstractOptimizer,
        inner,
        A::AbstractArray{<:AbstractFloat,2},
        b::Array{<:AbstractFloat,1},
        c::Array{<:AbstractFloat,1},
        l::Array{<:AbstractFloat,1},
        T,
        verbose,
        alpha_p,
        alpha_d,
        maxiter,
        mode,
        tp::Float64,
        ts::Float64
        )


        t1 = time()

        m,n = size(A)

        x_star = zeros(T,n)
        y_star = zeros(T,m)

        A_bar = Float64.(A)
        b_bar = Float64.(b)
        c_bar = Float64.(c)
        l_bar = Float64.(l)

        A = T.(A)
        b = T.(b)
        c = T.(c)
        l = T.(l)

        ls = 1:n
        eps_primal = sqrt(eps(T))

        if verbose
            println("Solving Linear program to get index")
        end

        opt_dual =  Model()
        set_optimizer(opt_dual, inner)

        @variable(opt_dual, y[1:m])
        @objective(opt_dual, Max, dot(b_bar,y)-dot(y,A_bar*l_bar) )
        @constraint(opt_dual, dual_constraint, A_bar' * y .<= c_bar)
        JuMP.optimize!(opt_dual)


        if termination_status(opt_dual)==MOI.OPTIMAL || termination_status(opt_dual)==MOI.ALMOST_OPTIMAL
                # feasible and optimal
                y_prime = JuMP.value.(y)
        else
                # other issue (e.g. numerical issue)
                #return termination_status(opt_dual),0,x_star, y_star
                @warn("not able to solve under this mode. use iterative linear programming mode instead")
                return _iter_ref_lp(inner,A,b,c,l,T,verbose,alpha_p,alpha_d,maxiter,tp,ts)
        end

        t2 = time()

        tp += (t2-t1)

        err = abs.(A'*y_prime-c)
        tmp = sortperm(err)

        idx = tmp[1:m]

        if verbose
            println("Index found. Start Solving linear system")
        end

        A_prime = A[:,idx]


        # plenty choice of solving linear equation
        try
            if mode == 1
                delta = _solve_ls_iter(A,A_prime,b,l,x_star,eps_primal,maxiter,idx,eps_primal,verbose)
            elseif mode == 2
                delta = _solve_ls(A_prime,b,x_star,idx)
            end
        catch e
            # TODO need further fix for "singluar matrix"
            println(e)
            @warn("not able to solve under this mode. use iterative linear programming mode instead")
            return _iter_ref_lp(inner,A,b,c,l,T,verbose,alpha_p,alpha_d,maxiter,tp,ts)
        end


        ts += (time() - t2)


        # TODO need further fix. Will iterative linear programming under this condition?
        # if delta > eps_primal
        #     return _iter_ref_ls(inner,A,b,c,l,T,verbose,alpha_p,alpha_d,itermax,mode)
        # end


        res = dot(c,x_star)
        return termination_status(opt_dual),res,x_star,y_star,tp,ts,dual_status(opt_dual),primal_status(opt_dual)
end


function _solve_ls_iter(A,A_prime,b,l,x_star,eps,iter_max,idx,eps_primal,verbose)
    lu_bar = lu(Float64.(Matrix(A_prime)))
    r_prime = copy(b)
    delta_primal = 100
    for k = 1:iter_max

            x_star[idx] += (lu_bar\r_prime)

            b_hat = b - A*x_star
            l_hat = l - x_star



            delta_primal = max(maximum(-b_hat),maximum(l_hat))
            delta_primal = max(minimum(b_hat),delta_primal)

            if verbose
                @show k
                @show delta_primal
            end


            if delta_primal<=eps_primal
                    return delta_primal

            end


            r_prime = b_hat
    end

    @warn("need more iterations")
    return delta_primal
end

function _solve_ls(A,b,x_star,idx)
    x[idx] = A\b
end
