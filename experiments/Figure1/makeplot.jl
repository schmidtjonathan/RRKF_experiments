cd("./experiments/Figure1/")

using LinearAlgebra
using Distributions
using Random
using ForwardDiff
using ProgressMeter
using ToeplitzMatrices
using KernelFunctions
using StatsBase
using JLD2
using CairoMakie
using LaTeXStrings

using RRKF


function Base.:*(M::AbstractMatrix, C::Circulant)
	return (C' * Matrix(M'))'
end


unitvec(i, d) = Matrix{Float64}(I, d, d)[:, i]

Random.seed!(20171027)



function sines(dim_state, dof=9)
    num_sines = floor((dof-1)/2)
    s = zeros(dim_state)
    is = collect(1.0:1.0:float(dim_state))
    for k in 0.0:1.0:num_sines
        aₖ = rand(Uniform(0, 1))
        φₖ = rand(Uniform(0, 2π))
        s += aₖ .* sin.(2.0 * π * k .* is ./ float(dim_state) .+ φₖ)
    end
    return s
    ms = mean(s)
    s_c = s .- ms
    return s ./ sqrt(s_c' * s_c)
end



function filtering_setup_la(;
    dim_state,
    dim_obs,
    obs_locations = round.(Int, LinRange(1, dim_state, dim_obs)),
    observation_noise_std = sqrt(0.1),
    tspan = (0, 800),
    ENSEMBLE_SIZE,
    TIME_BETWEEN_OBS = 5,
    dof
)
    Random.seed!(142)

    times = tspan[1]:tspan[2]
    obs_times = tspan[1]+1:TIME_BETWEEN_OBS:tspan[2]

    @show size(obs_times) size(times)

    circ_shift_matrix = Circulant(unitvec(2, dim_state))
    A = circ_shift_matrix
    H = Matrix{Float64}(I(dim_state))[obs_locations, :]
    R = Matrix{Float64}(observation_noise_std^2 * I(dim_obs))

    ground_truth = [sines(dim_state, dof)]
    observations = []
    for i in times[2:end]
        push!(ground_truth, A * ground_truth[end])
        if i in obs_times
            push!(observations, rand(MvNormal(H * ground_truth[end], R)))
        end
    end

    μ₀ = ((sines(dim_state, dof) + (ground_truth[1] .- 4.0)) ./ sqrt(2.0)) .+ 4.0

    init_fullD_ensemble = vecvec2mat([μ₀ + sines(length(μ₀), dof) for i in 1:dim_state])'
    m0, P0 = RRKF.ensemble_mean_cov(init_fullD_ensemble)
    init_ensemble = init_fullD_ensemble[:, 1:ENSEMBLE_SIZE]

    observations = vecvec2mat(observations)
    ground_truth = vecvec2mat(ground_truth)

    @show size(observations) size(ground_truth) size(init_ensemble)
    @assert size(ground_truth, 2) == dim_state
    @assert size(observations, 2) == dim_obs

    return m0, P0, init_ensemble, A, H, R, ground_truth, observations, times, obs_times, obs_locations
end



function run_LA_kf(setup_tuple)
    m0, P0, init_ensemble, A, H, R, ground_truth, observations, times, obs_times, obs_locations = setup_tuple

    previous_t = times[1]
    m, P = copy(m0), copy(P0)

    filter_sol = RRKF.FilteringSolution(P)
    RRKF.append_step!(filter_sol, previous_t, m , P)

    obs_seen_counter = 0
    @showprogress 0.1 "KF" for t in times[2:end]
        dt = t - previous_t
        previous_t = t
        m, P = RRKF.kf_predict(m, P, A, zeros(size(A)))
        if t in obs_times
            obs_seen_counter = obs_seen_counter + 1
            m, P = RRKF.kf_correct(m, P, H, R, observations[obs_seen_counter, :])
        end
        RRKF.append_step!(filter_sol, t, m, P)
    end

    @assert obs_seen_counter == length(obs_times)

    return filter_sol
end



function run_LA_rrkf(setup_tuple; r)
    m0, P0, init_ensemble, A, H, R, ground_truth, observations, times, obs_times, obs_locations = setup_tuple

    previous_t = times[1]
    m = copy(m0)
    U_P, s_P, V_P = tsvd(P0, r)
    P_sqrt = RRKF.LeftMatrixSqrt(U_P * Diagonal(sqrt.(s_P)))
    R_sqrt = RRKF.LeftMatrixSqrt(cholesky(R))

    filter_sol = RRKF.FilteringSolution(P_sqrt)
    RRKF.append_step!(filter_sol, previous_t, m , P_sqrt)

    obs_seen_counter = 0
    @showprogress 0.1 "rrkf" for t in times[2:end]
        dt = t - previous_t
        previous_t = t

        # Predict
        m = A * m
        P_sqrt = RRKF.LeftMatrixSqrt(A * P_sqrt.factor)
        # Correct
        if t in obs_times
            obs_seen_counter = obs_seen_counter + 1
            m, P_sqrt, ll = RRKF.rrkf_correct(m, P_sqrt, H, R_sqrt, observations[obs_seen_counter, :], r)
        end
        RRKF.append_step!(filter_sol, t, m, P_sqrt)
    end

    @assert obs_seen_counter == length(obs_times)

    return filter_sol
end


function run_LA_enkf(setup_tuple)
    m0, P0, init_ensemble, A, H, R, ground_truth, observations, times, obs_times, obs_locations = setup_tuple
    r = size(init_ensemble, 2)

    previous_t = times[1]
    m = copy(m0)
    P = RRKF.LeftMatrixSqrt(RRKF.centered_ensemble(init_ensemble) / (r - 1))

    filter_sol = RRKF.FilteringSolution(P)
    RRKF.append_step!(filter_sol, previous_t, m , P)
    obs_noise_mvndist = MvNormal(R)

    obs_seen_counter = 0
    ensemble = copy(init_ensemble)
    @showprogress 0.1 "EnKF" for t in times[2:end]
        dt = t - previous_t
        previous_t = t
        ensemble = A * ensemble
        if t in obs_times
            obs_seen_counter = obs_seen_counter + 1
            ensemble, ll = RRKF.enkf_correct(ensemble, H, obs_noise_mvndist, observations[obs_seen_counter, :])
        end
        m, P = RRKF.ensemble_mean_sqrt_cov(ensemble)
        RRKF.append_step!(filter_sol, t, m, P)
    end

    @assert obs_seen_counter == length(obs_times)

    return filter_sol
end


function run_LA_etkf(setup_tuple)
    m0, P0, init_ensemble, A, H, R, ground_truth, observations, times, obs_times, obs_locations = setup_tuple
    r = size(init_ensemble, 2)

    previous_t = times[1]
    m = copy(m0)
    P = RRKF.LeftMatrixSqrt(RRKF.centered_ensemble(init_ensemble) / (r - 1))

    filter_sol = RRKF.FilteringSolution(P)
    RRKF.append_step!(filter_sol, previous_t, m , P)
    obs_noise_mvndist = MvNormal(R)

    obs_seen_counter = 0
    ensemble = copy(init_ensemble)
    @showprogress 0.1 "ETKF" for t in times[2:end]
        dt = t - previous_t
        previous_t = t
        ensemble = A * ensemble
        if t in obs_times
            obs_seen_counter = obs_seen_counter + 1
            ensemble, ll = RRKF.etkf_correct(ensemble, H, obs_noise_mvndist, observations[obs_seen_counter, :])
        end
        m, P = RRKF.ensemble_mean_sqrt_cov(ensemble)
        RRKF.append_step!(filter_sol, t, m, P)
    end

    @assert obs_seen_counter == length(obs_times)

    return filter_sol
end



DOF = 7
DOF_less = DOF - 4
DOF_more = DOF + 4

setup_tuple = filtering_setup_la(; ENSEMBLE_SIZE=DOF, dof=DOF, dim_state=1000, dim_obs=20);

# COMPUTE SOLUTIONS
kf_sol = run_LA_kf(setup_tuple);
rrkfrDOF_sol = run_LA_rrkf(setup_tuple; r=DOF);
rrkfrless_sol = run_LA_rrkf(setup_tuple; r=DOF_less);
rrkfrmore_sol = run_LA_rrkf(setup_tuple; r=DOF_more);

# EXTRACT FINAL COVARIANCES
kf_final_cov = Matrix(kf_sol.Σ[end]);
rrkfrDOF_final_cov = Matrix(rrkfrDOF_sol.Σ[end]);
rrkfrless_final_cov = Matrix(rrkfrless_sol.Σ[end]);
rrkfrmore_final_cov = Matrix(rrkfrmore_sol.Σ[end]);

# EXTRACT FINAL COVARIANCE FACTORS
rrkfrDOF_final_cov_factor = Matrix(rrkfrDOF_sol.Σ[end].factor);
rrkfrless_final_cov_factor = Matrix(rrkfrless_sol.Σ[end].factor);
rrkfrmore_final_cov_factor = Matrix(rrkfrmore_sol.Σ[end].factor);



include("../plot_theme.jl")


WIDTH, HEIGHT = FULL_WIDTH, 0.9HALF_HEIGHT

DOWNSAMPLING_FACTOR = 10  # To prevent the size of the PDF file from getting too large


mycgrad_cov = :vik
mycgrad_factor = :vik

fig = begin
    grid_plot = Figure(;
        figure_padding=(5, 5, 10, 10),
        resolution=(WIDTH, HEIGHT),
    )
    kf_cov_axis = Axis(grid_plot[2, 1], aspect=1.0, yreversed=true, topspinevisible=true, rightspinevisible=true, leftspinevisible=true, bottomspinevisible=true,)
    heatmap!(kf_cov_axis, kf_final_cov[begin:DOWNSAMPLING_FACTOR:end, begin:DOWNSAMPLING_FACTOR:end], colormap=mycgrad_cov, )
    hidedecorations!(kf_cov_axis)


    # LESS

    rrkfrless_cov_axis = Axis(grid_plot[2, 3], aspect=1.0, yreversed=true, topspinevisible=true, rightspinevisible=true, leftspinevisible=true, bottomspinevisible=true,)
    heatmap!(rrkfrless_cov_axis, rrkfrless_final_cov[begin:DOWNSAMPLING_FACTOR:end, begin:DOWNSAMPLING_FACTOR:end], colormap=mycgrad_cov, )
    hidedecorations!(rrkfrless_cov_axis)

    rrkfrless_factor_left_axis = Axis(grid_plot[2, 2], aspect=1/7.5, yreversed=true, halign=:right, topspinevisible=true, rightspinevisible=true, leftspinevisible=true, bottomspinevisible=true,)
    heatmap!(rrkfrless_factor_left_axis, rrkfrless_final_cov_factor[begin:DOWNSAMPLING_FACTOR:end, :]', colormap=mycgrad_factor, colorrange=(-0.015, 0.015))
    hidedecorations!(rrkfrless_factor_left_axis)

    rrkfrless_factor_right_axis = Axis(grid_plot[1, 3], aspect=8, yreversed=true, topspinevisible=true, rightspinevisible=true, leftspinevisible=true, bottomspinevisible=true, valign=:top)
    heatmap!(rrkfrless_factor_right_axis, rrkfrless_final_cov_factor[begin:DOWNSAMPLING_FACTOR:end, :], colormap=mycgrad_factor, colorrange=(-0.015, 0.015))
    hidedecorations!(rrkfrless_factor_right_axis)



    # DOF

    rrkfrDOF_cov_axis = Axis(grid_plot[2, 5], aspect=1.0, yreversed=true, topspinevisible=true, rightspinevisible=true, leftspinevisible=true, bottomspinevisible=true,)
    heatmap!(rrkfrDOF_cov_axis, rrkfrDOF_final_cov[begin:DOWNSAMPLING_FACTOR:end, begin:DOWNSAMPLING_FACTOR:end], colormap=mycgrad_cov, )
    hidedecorations!(rrkfrDOF_cov_axis)

    rrkfrDOF_factor_left_axis = Axis(grid_plot[2, 4], aspect=1/5.6, yreversed=true, halign=:right, topspinevisible=true, rightspinevisible=true, leftspinevisible=true, bottomspinevisible=true,)
    factor_heatmap = heatmap!(rrkfrDOF_factor_left_axis, rrkfrDOF_final_cov_factor[begin:DOWNSAMPLING_FACTOR:end, :]', colormap=mycgrad_factor, colorrange=(-0.015, 0.015))
    hidedecorations!(rrkfrDOF_factor_left_axis)

    rrkfrDOF_factor_right_axis = Axis(grid_plot[1, 5], aspect=6, yreversed=true, topspinevisible=true, rightspinevisible=true, leftspinevisible=true, bottomspinevisible=true, valign=:top)
    heatmap!(rrkfrDOF_factor_right_axis, rrkfrDOF_final_cov_factor[begin:DOWNSAMPLING_FACTOR:end, :], colormap=mycgrad_factor, colorrange=(-0.015, 0.015))
    hidedecorations!(rrkfrDOF_factor_right_axis)



    # MORE

    rrkfrmore_cov_axis = Axis(grid_plot[2, 7], aspect=1.0, yreversed=true, topspinevisible=true, rightspinevisible=true, leftspinevisible=true, bottomspinevisible=true,)
    heatmap!(rrkfrmore_cov_axis, rrkfrmore_final_cov[begin:DOWNSAMPLING_FACTOR:end, begin:DOWNSAMPLING_FACTOR:end], colormap=mycgrad_cov, )
    hidedecorations!(rrkfrmore_cov_axis)

    rrkfrmore_factor_left_axis = Axis(
        grid_plot[2, 6],
        aspect=1/3.5, yreversed=true, halign=:right, topspinevisible=true, rightspinevisible=true,
        leftspinevisible=true, bottomspinevisible=true,
    xticks=[DOF],xticklabelspace=0.0
        )
    heatmap!(rrkfrmore_factor_left_axis, rrkfrmore_final_cov_factor[begin:DOWNSAMPLING_FACTOR:end, :]', colormap=mycgrad_factor, colorrange=(-0.015, 0.015), valign=:top)
    vlines!(rrkfrmore_factor_left_axis, [DOF], linewidth=0.5, color=:black, linestyle=:dash)
    hideydecorations!(rrkfrmore_factor_left_axis)
    hidexdecorations!(rrkfrmore_factor_left_axis, ticks=false, ticklabels=false)



    rrkfrmore_factor_right_axis = Axis(
        grid_plot[1, 7], aspect=4,
        yreversed=true, valign=:top,
        topspinevisible=true, rightspinevisible=true,
        leftspinevisible=true, bottomspinevisible=true, xticks=[8],
        )
    heatmap!(rrkfrmore_factor_right_axis, rrkfrmore_final_cov_factor[begin:DOWNSAMPLING_FACTOR:end, :], colormap=mycgrad_factor, colorrange=(-0.015, 0.015))
    hlines!(rrkfrmore_factor_right_axis, [DOF], linewidth=0.5, color=:black, linestyle=:dash)
    hidedecorations!(rrkfrmore_factor_right_axis)

    Label(grid_plot[3, 1, Bottom()], "Kalman filter", padding=(0, 0, 0, -3), fontsize=BASE_FONTSIZE - 1, font="Times New Roman regular")
    Label(grid_plot[3, 3, Bottom()], L"\text{ours:} r = %$(DOF_less)", padding=(0, 0, 0, -3), fontsize=BASE_FONTSIZE - 1, font="Times New Roman regular")
    Label(grid_plot[3, 5, Bottom()], L"\text{ours:} r = %$(DOF)", padding=(0, 0, 0, -3), fontsize=BASE_FONTSIZE - 1, font="Times New Roman regular")
    Label(grid_plot[3, 7, Bottom()], L"\text{ours:} r = %$(DOF_more)", padding=(0, 0, 0, -3), fontsize=BASE_FONTSIZE - 1, font="Times New Roman regular")
    Colorbar(grid_plot[2, 8], factor_heatmap, ticks=[0], size=6, ticksize=3)

    rowsize!(grid_plot.layout, 1, Aspect(1, 0.31))
    rowsize!(grid_plot.layout, 2, Aspect(1, 1.0))
    rowsize!(grid_plot.layout, 3, Aspect(1, 0.1))
    colsize!(grid_plot.layout, 2, Relative(0.071))
    colsize!(grid_plot.layout, 4, Relative(0.07))
    colsize!(grid_plot.layout, 6, Relative(0.07))

    colgap!(grid_plot.layout, 0.0)
    colgap!(grid_plot.layout, 1, 7.0)
    colgap!(grid_plot.layout, 3, 7.0)
    colgap!(grid_plot.layout, 5, 10.0)
    colgap!(grid_plot.layout, 7, 5.0)
    rowgap!(grid_plot.layout, 1.0)

    grid_plot
end

save("fig1.pdf", fig)