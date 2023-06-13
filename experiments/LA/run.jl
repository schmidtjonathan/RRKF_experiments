cd("./experiments/LA")

using LinearAlgebra
using Distributions
using Random
using ForwardDiff
using ProgressMeter
using KernelFunctions
using StatsBase
using ToeplitzMatrices
using JLD2

using RRKF


function Base.:*(M::AbstractMatrix, C::Circulant)
	return (C' * Matrix(M'))'
end


unitvec(i, d) = Matrix{Float64}(I, d, d)[:, i]

Random.seed!(20171027)


function sines(dim_state)
    s = zeros(dim_state)
    is = collect(1.0:1.0:float(dim_state))
    for k in 0.0:1.0:25.0
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
    dim_state=1024,
    dim_obs = 10,
    obs_locations = round.(Int, LinRange(1, dim_state, dim_obs)),
    observation_noise_std = sqrt(0.1),
    tspan = (0, 800),
    ENSEMBLE_SIZE = 50,
    TIME_BETWEEN_OBS = 5,
    rdseed=124,
)
    Random.seed!(rdseed)

    times = tspan[1]:tspan[2]
    obs_times = tspan[1]+1:TIME_BETWEEN_OBS:tspan[2]

    @show size(obs_times) size(times)

    circ_shift_matrix = Circulant(unitvec(2, dim_state))
    A = circ_shift_matrix
    H = Matrix{Float64}(I(dim_state))[obs_locations, :]
    R = Matrix{Float64}(observation_noise_std^2 * I(dim_obs))

    ground_truth = [sines(dim_state)]
    observations = []
    for i in times[2:end]
        push!(ground_truth, A * ground_truth[end])
        if i in obs_times
            push!(observations, rand(MvNormal(H * ground_truth[end], R)))
        end
    end

    μ₀ = ((sines(dim_state) + (ground_truth[1] .- 4.0)) ./ sqrt(2.0)) .+ 4.0

    init_fullD_ensemble = vecvec2mat([μ₀ + sines(length(μ₀)) for i in 1:dim_state])'
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



function run_LA_rrkf(setup_tuple; r, num_dlr_steps)
    m0, P0, init_ensemble, A, H, R, ground_truth, observations, times, obs_times, obs_locations = setup_tuple

    previous_t = times[1]
    m = copy(m0)
    U_P, s_P, V_P = tsvd(P0, r; mode=:manualhighest)
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


# ERROR VS NVAL
nval_list = LinRange(2, 100, 9)
nval_list = unique(sort(ceil.(Int64, nval_list)))

rrkf_rmse_to_truth_per_nval = Float64[]
rrkf_rmse_to_kf_per_nval = Float64[]
rrkf_cov_distance_per_nval = Float64[]
enkf_rmse_to_truth_per_nval = Vector{Float64}[]
enkf_rmse_to_kf_per_nval = Vector{Float64}[]
enkf_cov_distance_per_nval = Vector{Float64}[]
etkf_rmse_to_truth_per_nval = Vector{Float64}[]
etkf_rmse_to_kf_per_nval = Vector{Float64}[]
etkf_cov_distance_per_nval = Vector{Float64}[]

kf_rmse_to_truth_per_nval = Float64[]

for cur_nval in nval_list
    @info "R = $cur_nval"
    current_setup = filtering_setup_la(ENSEMBLE_SIZE=cur_nval)
    m0, P0, init_ensemble, A, H, R, ground_truth, observations, times, obs_times, obs_locations = current_setup

    cur_kf_sol = run_LA_kf(current_setup)
    cur_kf_rmse_to_truth = rmsd(RRKF.means(cur_kf_sol), ground_truth)
    push!(kf_rmse_to_truth_per_nval, cur_kf_rmse_to_truth)

    cur_rrkf_sol = run_LA_rrkf(current_setup; r=cur_nval, num_dlr_steps=1)
    cur_rrkf_means = RRKF.means(cur_rrkf_sol);

    cur_rrkf_rmse_to_truth = rmsd(cur_rrkf_means, ground_truth)
    cur_rrkf_rmse_to_kf = rmsd(cur_rrkf_means, RRKF.means(cur_kf_sol))
    cur_rrkf_cov_distance = mean([norm(Matrix(SC) - Matrix(KC)) / norm(Matrix(KC)) for (SC, KC) in zip(cur_rrkf_sol.Σ, cur_kf_sol.Σ)])

    push!(rrkf_rmse_to_truth_per_nval, cur_rrkf_rmse_to_truth)
    push!(rrkf_rmse_to_kf_per_nval, cur_rrkf_rmse_to_kf)
    push!(rrkf_cov_distance_per_nval, cur_rrkf_cov_distance)

    cur_enkf_rmse_to_truth_per_enkf_loop = Float64[]
    cur_enkf_rmse_to_kf_per_enkf_loop = Float64[]
    cur_enkf_cov_distance_per_enkf_loop = Float64[]

    cur_etkf_rmse_to_truth_per_etkf_loop = Float64[]
    cur_etkf_rmse_to_kf_per_etkf_loop = Float64[]
    cur_etkf_cov_distance_per_etkf_loop = Float64[]
    for enkf_loop in 1:20
        cur_enkf_sol = run_LA_enkf(current_setup)
        cur_enkf_means = RRKF.means(cur_enkf_sol)

        cur_enkf_rmse_to_truth = rmsd(cur_enkf_means, ground_truth)
        cur_enkf_rmse_to_kf = rmsd(cur_enkf_means, RRKF.means(cur_kf_sol))
        cur_enkf_cov_distance = mean([norm(Matrix(SC) - Matrix(KC)) / norm(Matrix(KC)) for (SC, KC) in zip(cur_enkf_sol.Σ, cur_kf_sol.Σ)])

        push!(cur_enkf_rmse_to_truth_per_enkf_loop, cur_enkf_rmse_to_truth)
        push!(cur_enkf_rmse_to_kf_per_enkf_loop, cur_enkf_rmse_to_kf)
        push!(cur_enkf_cov_distance_per_enkf_loop, cur_enkf_cov_distance)


        cur_etkf_sol = run_LA_etkf(current_setup)
        cur_etkf_means = RRKF.means(cur_etkf_sol)

        cur_etkf_rmse_to_truth = rmsd(cur_etkf_means, ground_truth)
        cur_etkf_rmse_to_kf = rmsd(cur_etkf_means, RRKF.means(cur_kf_sol))
        cur_etkf_cov_distance = mean([norm(Matrix(SC) - Matrix(KC)) / norm(Matrix(KC)) for (SC, KC) in zip(cur_etkf_sol.Σ, cur_kf_sol.Σ)])

        push!(cur_etkf_rmse_to_truth_per_etkf_loop, cur_etkf_rmse_to_truth)
        push!(cur_etkf_rmse_to_kf_per_etkf_loop, cur_etkf_rmse_to_kf)
        push!(cur_etkf_cov_distance_per_etkf_loop, cur_etkf_cov_distance)

    end
    push!(enkf_rmse_to_truth_per_nval, cur_enkf_rmse_to_truth_per_enkf_loop)
    push!(enkf_rmse_to_kf_per_nval, cur_enkf_rmse_to_kf_per_enkf_loop)
    push!(enkf_cov_distance_per_nval, cur_enkf_cov_distance_per_enkf_loop)
    push!(etkf_rmse_to_truth_per_nval, cur_etkf_rmse_to_truth_per_etkf_loop)
    push!(etkf_rmse_to_kf_per_nval, cur_etkf_rmse_to_kf_per_etkf_loop)
    push!(etkf_cov_distance_per_nval, cur_etkf_cov_distance_per_etkf_loop)

end


save(
    "./out/LA_error_data-20_enkf_runs.jld2",
    Dict(
        "nval_list" => float(nval_list),
        "rrkf" => Dict(
            "rmse_to_truth" => rrkf_rmse_to_truth_per_nval,
            "rmse_to_kf" => rrkf_rmse_to_kf_per_nval,
            "cov_distance" => rrkf_cov_distance_per_nval,
        ),
        "enkf" => Dict(
            "rmse_to_truth" => enkf_rmse_to_truth_per_nval,
            "rmse_to_kf" => enkf_rmse_to_kf_per_nval,
            "cov_distance" => enkf_cov_distance_per_nval,
        ),
        "etkf" => Dict(
            "rmse_to_truth" => etkf_rmse_to_truth_per_nval,
            "rmse_to_kf" => etkf_rmse_to_kf_per_nval,
            "cov_distance" => etkf_cov_distance_per_nval,
        ),
        "kf" => Dict(
            "rmse_to_truth" => kf_rmse_to_truth_per_nval,
        ),
    )
)
