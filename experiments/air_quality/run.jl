cd("./experiments/air_quality/")

using Revise

using LinearAlgebra
using Random
using Plots
using Distributions
using KernelFunctions
using Statistics
using StatsBase
using SparseArrays
using JLD2
using RRKF


nanmean(x) = mean(Base.filter(!isnan, x))
nanrmse(a, b) = sqrt(nanmean((a - b).^2))


# ##############################################################
# SETUP
# ##############################################################


# https://github.com/AaltoML/spatio-temporal-GPs/blob/c5b929e1fc07b14ff9671dd1d66b3b8041e2a2ce/experiments/air_quality/setup_data.py#L209
const TIME_AXIS = 1
const LATITUDE_AXIS = 2
const LONGITUDE_AXIS = 3

include("load_data.jl")

times, grid_lons, grid_lats, grid_lons_raw, grid_lats_raw, raw_Y, Y_train, Y_test, mean_Y, std_Y, min_Y, max_Y = load_london_data();


# Spatial Kernel
K(XY_loc, spatial_kernelfun, ℓₓ) = kernelmatrix(with_lengthscale(spatial_kernelfun(), ℓₓ), XY_loc) + 1e-8I;


proj(d, ν, q) = sparse(RRKF.projectionmatrix(d, ν, q))


sqrtkf_alg = RRKF.SqrtKalmanFilter()

lonlats_as_vec = collect(map(tuple2vec, (zip(grid_lons, grid_lats))))

HPARAMS = load("./out/hparams.jld2")


# --------- BUILD SSM
H_all = proj(length(grid_lats), HPARAMS["temporal_smoothness"], 0)

ν = HPARAMS["temporal_smoothness"]
dim = length(Y_train[1])
@assert dim == size(H_all, 1)

R = HPARAMS["σᵣ"]^2 * Diagonal(ones(dim))
spatial_K = K(lonlats_as_vec, HPARAMS["spatial_kernelfun"], HPARAMS["ℓxy"])
sqrt_diffusion = RRKF.LeftMatrixSqrt(cholesky(spatial_K))

prior_dynamics = RRKF.build_spatiotemporal_matern_process(
    ν,
    HPARAMS["ℓₜ"],
    HPARAMS["σₜ"],
    sqrt_diffusion,
)

ssm = RRKF.StateSpaceModel(
    prior_dynamics, H_all, R
)


# --------------------

sqrtkf_estim_stats = @timed RRKF.estimate_states(
    sqrtkf_alg,
    ssm,
    times,
    Y_train;
    smooth=true,
    save_all_steps=true,
    show_progress=true,
    compute_likelihood=true,
)

sqrtkf_estim = sqrtkf_estim_stats.value

LOW_RANK_DIM = 20
@info "R" r=LOW_RANK_DIM

NUM_DLR_STEPS = 1
rrkf_estim_stats = @timed RRKF.estimate_states(
    RRKF.RankReducedKalmanFilter(LOW_RANK_DIM, NUM_DLR_STEPS),
    ssm,
    times,
    Y_train;
    smooth=true,
    save_all_steps=true,
    show_progress=true,
    compute_likelihood=true,
)
rrkf_estim = rrkf_estim_stats.value

@info "RMSE RRKF vs. KF" rmsd(RRKF.means(sqrtkf_estim[:filter]), RRKF.means(rrkf_estim[:filter]))

enkf_estim_stats = @timed RRKF.estimate_states(
    RRKF.EnsembleKalmanFilter(LOW_RANK_DIM, RRKF.enkf_correct),
    ssm,
    times,
    Y_train;
    smooth=false,
    save_all_steps=true,
    show_progress=true,
    compute_likelihood=true,
)
enkf_estim = enkf_estim_stats.value

@info "RMSE EnKF vs. KF" rmsd(RRKF.means(enkf_estim[:filter]), RRKF.means(rrkf_estim[:filter]))



etkf_estim_stats = @timed RRKF.estimate_states(
    RRKF.EnsembleKalmanFilter(LOW_RANK_DIM, RRKF.etkf_correct),
    ssm,
    times,
    Y_train;
    smooth=false,
    save_all_steps=true,
    show_progress=true,
    compute_likelihood=false,
)
etkf_estim = etkf_estim_stats.value


@info "RMSE ETKF vs. KF" rmsd(RRKF.means(etkf_estim[:filter]), RRKF.means(rrkf_estim[:filter]))


sqrtkf_filter_means = RRKF.means(sqrtkf_estim[:filter]) * H_all';
sqrtkf_filter_stds = RRKF.stds(sqrtkf_estim[:filter]) * H_all';

kf_rmse_to_test = nanrmse(sqrtkf_filter_means .* std_Y .+ mean_Y, vecvec2mat(Y_test) .* std_Y .+ mean_Y)


rrkf_filter_means = RRKF.means(rrkf_estim[:filter]) * H_all';
rrkf_filter_stds = RRKF.stds(rrkf_estim[:filter]) * H_all';
rrkf_filter_rmse = nanrmse(rrkf_filter_means .* std_Y .+ mean_Y, vecvec2mat(Y_test) .* std_Y .+ mean_Y)


enkf_means = RRKF.means(enkf_estim[:filter]) * H_all';
enkf_stds = RRKF.stds(enkf_estim[:filter]) * H_all';
enkf_rmse = nanrmse(enkf_means .* std_Y .+ mean_Y, vecvec2mat(Y_test) .* std_Y .+ mean_Y)



nval_list = LinRange(10, size(H_all, 2), 9)
nval_list = unique(sort(round.(Int64, nval_list)))



rrkf_rmse_to_test_per_nval = Float64[]
rrkf_rmse_to_kf_per_nval = Float64[]
rrkf_cov_distance_per_nval = Float64[]
enkf_rmse_to_test_per_nval = Vector{Float64}[]
enkf_rmse_to_kf_per_nval = Vector{Float64}[]
enkf_cov_distance_per_nval = Vector{Float64}[]
etkf_rmse_to_test_per_nval = Vector{Float64}[]
etkf_rmse_to_kf_per_nval = Vector{Float64}[]
etkf_cov_distance_per_nval = Vector{Float64}[]

for cur_nval in nval_list
    @info "R = $cur_nval"

    cur_rrkf_estim = RRKF.estimate_states(
        RRKF.RankReducedKalmanFilter(cur_nval, NUM_DLR_STEPS),
        ssm,
        times,
        Y_train;
        smooth=false,
        save_all_steps=true,
        show_progress=true,
        compute_likelihood=false,
    )
    cur_rrkf_sol = cur_rrkf_estim[:filter]
    cur_rrkf_means = RRKF.means(cur_rrkf_sol) * H_all';

    cur_rrkf_rmse_to_test = nanrmse(cur_rrkf_means .* std_Y .+ mean_Y, vecvec2mat(Y_test) .* std_Y .+ mean_Y)
    cur_rrkf_rmse_to_kf = rmsd(cur_rrkf_means .* std_Y .+ mean_Y, sqrtkf_filter_means .* std_Y .+ mean_Y)
    cur_rrkf_cov_distance = mean([norm(Matrix(SC) - Matrix(KC)) / norm(Matrix(KC)) for (SC, KC) in zip(cur_rrkf_sol.Σ, sqrtkf_estim[:filter].Σ)])

    push!(rrkf_rmse_to_test_per_nval, cur_rrkf_rmse_to_test)
    push!(rrkf_rmse_to_kf_per_nval, cur_rrkf_rmse_to_kf)
    push!(rrkf_cov_distance_per_nval, cur_rrkf_cov_distance)


    cur_enkf_rmse_to_test_per_enkf_loop = Float64[]
    cur_enkf_rmse_to_kf_per_enkf_loop = Float64[]
    cur_enkf_cov_distance_per_enkf_loop = Float64[]

    cur_etkf_rmse_to_test_per_etkf_loop = Float64[]
    cur_etkf_rmse_to_kf_per_etkf_loop = Float64[]
    cur_etkf_cov_distance_per_etkf_loop = Float64[]
    for enkf_loop in 1:2 # 20

        cur_enkf_estim = RRKF.estimate_states(
            RRKF.EnsembleKalmanFilter(cur_nval, RRKF.enkf_correct),
            ssm,
            times,
            Y_train;
            smooth=false,
            save_all_steps=true,
            show_progress=true,
            compute_likelihood=false,
        )
        cur_enkf_sol = cur_enkf_estim[:filter]
        cur_enkf_means = RRKF.means(cur_enkf_sol) * H_all'

        cur_enkf_rmse_to_test = nanrmse(cur_enkf_means .* std_Y .+ mean_Y, vecvec2mat(Y_test) .* std_Y .+ mean_Y)
        cur_enkf_rmse_to_kf = rmsd(cur_enkf_means .* std_Y .+ mean_Y, sqrtkf_filter_means .* std_Y .+ mean_Y)
        cur_enkf_cov_distance = mean([norm(Matrix(SC) - Matrix(KC)) / norm(Matrix(KC)) for (SC, KC) in zip(cur_enkf_sol.Σ, sqrtkf_estim[:filter].Σ)])

        push!(cur_enkf_rmse_to_test_per_enkf_loop, cur_enkf_rmse_to_test)
        push!(cur_enkf_rmse_to_kf_per_enkf_loop, cur_enkf_rmse_to_kf)
        push!(cur_enkf_cov_distance_per_enkf_loop, cur_enkf_cov_distance)

        cur_etkf_estim = RRKF.estimate_states(
            RRKF.EnsembleKalmanFilter(cur_nval, RRKF.etkf_correct),
            ssm,
            times,
            Y_train;
            smooth=false,
            save_all_steps=true,
            show_progress=true,
            compute_likelihood=false,
        )
        cur_etkf_sol = cur_etkf_estim[:filter]
        cur_etkf_means = RRKF.means(cur_etkf_sol) * H_all'

        cur_etkf_rmse_to_test = nanrmse(cur_etkf_means .* std_Y .+ mean_Y, vecvec2mat(Y_test) .* std_Y .+ mean_Y)
        cur_etkf_rmse_to_kf = rmsd(cur_etkf_means .* std_Y .+ mean_Y, sqrtkf_filter_means .* std_Y .+ mean_Y)
        cur_etkf_cov_distance = mean([norm(Matrix(SC) - Matrix(KC)) / norm(Matrix(KC)) for (SC, KC) in zip(cur_etkf_sol.Σ, sqrtkf_estim[:filter].Σ)])

        push!(cur_etkf_rmse_to_test_per_etkf_loop, cur_etkf_rmse_to_test)
        push!(cur_etkf_rmse_to_kf_per_etkf_loop, cur_etkf_rmse_to_kf)
        push!(cur_etkf_cov_distance_per_etkf_loop, cur_etkf_cov_distance)
    end

    push!(enkf_rmse_to_test_per_nval, cur_enkf_rmse_to_test_per_enkf_loop)
    push!(enkf_rmse_to_kf_per_nval, cur_enkf_rmse_to_kf_per_enkf_loop)
    push!(enkf_cov_distance_per_nval, cur_enkf_cov_distance_per_enkf_loop)

    push!(etkf_rmse_to_test_per_nval, cur_etkf_rmse_to_test_per_etkf_loop)
    push!(etkf_rmse_to_kf_per_nval, cur_etkf_rmse_to_kf_per_etkf_loop)
    push!(etkf_cov_distance_per_nval, cur_etkf_cov_distance_per_etkf_loop)

end


save(
    "./out/london_error_data.jld2",
    Dict(
        "nval_list" => float(nval_list),
        "rrkf" => Dict(
            "rmse_to_test" => rrkf_rmse_to_test_per_nval,
            "rmse_to_kf" => rrkf_rmse_to_kf_per_nval,
            "cov_distance" => rrkf_cov_distance_per_nval,
        ),
        "enkf" => Dict(
            "rmse_to_test" => enkf_rmse_to_test_per_nval,
            "rmse_to_kf" => enkf_rmse_to_kf_per_nval,
            "cov_distance" => enkf_cov_distance_per_nval,
        ),
        "etkf" => Dict(
            "rmse_to_test" => etkf_rmse_to_test_per_nval,
            "rmse_to_kf" => etkf_rmse_to_kf_per_nval,
            "cov_distance" => etkf_cov_distance_per_nval,
        ),
        "kf" => Dict(
            "rmse_to_test" => kf_rmse_to_test,
        ),
    )
)

