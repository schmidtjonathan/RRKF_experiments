cd("./experiments/on-model-matern/")


using Revise

using LinearAlgebra
using Random
using CairoMakie
using Distributions
using KernelFunctions
using Statistics
using StatsBase
using SparseArrays
using JLD2
using RRKF


include("plotting.jl")

nanmean(x) = mean(Base.filter(!isnan, x))
nanrmse(a, b) = sqrt(nanmean((a - b).^2))

function make_2d_grid(x1_range, x2_range)
    reshape(
        map(tuple2vec, collect(Iterators.product(x1_range, x2_range))),
        length(x1_range),
        length(x2_range),
    )
end

function train_test_split_indices(N; split=0.5, seed=0)
	Random.seed!(seed)
    rand_index = Random.randperm(N)

    N_tr =  ceil(Int64, N * split)

    return sort(rand_index[1:N_tr]), sort(rand_index[N_tr+1:end])
end



function simulate_linear_ssm(
    ssm::RRKF.StateSpaceModel, simulation_grid, init_t
)

	μ₀, Σ₀ = RRKF.stationary_moments(ssm.dynamics)

	trajectory = Vector{Float64}[]
    measurements = Vector{Float64}[]
    x = rand(MvNormal(μ₀, Symmetric(Σ₀)))

	prev_t = init_t
    for t in simulation_grid
        dt = t - prev_t
		A, Q = RRKF.discretize(ssm.dynamics, dt)
        x = rand(MvNormal(A * x, Symmetric(Q)))
		push!(trajectory, x)
        push!(measurements, rand(MvNormal(ssm.H * trajectory[end], ssm.R)))
		prev_t = t
    end
    return trajectory, measurements
end




function filtering_setup_spatiotemporal_matern32(;
    dx = 0.1,
	N_x_1d = 21,
	ν = 1/2,
	dt = 0.1,
	ℓ_x = 0.7,
	σ_r = 2.0,
	N_t = 100,
)

    ℓ_t = 1.0
    σ_t = 1.0

    X_grid_1d = collect(0.0:dx:(N_x_1d - 1) * dx)
    X_grid_2d = make_2d_grid(X_grid_1d, X_grid_1d)
    d = size(X_grid_2d, 1) * size(X_grid_2d, 2)
    D = d * round(Int, ν + 0.5)

    simulation_grid = dt:dt:N_t * dt

    K(ℓₓ, σₓ = 1.0) = kernelmatrix(with_lengthscale(Matern32Kernel(), ℓₓ), vec(X_grid_2d)) + 1e-7I;

    sqrt_diffusion = RRKF.LeftMatrixSqrt(cholesky(K(ℓ_x)))

	true_dynamics = RRKF.build_spatiotemporal_matern_process(
	    ν,
	    ℓ_t,
	    σ_t,
	    sqrt_diffusion,
	)

    proj(q, ν) = RRKF.projectionmatrix(d, ν, q)

    H = proj(0, ν);
    R = Diagonal(σ_r * ones(d));
    ssm = RRKF.StateSpaceModel(true_dynamics, H, R)

    prior_draw, observations = simulate_linear_ssm(
        ssm, simulation_grid, 0.0
    )

    ground_truth = [H * s for s in prior_draw]

    train_observations = deepcopy(observations);
    test_observations = deepcopy(observations);

    for i_y in 1:length(observations)
        cur_train_idcs, cur_test_idcs = train_test_split_indices(length(observations[i_y]); split=0.5, seed=i_y)
        train_observations[i_y][cur_test_idcs] .= NaN64
        test_observations[i_y][cur_train_idcs] .= NaN64
    end

    return ssm, ground_truth, train_observations, test_observations, simulation_grid
end

mock_ssm, = filtering_setup_spatiotemporal_matern32()

const nval_list = unique(sort(round.(Int64, LinRange(10, size(mock_ssm.H , 2), 10))))
const ℓₓ_list = [0.01, 0.1, 0.25, 1.0]


function evaluation_for_spatial_lengthscale(ℓₓ)
    Random.seed!(1234)

    ssm, ground_truth, train_observations, test_observations, simulation_grid = filtering_setup_spatiotemporal_matern32(ℓ_x = ℓₓ)

    kf_estimate = RRKF.estimate_states(
        RRKF.SqrtKalmanFilter(),
        ssm,
        simulation_grid,
        train_observations,
        smooth=false,
        compute_likelihood=false,
        save_all_steps=true,
        show_progress=true
    )
    kf_sol = kf_estimate[:filter]

    stacked_test_observations = vecvec2mat(test_observations)
    stacked_ground_truth = vecvec2mat(ground_truth)

    kf_means = RRKF.means(kf_sol)
    kf_rmse_to_test = nanrmse(kf_means * ssm.H', stacked_test_observations)
    kf_rmse_to_truth = rmsd(kf_means * ssm.H', stacked_ground_truth)


    spectrum_of_kf_cov = eigvals(Matrix(kf_sol.Σ[end]), sortby=l -> -l)
    _, _QU = RRKF.discretize(ssm.dynamics, mean(diff(simulation_grid)))
    spectrum_of_process_noise_cov = eigvals(_QU, sortby=l -> -l)


    # ERROR VS NVAL
    rrkf_rmse_to_test_per_nval = Float64[]
    rrkf_rmse_to_truth_per_nval = Float64[]
    rrkf_rmse_to_kf_per_nval = Float64[]
    rrkf_cov_distance_per_nval = Float64[]
    enkf_rmse_to_test_per_nval = Vector{Float64}[]
    enkf_rmse_to_truth_per_nval = Vector{Float64}[]
    enkf_rmse_to_kf_per_nval = Vector{Float64}[]
    enkf_cov_distance_per_nval = Vector{Float64}[]
    etkf_rmse_to_test_per_nval = Vector{Float64}[]
    etkf_rmse_to_truth_per_nval = Vector{Float64}[]
    etkf_rmse_to_kf_per_nval = Vector{Float64}[]
    etkf_cov_distance_per_nval = Vector{Float64}[]

    for cur_nval in nval_list
        @info "R = $cur_nval"

        cur_rrkf_estimate = RRKF.estimate_states(
            RRKF.RankReducedKalmanFilter(cur_nval, 1),
            ssm,
            simulation_grid,
            train_observations,
            smooth=false,
            compute_likelihood=false,
            save_all_steps=true,
            show_progress=true
        )
        cur_rrkf_sol = cur_rrkf_estimate[:filter]
        cur_rrkf_means = RRKF.means(cur_rrkf_sol);

        cur_rrkf_rmse_to_test = nanrmse(cur_rrkf_means * ssm.H', stacked_test_observations)
        cur_rrkf_rmse_to_truth = nanrmse(cur_rrkf_means * ssm.H', stacked_ground_truth)
        cur_rrkf_rmse_to_kf = rmsd(cur_rrkf_means, kf_means)
        cur_rrkf_cov_distance = mean([norm(Matrix(SC) - Matrix(KC)) / norm(Matrix(KC)) for (SC, KC) in zip(cur_rrkf_sol.Σ, kf_sol.Σ)])

        push!(rrkf_rmse_to_test_per_nval, cur_rrkf_rmse_to_test)
        push!(rrkf_rmse_to_truth_per_nval, cur_rrkf_rmse_to_truth)
        push!(rrkf_rmse_to_kf_per_nval, cur_rrkf_rmse_to_kf)
        push!(rrkf_cov_distance_per_nval, cur_rrkf_cov_distance)
        @info "rrkf($cur_nval) error" cur_rrkf_rmse_to_kf cur_rrkf_rmse_to_truth


        cur_enkf_rmse_to_test_per_enkf_loop = Float64[]
        cur_enkf_rmse_to_truth_per_enkf_loop = Float64[]
        cur_enkf_rmse_to_kf_per_enkf_loop = Float64[]
        cur_enkf_cov_distance_per_enkf_loop = Float64[]

        cur_etkf_rmse_to_test_per_etkf_loop = Float64[]
        cur_etkf_rmse_to_truth_per_etkf_loop = Float64[]
        cur_etkf_rmse_to_kf_per_etkf_loop = Float64[]
        cur_etkf_cov_distance_per_etkf_loop = Float64[]
        for enkf_loop in 1:2  # 20
            cur_enkf_estimate = RRKF.estimate_states(
                RRKF.EnsembleKalmanFilter(cur_nval, RRKF.enkf_correct),
                ssm,
                simulation_grid,
                train_observations,
                smooth=false,
                compute_likelihood=false,
                save_all_steps=true,
                show_progress=true
            )
            cur_enkf_sol = cur_enkf_estimate[:filter]
            cur_enkf_means = RRKF.means(cur_enkf_sol);

            cur_enkf_rmse_to_test = nanrmse(cur_enkf_means * ssm.H', stacked_test_observations)
            cur_enkf_rmse_to_truth = nanrmse(cur_enkf_means * ssm.H', stacked_ground_truth)
            cur_enkf_rmse_to_kf = rmsd(cur_enkf_means, kf_means)
            cur_enkf_cov_distance = mean([norm(Matrix(SC) - Matrix(KC)) / norm(Matrix(KC)) for (SC, KC) in zip(cur_enkf_sol.Σ, kf_sol.Σ)])

            push!(cur_enkf_rmse_to_test_per_enkf_loop, cur_enkf_rmse_to_test)
            push!(cur_enkf_rmse_to_truth_per_enkf_loop, cur_enkf_rmse_to_truth)
            push!(cur_enkf_rmse_to_kf_per_enkf_loop, cur_enkf_rmse_to_kf)
            push!(cur_enkf_cov_distance_per_enkf_loop, cur_enkf_cov_distance)

            cur_etkf_estimate = RRKF.estimate_states(
                RRKF.EnsembleKalmanFilter(cur_nval, RRKF.etkf_correct),
                ssm,
                simulation_grid,
                train_observations,
                smooth=false,
                compute_likelihood=false,
                save_all_steps=true,
                show_progress=true
            )
            cur_etkf_sol = cur_etkf_estimate[:filter]
            cur_etkf_means = RRKF.means(cur_etkf_sol);

            cur_etkf_rmse_to_test = nanrmse(cur_etkf_means * ssm.H', stacked_test_observations)
            cur_etkf_rmse_to_truth = nanrmse(cur_etkf_means * ssm.H', stacked_ground_truth)
            cur_etkf_rmse_to_kf = rmsd(cur_etkf_means, kf_means)
            cur_etkf_cov_distance = mean([norm(Matrix(SC) - Matrix(KC)) / norm(Matrix(KC)) for (SC, KC) in zip(cur_etkf_sol.Σ, kf_sol.Σ)])
            push!(cur_etkf_rmse_to_test_per_etkf_loop, cur_etkf_rmse_to_test)
            push!(cur_etkf_rmse_to_truth_per_etkf_loop, cur_etkf_rmse_to_truth)
            push!(cur_etkf_rmse_to_kf_per_etkf_loop, cur_etkf_rmse_to_kf)
            push!(cur_etkf_cov_distance_per_etkf_loop, cur_etkf_cov_distance)
        end

        push!(enkf_rmse_to_test_per_nval, cur_enkf_rmse_to_test_per_enkf_loop)
        push!(enkf_rmse_to_truth_per_nval, cur_enkf_rmse_to_truth_per_enkf_loop)
        push!(enkf_rmse_to_kf_per_nval, cur_enkf_rmse_to_kf_per_enkf_loop)
        push!(enkf_cov_distance_per_nval, cur_enkf_cov_distance_per_enkf_loop)

        push!(etkf_rmse_to_test_per_nval, cur_etkf_rmse_to_test_per_etkf_loop)
        push!(etkf_rmse_to_truth_per_nval, cur_etkf_rmse_to_truth_per_etkf_loop)
        push!(etkf_rmse_to_kf_per_nval, cur_etkf_rmse_to_kf_per_etkf_loop)
        push!(etkf_cov_distance_per_nval, cur_etkf_cov_distance_per_etkf_loop)

    end

    return Dict(
        "rrkf" => Dict(
            "rmse_to_test" => rrkf_rmse_to_test_per_nval,
            "rmse_to_truth" => rrkf_rmse_to_truth_per_nval,
            "rmse_to_kf" => rrkf_rmse_to_kf_per_nval,
            "cov_distance" => rrkf_cov_distance_per_nval,
        ),
        "enkf" => Dict(
            "rmse_to_test" => enkf_rmse_to_test_per_nval,
            "rmse_to_truth" => enkf_rmse_to_truth_per_nval,
            "rmse_to_kf" => enkf_rmse_to_kf_per_nval,
            "cov_distance" => enkf_cov_distance_per_nval,
        ),
        "etkf" => Dict(
            "rmse_to_test" => etkf_rmse_to_test_per_nval,
            "rmse_to_truth" => etkf_rmse_to_truth_per_nval,
            "rmse_to_kf" => etkf_rmse_to_kf_per_nval,
            "cov_distance" => etkf_cov_distance_per_nval,
        ),
        "kf" => Dict(
            "rmse_to_test" => kf_rmse_to_test,
            "rmse_to_truth" => kf_rmse_to_truth,
        ),
        "spectrum" => Dict(
            "kf_cov" => spectrum_of_kf_cov,
            "Q" => spectrum_of_process_noise_cov,
        )
    )
end


eval_results = []
for cur_lx in ℓₓ_list
    push!(eval_results, evaluation_for_spatial_lengthscale(cur_lx))
    GC.gc()
end


save(
    "./out/on-model_results_matern12.jld2",
    Dict(
        "nval_list" => float(nval_list),
        "spatial_lengthscale_list" => ℓₓ_list,
        "eval_results" => eval_results
    )
)
