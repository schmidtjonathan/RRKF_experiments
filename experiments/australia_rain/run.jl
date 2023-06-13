ENV["RASTERDATASOURCES_PATH"] = abspath("./experiments/australia_rain/data/")

cd(abspath("./experiments/australia_rain"))

import ArchGDAL
using Rasters
using RasterDataSources
using LinearAlgebra
using MKL
using Dates
using Shapefile
using Downloads
using Statistics
using DataFrames
using SparseArrays
using KernelFunctions
using LinearAlgebra
using Distributions
using JLD2
using Plots
using CSV
using Random
using Kronecker

using RRKF

mask_trim(climate, poly) = trim(mask(climate; with=poly))

shapefile_url = "https://github.com/nvkelso/natural-earth-vector/raw/master/10m_cultural/ne_10m_admin_0_countries.shp"
shapefile_name = "boundary.shp"
shapefile_path = joinpath(ENV["RASTERDATASOURCES_PATH"], shapefile_name)

isfile(shapefile_path) || Downloads.download(shapefile_url, shapefile_path)
shapes = Shapefile.Handle(shapefile_path)
australia_shape = shapes.shapes[177] # Australia



# Spatial Kernel
function joint_K(XY, spatial_kernelfun, ℓxy, jitterval = 1e-8)
    return kernelmatrix(with_lengthscale(spatial_kernelfun(), ℓxy), XY) + jitterval * I
end


# State-space model
proj(d, ν, q) = sparse(RRKF.projectionmatrix(d, ν, q))




rrkf_NVALS = 200
NUM_DLR_STEPS = 1
rrkf_alg = RRKF.RankReducedKalmanFilter(rrkf_NVALS, NUM_DLR_STEPS)



# ---------- EVALUATE

evaluate_date_range = collect(Date(2023, 01, 25):Day(1):Date(2023, 03, 5))
evaluate_all_times = 1.0:1.0:length(evaluate_date_range)

evaluate_rain_raster = RasterSeries(AWAP, :rainfall; date=evaluate_date_range)
evaluate_aus_rain_raster = map(
    r -> mask_trim(
        resample(r; res=0.4, method=:cubicspline)[Rasters.Band(1)],
        australia_shape
       ),
       evaluate_rain_raster
)



_evaluate_dummy_aoi_frame = evaluate_aus_rain_raster[1]
_evaluate_prnt = parent(_evaluate_dummy_aoi_frame)
evaluate_land_mask = _evaluate_prnt .!= missingval(_evaluate_dummy_aoi_frame)

evaluate_all_lons = [_evaluate_dummy_aoi_frame.dims[1][i] for i in 1:length(_evaluate_dummy_aoi_frame.dims[1])]
evaluate_all_lats = [_evaluate_dummy_aoi_frame.dims[2][i] for i in 1:length(_evaluate_dummy_aoi_frame.dims[2])]
evaluate_all_lonlats = [[ln, lt] for ln in evaluate_all_lons, lt in evaluate_all_lats]

evaluate_all_lonlats = vecvec2mat(vec(evaluate_all_lonlats))
evaluate_normalized_all_lonlats = (evaluate_all_lonlats .- mean(evaluate_all_lonlats, dims=1)) ./ std(evaluate_all_lonlats, dims=1)



evaluate_raw_rainfall_mat = Float64.(vecvec2mat([parent(r)[evaluate_land_mask] for r in evaluate_aus_rain_raster]))
evaluate_mean_rainfall = mean(evaluate_raw_rainfall_mat)
evaluate_std_rainfall = std(evaluate_raw_rainfall_mat)


evaluate_normalized_data_mat = copy(evaluate_raw_rainfall_mat)
evaluate_normalized_data_mat = (evaluate_normalized_data_mat .- evaluate_mean_rainfall) ./ evaluate_std_rainfall


evaluate_all_linear_idcs = LinearIndices(evaluate_land_mask)
evaluate_ENTIRE_DIMENSION = size(evaluate_all_linear_idcs) |> prod
evaluate_LAND_DIMENSION = sum(evaluate_land_mask)
evaluate_all_cartesian_idcs = collect(CartesianIndices(evaluate_land_mask))
evaluate_land_linear_idcs = evaluate_all_linear_idcs[evaluate_land_mask]
evaluate_land_cartesian_idcs = evaluate_all_cartesian_idcs[evaluate_land_mask]




# --- Lying S shape
circrange1 = (10, 20)
centerx1, centery1 = 60, 30
xinterval1 = (0, 1000)
yinterval1 = (0, 30)
circrange2 = (10, 20)
centerx2, centery2 = 30, 30
xinterval2 = (0, 1000)
yinterval2 = (30, 1000)
evaluate_land_patch = findall(
    i -> circrange1[1] <= sqrt((i.I[1]-centerx1)^2 + (i.I[2]-centery1)^2) <= circrange1[2] && xinterval1[1] <= i.I[1] <= xinterval1[2] && yinterval1[1] <= i.I[2] <= yinterval1[2] || circrange2[1] <= sqrt((i.I[1]-centerx2)^2 + (i.I[2]-centery2)^2) <= circrange2[2] && xinterval2[1] <= i.I[1] <= xinterval2[2] && yinterval2[1] <= i.I[2] <= yinterval2[2],
    evaluate_land_cartesian_idcs
)



evaluate_land_patch_indices = evaluate_land_linear_idcs[evaluate_land_patch]


evaluate_rainfall_mat = Float64.(vecvec2mat([vec(parent(r)) for r in evaluate_aus_rain_raster]))
evaluate_rainfall_mat[:, evaluate_land_patch_indices] .= NaN
if true
    heatmap(reshape(evaluate_rainfall_mat[1, :, :],  (102, 82))', yflip=true)
end
evaluate_rainfall_mat = evaluate_rainfall_mat[:, evaluate_land_linear_idcs]
evaluate_rainfall_mat = (evaluate_rainfall_mat .- evaluate_mean_rainfall) ./ evaluate_std_rainfall


evaluate_lonlats_as_vec = [evaluate_normalized_all_lonlats[sgp, :] for sgp in evaluate_land_linear_idcs]

data_evaluate = [evaluate_rainfall_mat[t, :] for t in axes(evaluate_rainfall_mat, 1)]


minimizer = load("out/hparams.jld2")


H_all = proj(evaluate_LAND_DIMENSION, minimizer["temporal_smoothness"], 0)


# --------- BUILD SSM

ν = minimizer["temporal_smoothness"]
dim = length(data_evaluate[1])
@assert dim == size(H_all, 1)

R = minimizer["σᵣ"]^2 * Diagonal(ones(dim))
spatial_K = joint_K(evaluate_lonlats_as_vec, minimizer["spatial_kernelfun"], minimizer["ℓxy"])


prior_dynamics = RRKF.build_spatiotemporal_matern_process(
    ν,
    minimizer["ℓₜ"],
    minimizer["σₜ"],
    spatial_K,
)

ssm = RRKF.StateSpaceModel(
    prior_dynamics, H_all, R
)


# --------------------

rrkf_estim_stats = @timed RRKF.estimate_states(
    rrkf_alg,
    ssm,
    evaluate_all_times,
    data_evaluate;
    smooth=true,
    compute_likelihood=true,
    save_all_steps=true,
    show_progress=true,
)
rrkf_estim = rrkf_estim_stats.value

@info "DIMS" measdim=sum(evaluate_land_mask) - length(evaluate_land_patch_indices) statedim=sum(evaluate_land_mask)

proj_to_0 = proj(length(data_evaluate[1]), minimizer["temporal_smoothness"], 0)
rrkf_means = RRKF.means(rrkf_estim[:smoother]) * proj_to_0'
rrkf_stds = RRKF.stds(rrkf_estim[:smoother]) * proj_to_0'

rescaled_rrkf_means = rrkf_means .* evaluate_std_rainfall .+ evaluate_mean_rainfall
rescaled_rrkf_stds = rrkf_stds .* evaluate_std_rainfall

rrkf_absolute_errors = abs.(rescaled_rrkf_means .- evaluate_raw_rainfall_mat)
rrkf_norm_errors = abs.(rrkf_means .- evaluate_normalized_data_mat)
rrkf_zscores = (rrkf_means .- evaluate_normalized_data_mat) ./ rrkf_stds


# save(
#     "./out/rrkf-r$(rrkf_NVALS)_results.jld2",
#     Dict(
#         "rrkf_means_datascale" => rescaled_rrkf_means,
#         "rrkf_stds_datascale" => rescaled_rrkf_stds,
#         "rrkf_means_normscale" => rrkf_means,
#         "rrkf_stds_normscale" => rrkf_stds,
#         "rrkf_absolute_errors_datascale" => rrkf_absolute_errors,
#         "rrkf_absolute_errors_normscale" => rrkf_norm_errors,
#         "rrkf_zscores" => rrkf_zscores,
#         "prior_parameters" => minimizer,
#         "elapsed_time" => rrkf_estim_stats.time,
#         "prior_hparams" => minimizer,
#         "date_range" => evaluate_date_range,
#     )
# )


function borders!(p)
    plot!(p, australia_shape; fillalpha=0, linewidth=3.0)
    return p
end



train_xmin, train_xmax = minimum(evaluate_all_lonlats[evaluate_land_patch_indices, 1]), maximum(evaluate_all_lonlats[evaluate_land_patch_indices, 1])
train_ymin, train_ymax = minimum(evaluate_all_lonlats[evaluate_land_patch_indices, 2]), maximum(evaluate_all_lonlats[evaluate_land_patch_indices, 2])



anim = @animate for t in axes(rrkf_means, 1)
    raster_data_in_datascale = copy(evaluate_aus_rain_raster[t])
    raster_data_in_zscore = copy(evaluate_aus_rain_raster[t])
    raster_means_in_datascale = copy(evaluate_aus_rain_raster[t])
    raster_means_in_zscore = copy(evaluate_aus_rain_raster[t])
    raster_stds_in_datascale = copy(evaluate_aus_rain_raster[t])
    raster_stds_in_zscore = copy(evaluate_aus_rain_raster[t])
    raster_zscores = copy(evaluate_aus_rain_raster[t])
    raster_absolute_errors = copy(evaluate_aus_rain_raster[t])
    raster_norm_errors = copy(evaluate_aus_rain_raster[t])
    raster_data_stds = copy(evaluate_aus_rain_raster[t])
    raster_evalmask = copy(evaluate_aus_rain_raster[t])

    raster_data_in_datascale[evaluate_land_linear_idcs] .= evaluate_raw_rainfall_mat[t, :]
    raster_data_in_zscore[evaluate_land_linear_idcs] .= evaluate_normalized_data_mat[t, :]
    raster_means_in_datascale[evaluate_land_linear_idcs] .= rescaled_rrkf_means[t, :]
    raster_means_in_zscore[evaluate_land_linear_idcs] .= rrkf_means[t, :]
    raster_stds_in_datascale[evaluate_land_linear_idcs] .= 1.97 .* rescaled_rrkf_stds[t, :]
    raster_stds_in_zscore[evaluate_land_linear_idcs] .= 1.97 .* rrkf_stds[t, :]
    raster_zscores[evaluate_land_linear_idcs] .= rrkf_zscores[t, :]
    raster_absolute_errors[evaluate_land_linear_idcs] .= rrkf_absolute_errors[t, :]
    raster_norm_errors[evaluate_land_linear_idcs] .= rrkf_norm_errors[t, :]
    raster_data_stds[evaluate_land_linear_idcs] .= evaluate_std_rainfall
    raster_evalmask[evaluate_land_linear_idcs] .= 0.0
    raster_evalmask[evaluate_land_patch_indices] .= 1.0

    p_data = plot(mask_trim(raster_data_in_datascale, australia_shape), title="$(evaluate_date_range[t]) Data",  showaxis=false, gridalpha=0.0, xlabel="", ylabel="")
	plot!(p_data, Shape([train_xmin, train_xmax, train_xmax, train_xmin], [train_ymin, train_ymin, train_ymax, train_ymax]), linecolor=:white, linewidth=3, fillalpha=0.0, label="")

    p_datanorm = plot(mask_trim(raster_data_in_zscore, australia_shape), title="$(evaluate_date_range[t]) norm. Data",  showaxis=false, gridalpha=0.0, xlabel="", ylabel="")
	plot!(p_data, Shape([train_xmin, train_xmax, train_xmax, train_xmin], [train_ymin, train_ymin, train_ymax, train_ymax]), linecolor=:white, linewidth=3, fillalpha=0.0, label="")

    p_estim = plot(mask_trim(raster_means_in_datascale, australia_shape), title="rrkf(r=$(rrkf_NVALS)) Means datascale", clim=Plots.get_clims(p_data.subplots[1]), showaxis=false, gridalpha=0.0, xlabel="", ylabel="")
	plot!(p_estim, Shape([train_xmin, train_xmax, train_xmax, train_xmin], [train_ymin, train_ymin, train_ymax, train_ymax]), linecolor=:white, linewidth=3, fillalpha=0.0, label="")

    p_estimnorm = plot(mask_trim(raster_means_in_zscore, australia_shape), title="rrkf(r=$(rrkf_NVALS)) norm. Means", clim=Plots.get_clims(p_datanorm.subplots[1]), showaxis=false, gridalpha=0.0, xlabel="", ylabel="")
	plot!(p_estim, Shape([train_xmin, train_xmax, train_xmax, train_xmin], [train_ymin, train_ymin, train_ymax, train_ymax]), linecolor=:white, linewidth=3, fillalpha=0.0, label="")

    p_norm_std = plot(mask_trim(raster_stds_in_zscore, australia_shape), title="STDs normscale", showaxis=false, gridalpha=0.0, xlabel="", ylabel="", )
	plot!(p_norm_std, Shape([train_xmin, train_xmax, train_xmax, train_xmin], [train_ymin, train_ymin, train_ymax, train_ymax]), linecolor=:white, linewidth=3, fillalpha=0.0, label="")

    p_estim_std = plot(mask_trim(raster_stds_in_datascale, australia_shape), title="STDs datascale", showaxis=false, gridalpha=0.0, xlabel="", ylabel="", )
	plot!(p_estim_std, Shape([train_xmin, train_xmax, train_xmax, train_xmin], [train_ymin, train_ymin, train_ymax, train_ymax]), linecolor=:white, linewidth=3, fillalpha=0.0, label="")

    p_data_std = plot(mask_trim(raster_data_stds, australia_shape), title="Data STD", showaxis=false, gridalpha=0.0, xlabel="", ylabel="", )
	plot!(p_data_std, Shape([train_xmin, train_xmax, train_xmax, train_xmin], [train_ymin, train_ymin, train_ymax, train_ymax]), linecolor=:white, linewidth=3, fillalpha=0.0, label="")

    p_absolute_errors = plot(mask_trim(raster_absolute_errors, australia_shape), title="Error", showaxis=false, gridalpha=0.0, xlabel="", ylabel="", )
	plot!(p_absolute_errors, Shape([train_xmin, train_xmax, train_xmax, train_xmin], [train_ymin, train_ymin, train_ymax, train_ymax]), linecolor=:white, linewidth=3, fillalpha=0.0, label="")

    p_norm_errors = plot(mask_trim(raster_norm_errors, australia_shape), title="norm. Error", showaxis=false, gridalpha=0.0, xlabel="", ylabel="", )
	plot!(p_norm_errors, Shape([train_xmin, train_xmax, train_xmax, train_xmin], [train_ymin, train_ymin, train_ymax, train_ymax]), linecolor=:white, linewidth=3, fillalpha=0.0, label="")

    p_zscores = plot(mask_trim(raster_zscores, australia_shape), title="Z-scores", showaxis=false, gridalpha=0.0, xlabel="", ylabel="", c=:vik, clims=(-3, 3))
	plot!(p_zscores, Shape([train_xmin, train_xmax, train_xmax, train_xmin], [train_ymin, train_ymin, train_ymax, train_ymax]), linecolor=:white, linewidth=3, fillalpha=0.0, label="")

    p_evalmask = plot(mask_trim(raster_evalmask, australia_shape), title="Evaluation mask", showaxis=false, gridalpha=0.0, xlabel="", ylabel="", colorscale=:viridis )
	plot!(p_evalmask, Shape([train_xmin, train_xmax, train_xmax, train_xmin], [train_ymin, train_ymin, train_ymax, train_ymax]), linecolor=:white, linewidth=3, fillalpha=0.0, label="")

    plot(
        borders!(p_data),
        borders!(p_datanorm),
        borders!(p_data_std),
        borders!(p_evalmask),
        borders!(p_estim),
        borders!(p_estimnorm),
        borders!(p_estim_std),
        borders!(p_norm_std),
        borders!(p_absolute_errors),
        borders!(p_norm_errors),
        borders!(p_zscores),
        plot(),
        layout=(3, 4), size=(2500, 2000), legend=false, ticks=false
    )
end
# gif(anim, "./out/rrkf-r$(rrkf_NVALS)_estim.gif", fps=1)
gif(anim, fps=1)


