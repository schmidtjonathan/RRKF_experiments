using JLD2
# using TuePlots
using Statistics
using BenchmarkTools
using CairoMakie
using LaTeXStrings

CairoMakie.activate!(type = "svg")

cd("./experiments/err_runtime/")


loaded_results = load("./out/on-model_err_runtime_matern12.jld2")

to_seconds(tm) = tm / 1e9

include("../plot_theme.jl")


WIDTH, HEIGHT = FULL_WIDTH, 1.2FULL_HEIGHT


fig = begin

    grid_plot = Figure(;
        resolution=(WIDTH, HEIGHT),
        figure_padding=(5, 15, 1, 1),
    )

    rmse_axes = []
    legend_handles = []
    for (plot_i, cur_lx) in enumerate(loaded_results["spatial_lengthscale_list"])
        push!(
            rmse_axes,
            Axis(
                grid_plot[1, plot_i],
                # xticks=round.(Int, LinRange(1, loaded_results["nval_list"][end], 5)),
                xtrimspine=false,
                ytrimspine=(true, false),
                topspinevisible = false,
                rightspinevisible = false,
                xgridvisible = false,
                ygridvisible = false,
                xticklabelsvisible=true,
                aspect=1.,
                xlabel="Wall time [sec]",
                # title="vs. KF mean",
                titlesize=BASE_FONTSIZE - 3,
                titlealign=:left,
                titlegap=0.0,
            )
        )
        Label(grid_plot[1, plot_i, Top()], L"\ell_x = %$cur_lx", padding=(0, 0, 0, 0), fontsize=BASE_FONTSIZE-1)

        spectrum_legendhandle = lines!(
            rmse_axes[end],
            # collect(map(to_seconds ∘ time ∘ mean, loaded_results["eval_results"][plot_i]["rrkf"]["bench"]))[1:end-1],
            LinRange(collect(map(to_seconds ∘ time ∘ mean, loaded_results["eval_results"][plot_i]["rrkf"]["bench"]))[begin], collect(map(to_seconds ∘ time ∘ mean, loaded_results["eval_results"][plot_i]["rrkf"]["bench"]))[end], length(cumsum(loaded_results["eval_results"][plot_i]["spectrum"]["kf_cov"]) / sum(loaded_results["eval_results"][plot_i]["spectrum"]["kf_cov"]))),
            cumsum(loaded_results["eval_results"][plot_i]["spectrum"]["kf_cov"]) / sum(loaded_results["eval_results"][plot_i]["spectrum"]["kf_cov"]),
            color=:grey80,
            linewidth=1.5,
        )



        _enkfline = scatterlines!(
            rmse_axes[end],
            collect(map(to_seconds ∘ time ∘ mean, loaded_results["eval_results"][plot_i]["enkf"]["bench"])),
            [mean(res) for res in loaded_results["eval_results"][plot_i]["enkf"]["rmse_to_kf"]],
            label="EnKF",
            marker=:rect,             color=COLORS[2]
        )



        _etkfline = scatterlines!(
            rmse_axes[end],
            collect(map(to_seconds ∘ time ∘ mean, loaded_results["eval_results"][plot_i]["etkf"]["bench"])),
            [mean(res) for res in loaded_results["eval_results"][plot_i]["etkf"]["rmse_to_kf"]],
            label="ETKF",
                        color=COLORS[1]
        )

        _rrkfline = scatterlines!(
            rmse_axes[end],
            collect(map(to_seconds ∘ time ∘ mean, loaded_results["eval_results"][plot_i]["rrkf"]["bench"])),
            loaded_results["eval_results"][plot_i]["rrkf"]["rmse_to_kf"],
            label="SKF",
            marker=:diamond,             color=COLORS[3], zorder=10
        )
        lastpoint_legend_handle = CairoMakie.scatter!(
            rmse_axes[end],
            collect(map(to_seconds ∘ time ∘ mean, loaded_results["eval_results"][plot_i]["rrkf"]["bench"]))[end:end],
            loaded_results["eval_results"][plot_i]["rrkf"]["rmse_to_kf"][end:end],
            label="SKF",
            marker=:diamond,
            color=COLORS[4],
            strokecolor=:black,
            strokewidth=0.5,
            zorder=10,
        )

        rmse_axes[begin].ylabel = "RMSE"

        if plot_i == 1
            push!(legend_handles, _rrkfline)
            push!(legend_handles, _enkfline)
            push!(legend_handles, _etkfline)
            push!(legend_handles, spectrum_legendhandle)
        end

    end

    Legend(grid_plot[0, :], legend_handles, [rich("RRKF (ours)", font="Times New Roman bold"), "EnKF", "ETKF", "cumulative spectrum"], orientation=:horizontal)
    colgap!(grid_plot.layout, 2.0)
    rowgap!(grid_plot.layout, 0.0)
    rowgap!(grid_plot.layout, 1, 8.0)

    linkyaxes!(rmse_axes...)

    grid_plot
end

display(fig)
save("./out/on-model_err_runtime_matern12.pdf", fig, pt_per_unit = 1)



