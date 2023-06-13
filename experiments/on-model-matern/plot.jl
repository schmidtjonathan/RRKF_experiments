using JLD2
# using TuePlots
using Statistics
using CairoMakie
using LaTeXStrings

CairoMakie.activate!(type = "svg")

cd("/home/jschmidt/.julia/dev/SpectralKF/experiments/correct_st_model/")


loaded_results = load("./out_new/on-model_error_data_matern12.jld2")


include("../plot_theme.jl")


WIDTH, HEIGHT = FULL_WIDTH, 1.35FULL_HEIGHT


fig = begin

    grid_plot = Figure(;
        resolution=(WIDTH, HEIGHT),
        figure_padding=1,
    )

    rmse_axes = []
    covdist_axes = []
    # spec_axes = []
    legend_handles = []
    for (plot_i, cur_lx) in enumerate(loaded_results["spatial_lengthscale_list"])
        push!(
            rmse_axes,
            Axis(
                grid_plot[1, plot_i],
                # title=L"\ell_x = %$cur_lx",
                # yscale=log10,
                xticks=round.(Int, LinRange(1, loaded_results["nval_list"][end], 5)),
                xtrimspine=true,
                ytrimspine=(true, false),
                topspinevisible = false,
                rightspinevisible = false,
                xgridvisible = false,
                ygridvisible = false,
                xticklabelsvisible=false,
                aspect=1.5,
                # title="vs. KF mean",
                titlesize=BASE_FONTSIZE - 3,
                titlealign=:left,
                titlegap=0.0,
            )
        )
        Label(grid_plot[1, plot_i, Top()], L"\ell_x = %$cur_lx", padding=(0, 0, 0, 0), fontsize=BASE_FONTSIZE-1)

        spectrum_legendhandle = lines!(
            rmse_axes[end],
            # loaded_results["nval_list"],
            cumsum(loaded_results["eval_results"][plot_i]["spectrum"]["kf_cov"]) / sum(loaded_results["eval_results"][plot_i]["spectrum"]["kf_cov"]),
            color=:grey80,
            linewidth=1.5,
        )


        band!(
            rmse_axes[end],
            loaded_results["nval_list"],
            [mean(res) for res in loaded_results["eval_results"][plot_i]["enkf"]["rmse_to_kf"]] - 1.97 .* [std(res) for res in loaded_results["eval_results"][plot_i]["enkf"]["rmse_to_kf"]],
            [mean(res) for res in loaded_results["eval_results"][plot_i]["enkf"]["rmse_to_kf"]] + 1.97 .* [std(res) for res in loaded_results["eval_results"][plot_i]["enkf"]["rmse_to_kf"]],
            color=(COLORS[2], 0.4)
        )
        _enkfline = scatterlines!(
            rmse_axes[end],
            loaded_results["nval_list"],
            [mean(res) for res in loaded_results["eval_results"][plot_i]["enkf"]["rmse_to_kf"]],
            label="EnKF",
            marker=:rect,             color=COLORS[2]
        )


        band!(
            rmse_axes[end],
            loaded_results["nval_list"],
            [mean(res) for res in loaded_results["eval_results"][plot_i]["etkf"]["rmse_to_kf"]] - 1.97 .* [std(res) for res in loaded_results["eval_results"][plot_i]["etkf"]["rmse_to_kf"]],
            [mean(res) for res in loaded_results["eval_results"][plot_i]["etkf"]["rmse_to_kf"]] + 1.97 .* [std(res) for res in loaded_results["eval_results"][plot_i]["etkf"]["rmse_to_kf"]],
            color=(COLORS[1], 0.4)
        )
        _etkfline = scatterlines!(
            rmse_axes[end],
            loaded_results["nval_list"],
            [mean(res) for res in loaded_results["eval_results"][plot_i]["etkf"]["rmse_to_kf"]],
            label="ETKF",
                        color=COLORS[1]
        )

        _skfline = scatterlines!(
            rmse_axes[end],
            loaded_results["nval_list"],
            loaded_results["eval_results"][plot_i]["skf"]["rmse_to_kf"],
            label="SKF",
            marker=:diamond,             color=COLORS[3], zorder=10
        )
        lastpoint_legend_handle = CairoMakie.scatter!(
            rmse_axes[end],
            loaded_results["nval_list"][end:end],
            loaded_results["eval_results"][plot_i]["skf"]["rmse_to_kf"][end:end],
            label="SKF",
            marker=:diamond,
            color=COLORS[4],
            strokecolor=:black,
            strokewidth=0.5,
            zorder=10,
            # glowwidth=20,
            # glowcolor=PN_COLORS[1],
        )

        if plot_i == 1
            push!(legend_handles, _skfline)
            # push!(legend_handles, lastpoint_legend_handle)
            push!(legend_handles, _enkfline)
            push!(legend_handles, _etkfline)
            push!(legend_handles, spectrum_legendhandle)
        end

        push!(
            covdist_axes,
            Axis(
                grid_plot[2, plot_i],
                # yscale=log10,
                xticks=round.(Int, LinRange(1, loaded_results["nval_list"][end], 5)),
                xtrimspine=true,
                ytrimspine=(true, false),
                topspinevisible = false,
                rightspinevisible = false,
                xgridvisible = false,
                ygridvisible = false,
                xticklabelsvisible=true,
                aspect=1.5,
                xlabel="low-rank dim.",
                # title="vs. KF covariance",
                titlesize=BASE_FONTSIZE-3,
                titlegap=0.1,
                titlealign=:left,
            )
        )
        lines!(
            covdist_axes[end],
            # loaded_results["nval_list"],
            cumsum(loaded_results["eval_results"][plot_i]["spectrum"]["kf_cov"]) / sum(loaded_results["eval_results"][plot_i]["spectrum"]["kf_cov"]),
            color=:grey80,
            linewidth=1.5,
        )



        band!(
            covdist_axes[end],
            loaded_results["nval_list"],
            [mean(res) for res in loaded_results["eval_results"][plot_i]["enkf"]["cov_distance"]] - 1.97 .* [std(res) for res in loaded_results["eval_results"][plot_i]["enkf"]["cov_distance"]],
            [mean(res) for res in loaded_results["eval_results"][plot_i]["enkf"]["cov_distance"]] + 1.97 .* [std(res) for res in loaded_results["eval_results"][plot_i]["enkf"]["cov_distance"]],
            color=(COLORS[2], 0.4)
        )
        scatterlines!(
            covdist_axes[end],
            loaded_results["nval_list"],
            [mean(res) for res in loaded_results["eval_results"][plot_i]["enkf"]["cov_distance"]],
            label="EnKF",
            marker=:rect, color=COLORS[2],
            )



        band!(
            covdist_axes[end],
            loaded_results["nval_list"],
            [mean(res) for res in loaded_results["eval_results"][plot_i]["etkf"]["cov_distance"]] - 1.97 .* [std(res) for res in loaded_results["eval_results"][plot_i]["etkf"]["cov_distance"]],
            [mean(res) for res in loaded_results["eval_results"][plot_i]["etkf"]["cov_distance"]] + 1.97 .* [std(res) for res in loaded_results["eval_results"][plot_i]["etkf"]["cov_distance"]],
            color=(COLORS[1], 0.4)
        )
        scatterlines!(
            covdist_axes[end],
            loaded_results["nval_list"],
            [mean(res) for res in loaded_results["eval_results"][plot_i]["etkf"]["cov_distance"]],
            label="ETKF",
            color=COLORS[1],
            )


        scatterlines!(
            covdist_axes[end],
            loaded_results["nval_list"],
            loaded_results["eval_results"][plot_i]["skf"]["cov_distance"],
            label="SKF",
            marker=:diamond, color=COLORS[3], zorder=10
        )
        CairoMakie.scatter!(
            covdist_axes[end],
            loaded_results["nval_list"][end:end],
            loaded_results["eval_results"][plot_i]["skf"]["cov_distance"][end:end],
            label="SKF",
            marker=:diamond,
            color=COLORS[4],
            strokecolor=:black,
            strokewidth=0.5,
            zorder=10,
            # glowwidth=20,
            # glowcolor=PN_COLORS[1],
        )


    end

    Legend(grid_plot[0, :], legend_handles, [rich("RRKF (ours)", font="Times New Roman bold"), "EnKF", "ETKF", "cumulative spectrum"], orientation=:horizontal)
    # axislegend(rmse_axes[begin], legend_handles, ["ours", "EnKF", "ETKF"], framevisible = true, rowgap=0, colgap=0, framewidth=0.5, position=(5.0, 2.0))


    rmse_axes[begin].ylabel = "RMSE"
    covdist_axes[begin].ylabel = rich("Frobenius", "\n", rich("distance", offset = (0.0, 1.0)))
    # spec_axes[begin].ylabel = "Cumulative\nspectrum"
    # Label(
    #     grid_plot[3, 1, Left()],
    #     "Spectrum",
    #     # font = "Times New Roman bold",
    #     padding = (0, 30, 0, 0),
    #     rotation=π/2
    # )

    # linkyaxes!(spec_axes...)
    linkyaxes!(rmse_axes...)
    linkyaxes!(covdist_axes...)
    # hidespines!(spec_axes[2], :l)
    # hidespines!(spec_axes[3], :l)
    # hidespines!(spec_axes[4], :l)

    # colsize!(grid_plot.layout, 1, Auto(0.22))
    # colsize!(grid_plot.layout, 2, Auto(0.27))
    # colsize!(grid_plot.layout, 3, Auto(0.27))
    # colsize!(grid_plot.layout, 4, Auto(0.27))

    rowgap!(grid_plot.layout, 8.0)
    rowgap!(grid_plot.layout, 1, 8.0)
    # rowgap!(grid_plot.layout, 2, 0.0)
    colgap!(grid_plot.layout, 5.0)

    # resize_to_layout!(grid_plot)

    grid_plot
end

display(fig)
save("./out_new/on-model_grid_plot_matern12_reduced_cramped.pdf", fig, pt_per_unit = 1)



