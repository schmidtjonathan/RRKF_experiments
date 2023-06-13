using JLD2
# using TuePlots
using CairoMakie
using Statistics
using LaTeXStrings

CairoMakie.activate!(type = "svg")

cd("./experiments/air_quality")
loaded_results = load("./out/london_results.jld2")


include("../plot_theme.jl")


WIDTH, HEIGHT = 0.53FULL_WIDTH, HALF_HEIGHT


fig = begin
    grid_plot = Figure(;
        resolution=(WIDTH, HEIGHT),
        figure_padding=1,
    )

    ax_rmse_kf = Axis(
        grid_plot[1, 1],
        xtrimspine=true,
        ytrimspine=(true, false),
        topspinevisible = false,
        rightspinevisible = false,
        xgridvisible = false,
        ygridvisible = false,
        # xticklabelsvisible=false,
        xticks=loaded_results["nval_list"][begin:4:end],
        # xlabel="Low-rank dim.",
        aspect=0.55,
        # ylabel="RMSE",
        title=rich(
            rich("A.  ", font="Times New Roman bold", fontsize=BASE_FONTSIZE-1), "vs. KF mean       "
        ),
        titlegap=2.0,
        titlesize=BASE_FONTSIZE-2,
        titlealign=:center,
    )
    ax_cov_dist = Axis(
        grid_plot[1, 2],
        xtrimspine=true,
        ytrimspine=(true, false),
        topspinevisible = false,
        rightspinevisible = false,
        xgridvisible = false,
        ygridvisible = false,
        xticks=loaded_results["nval_list"][begin:4:end],
        # xlabel="Low-rank dim.",
        aspect=0.55,
        title=rich(
            rich("B.  ", font="Times New Roman bold", fontsize=BASE_FONTSIZE-1), "vs. KF cov.    "
        ),
        titlegap=2.0,
        titlesize=BASE_FONTSIZE-2,
        titlealign=:center,
        # ylabel="Frobenius distance"
    )


    rrkf_rmse_to_kf = loaded_results["rrkf"]["rmse_to_kf"]
    enkf_rmse_to_kf = [mean(res) for res in loaded_results["enkf"]["rmse_to_kf"]]
    etkf_rmse_to_kf = [mean(res) for res in loaded_results["etkf"]["rmse_to_kf"]]
    enkf_rmse_to_kf_std = [std(res) for res in loaded_results["enkf"]["rmse_to_kf"]]
    etkf_rmse_to_kf_std = [std(res) for res in loaded_results["etkf"]["rmse_to_kf"]]

    rrkf_cov_distance = loaded_results["rrkf"]["cov_distance"]
    enkf_cov_distance = [mean(res) for res in loaded_results["enkf"]["cov_distance"]]
    etkf_cov_distance = [mean(res) for res in loaded_results["etkf"]["cov_distance"]]
    enkf_cov_distance_std = [std(res) for res in loaded_results["enkf"]["cov_distance"]]
    etkf_cov_distance_std = [std(res) for res in loaded_results["etkf"]["cov_distance"]]




    band!(
        ax_rmse_kf,
        loaded_results["nval_list"],
        enkf_rmse_to_kf - 1.97 .* enkf_rmse_to_kf_std,
        enkf_rmse_to_kf + 1.97 .* enkf_rmse_to_kf_std,
        color=(COLORS[2], 0.4)
    )
    enkf_legend_handle = scatterlines!(
        ax_rmse_kf,
        loaded_results["nval_list"],
        enkf_rmse_to_kf,
        marker=:rect, color=COLORS[2]
    )


    band!(
        ax_rmse_kf,
        loaded_results["nval_list"],
        etkf_rmse_to_kf - 1.97 .* etkf_rmse_to_kf_std,
        etkf_rmse_to_kf + 1.97 .* etkf_rmse_to_kf_std,
         color=(COLORS[1], 0.4))
    etkf_legend_handle = scatterlines!(
        ax_rmse_kf,
        loaded_results["nval_list"],
        etkf_rmse_to_kf,
        color=COLORS[1]
    )

    rrkf_legend_handle = scatterlines!(
        ax_rmse_kf,
        loaded_results["nval_list"],
        rrkf_rmse_to_kf,
        marker=:diamond, color=COLORS[3]
    )


    band!(
        ax_cov_dist,
        loaded_results["nval_list"],
        enkf_cov_distance - 1.97 .* enkf_cov_distance_std,
        enkf_cov_distance + 1.97 .* enkf_cov_distance_std,
        color=(COLORS[2], 0.4)
    )
    scatterlines!(
        ax_cov_dist,
        loaded_results["nval_list"],
        enkf_cov_distance,
        marker=:rect, color=COLORS[2]
    )

    band!(
        ax_cov_dist,
        loaded_results["nval_list"],
        etkf_cov_distance - 1.97 .* etkf_cov_distance_std,
        etkf_cov_distance + 1.97 .* etkf_cov_distance_std,
        color=(COLORS[1], 0.4)
    )
    scatterlines!(
        ax_cov_dist,
        loaded_results["nval_list"],
        etkf_cov_distance,
        label="etkf", color=COLORS[1]
    )

    scatterlines!(
        ax_cov_dist,
        loaded_results["nval_list"],
        rrkf_cov_distance,
        marker=:diamond, color=COLORS[3]
    )

    CairoMakie.scatter!(
        ax_rmse_kf,
        loaded_results["nval_list"][end:end],
        loaded_results["rrkf"]["rmse_to_kf"][end:end],
        marker=:diamond,
        color=COLORS[4],
        strokecolor=:black,
        strokewidth=0.5,
        zorder=10,
        # glowwidth=20,
        # glowcolor=PN_COLORS[1],
    )
    CairoMakie.scatter!(
        ax_cov_dist,
        loaded_results["nval_list"][end:end],
        loaded_results["rrkf"]["cov_distance"][end:end],
        marker=:diamond,
        color=COLORS[4],
        strokecolor=:black,
        strokewidth=0.5,
        zorder=10,
        # glowwidth=20,
        # glowcolor=PN_COLORS[1],
    )

    xlims!(ax_rmse_kf, high=153)
    xlims!(ax_cov_dist, high=153)

    # Legend(grid_plot[1, 4], [rrkf_legend_handle, enkf_legend_handle, etkf_legend_handle, kf_legend_handle], ["ours", "EnKF", "dEnKF", "KF"])
    Legend(grid_plot[1, 3], [rrkf_legend_handle, enkf_legend_handle, etkf_legend_handle], [rich(rich("RRKF", font="Times New Roman bold"), "\n", rich("(ours)", font="Times New Roman bold", offset = (0.0, 1.0))), "EnKF", "ETKF"])#, position=(-4.0, -1.0))

    ax_rmse_kf.ylabel = "RMSE"
    # ax_rmse_kf.ylabel = "RMSE to KF"
    ax_cov_dist.ylabel = "Frobenius distance" #rich("Frobenius", "\n", rich("distance", offset = (0.0, 1.0)))
    ax_rmse_kf.xlabelpadding=0.0
    ax_cov_dist.xlabelpadding=0.0

    colgap!(grid_plot.layout, 1, 8.0)
    colgap!(grid_plot.layout, 2, 5.0)
    Label(grid_plot[1, 1:end-1, Bottom()], "low-rank dimension ", fontsize=BASE_FONTSIZE-1, font="Times New Roman regular", halign = :center, valign=:top, padding=(0, 0, 0, 15))


    grid_plot
end

display(fig)
save("./out/london_results.pdf", grid_plot, pt_per_unit = 1)
