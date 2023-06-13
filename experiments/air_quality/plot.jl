using JLD2
# using TuePlots
using CairoMakie
using LaTeXStrings

CairoMakie.activate!(type = "svg")

cd("/home/jschmidt/.julia/dev/SpectralKF/experiments/air_quality")

loaded_results = load("./out_new/london_error_data.jld2")


include("../plot_theme.jl")


WIDTH, HEIGHT = FULL_WIDTH, 0.9HALF_HEIGHT


fig = begin
    grid_plot = Figure(;
        resolution=(WIDTH, HEIGHT),
        figure_padding=1,
    )
    ax_rmse_test = Axis(
        grid_plot[1, 1],
        xtrimspine=true,
        ytrimspine=false,
        topspinevisible = false,
        rightspinevisible = false,
        xgridvisible = false,
        ygridvisible = false,
        xticks=loaded_results["nval_list"][begin:2:end],
        xlabel="Low-rank dimension",
        aspect=1.2,
        # ylabel="Error",
        title=rich(
            rich("A.  ", font="Times New Roman bold", fontsize=BASE_FONTSIZE-1), "vs. test data         "
        ),
        titlegap=2.0,
        titlesize=BASE_FONTSIZE-2,
        titlealign=:center,
    )
    ax_rmse_kf = Axis(
        grid_plot[1, 2],
        xtrimspine=true,
        ytrimspine=(true, false),
        topspinevisible = false,
        rightspinevisible = false,
        xgridvisible = false,
        ygridvisible = false,
        xticks=loaded_results["nval_list"][begin:2:end],
        xlabel="Low-rank dimension",
        aspect=1.2,
        # ylabel="RMSE",
        title=rich(
            rich("B.  ", font="Times New Roman bold", fontsize=BASE_FONTSIZE-1), "vs. KF mean       "
        ),
        titlegap=2.0,
        titlesize=BASE_FONTSIZE-2,
        titlealign=:center,
    )
    ax_cov_dist = Axis(
        grid_plot[1, 3],
        xtrimspine=true,
        ytrimspine=(true, false),
        topspinevisible = false,
        rightspinevisible = false,
        xgridvisible = false,
        ygridvisible = false,
        xticks=loaded_results["nval_list"][begin:2:end],
        xlabel="Low-rank dimension",
        aspect=1.2,
        title=rich(
            rich("C.  ", font="Times New Roman bold", fontsize=BASE_FONTSIZE-1), "vs. KF covariance "
        ),
        titlegap=2.0,
        titlesize=BASE_FONTSIZE-2,
        titlealign=:center,
        # ylabel="Frobenius distance"
    )

    kf_legend_handle = hlines!(ax_rmse_test, loaded_results["kf"]["rmse_to_test"], xmin=0.045, xmax=0.955, color=COLORS[4], label="KF")
    # lines!(ax_rmse_test, loaded_results["nval_list"], loaded_results["rrkf"]["rmse_to_test"])
    rrkf_legend_handle = scatterlines!(ax_rmse_test, loaded_results["nval_list"], loaded_results["rrkf"]["rmse_to_test"], label="SKF", marker=:diamond, color=COLORS[3])
    # lines!(ax_rmse_test, loaded_results["nval_list"], loaded_results["enkf"]["rmse_to_test"])
    enkf_legend_handle = scatterlines!(ax_rmse_test, loaded_results["nval_list"], loaded_results["enkf"]["rmse_to_test"], label="EnKF", marker=:rect, color=COLORS[2])
    # lines!(ax_rmse_test, loaded_results["nval_list"], loaded_results["etkf"]["rmse_to_test"])
    etkf_legend_handle = scatterlines!(ax_rmse_test, loaded_results["nval_list"], loaded_results["etkf"]["rmse_to_test"], label="dEnKF", color=COLORS[1])

    ylims!(ax_rmse_test, low=7.5)

    # lines!(ax_rmse_kf, loaded_results["nval_list"], loaded_results["rrkf"]["rmse_to_kf"])
    # lines!(ax_rmse_kf, loaded_results["nval_list"], loaded_results["enkf"]["rmse_to_kf"])
    # lines!(ax_rmse_kf, loaded_results["nval_list"], loaded_results["etkf"]["rmse_to_kf"])
    scatterlines!(ax_rmse_kf, loaded_results["nval_list"], loaded_results["enkf"]["rmse_to_kf"], label="EnKF", marker=:rect, color=COLORS[2])
    scatterlines!(ax_rmse_kf, loaded_results["nval_list"], loaded_results["etkf"]["rmse_to_kf"], label="dEnKF", color=COLORS[1])
    scatterlines!(ax_rmse_kf, loaded_results["nval_list"], loaded_results["rrkf"]["rmse_to_kf"], label="SKF", marker=:diamond, color=COLORS[3])

    # lines!(ax_cov_dist, loaded_results["nval_list"], loaded_results["rrkf"]["cov_distance"])
    # lines!(ax_cov_dist, loaded_results["nval_list"], loaded_results["enkf"]["cov_distance"])
    # lines!(ax_cov_dist, loaded_results["nval_list"], loaded_results["etkf"]["cov_distance"])
    scatterlines!(ax_cov_dist, loaded_results["nval_list"], loaded_results["enkf"]["cov_distance"], label="EnKF", marker=:rect, color=COLORS[2])
    scatterlines!(ax_cov_dist, loaded_results["nval_list"], loaded_results["etkf"]["cov_distance"], label="dEnKF", color=COLORS[1])
    scatterlines!(ax_cov_dist, loaded_results["nval_list"], loaded_results["rrkf"]["cov_distance"], label="SKF", marker=:diamond, color=COLORS[3])
    lastpoint_legend_handle = CairoMakie.scatter!(
        ax_rmse_test,
        loaded_results["nval_list"][end:end],
        loaded_results["rrkf"]["rmse_to_test"][end:end],
        marker=:diamond,
        color=COLORS[4],
        strokecolor=:black,
        strokewidth=1,
        zorder=10,
        # glowwidth=20,
        # glowcolor=PN_COLORS[1],
        )
    CairoMakie.scatter!(
        ax_rmse_kf,
        loaded_results["nval_list"][end:end],
        loaded_results["rrkf"]["rmse_to_kf"][end:end],
        marker=:diamond,
        color=COLORS[4],
        strokecolor=:black,
        strokewidth=1,
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
        strokewidth=1,
        zorder=10,
        # glowwidth=20,
        # glowcolor=PN_COLORS[1],
    )

    # Legend(grid_plot[1, 4], [rrkf_legend_handle, enkf_legend_handle, etkf_legend_handle, kf_legend_handle], ["ours", "EnKF", "dEnKF", "KF"])
    Legend(grid_plot[1,4], [rrkf_legend_handle, lastpoint_legend_handle, enkf_legend_handle, etkf_legend_handle, kf_legend_handle], ["ours", "ours (r = n)", "EnKF", "ETKF", "KF"])#, position=(-4.0, -1.0))

    # Label(
    #     grid_plot[1, 1, TopLeft()],
    #     "A.",
    #     font = "Times New Roman bold",
    #     padding = (0, 0, 0, 0),
    #     halign = :right,
    #     fontsize=BASE_FONTSIZE - 1,
    #     )
    # Label(
    #     grid_plot[1, 1, Top()],
    #     "vs. test data",
    #     font = "Times New Roman regular",
    #     padding = (5, 0, 0, 0),
    #     halign = :left,
    #     fontsize=BASE_FONTSIZE - 2,
    #     valign=:bottom,
    #     )
    # Label(
    #     grid_plot[1, 2, TopLeft()],
    #     "B.",
    #     font = "Times New Roman bold",
    #     padding = (0, 0, 0, 0),
    #     halign = :right,
    #     fontsize=BASE_FONTSIZE - 1,
    #     )
    # Label(
    #     grid_plot[1, 2, Top()],
    #     "vs. KF mean",
    #     font = "Times New Roman regular",
    #     padding = (5, 0, 0, 0),
    #     halign = :left,
    #     fontsize=BASE_FONTSIZE - 2,
    #     valign=:bottom,
    #     )
    # Label(
    #     grid_plot[1, 3, TopLeft()],
    #     "C.",
    #     font = "Times New Roman bold",
    #     padding = (0, 0, 0, 0), #(0, 5, 5, 0),
    #     halign = :right,
    #     fontsize=BASE_FONTSIZE-1
    #     )
    # Label(
    #     grid_plot[1, 3, Top()],
    #     "vs. KF covariance",
    #     font = "Times New Roman regular",
    #     padding = (5, 0, 0, 0),
    #     halign = :left,
    #     fontsize=BASE_FONTSIZE - 2,
    #     valign=:bottom,
    #     )

    # Label(grid_plot[2:3, 0], "Error", rotation=π/2)
    # Label(grid_plot[3, :], L"Low-rank dimension $r$")
    # Label(grid_plot[2, 1], "A", font="Times New Roman bold", halign = :right)
    # Label(grid_plot[2, 3], "B", font="Times New Roman bold", halign = :right)
    # Label(grid_plot[2, 5], "C", font="Times New Roman bold", halign = :right)

    # ylims!(ax_rmse_test, (-0.1, 2.5))
    # ylims!(ax_rmse_kf, (-0.1, 2.5))
    # ylims!(ax_cov_dist, (-0.1, 1.1))


    # rowgap!(grid_plot.layout, 1, 1.0)
    # rowgap!(grid_plot.layout, 2, 1.0)
    # rowgap!(grid_plot.layout, 3, 8.0)
    # colgap!(grid_plot.layout, 1, 0.0)
    # colgap!(grid_plot.layout, 2, 0.0)
    # colgap!(grid_plot.layout, 3, 10.0)
    # colgap!(grid_plot.layout, 4, 0.0)
    # colgap!(grid_plot.layout, 5, 10.0)
    # colgap!(grid_plot.layout, 6, 0.0)

    colsize!(grid_plot.layout, 1, Aspect(1, 1.15))
    colsize!(grid_plot.layout, 2, Aspect(1, 1.15))
    colsize!(grid_plot.layout, 3, Aspect(1, 1.15))
    colsize!(grid_plot.layout, 4, Aspect(1, 0.9))
    colgap!(grid_plot.layout, 12.0)
    # colgap!(grid_plot.layout, 3, 20.0)
    ax_rmse_test.ylabel = "RMSE"
    # ax_rmse_kf.ylabel = "RMSE to KF"
    ax_cov_dist.ylabel = "Frobenius distance"
    ax_rmse_truth.xlabelpadding=0.0
    ax_rmse_kf.xlabelpadding=0.0
    ax_cov_dist.xlabelpadding=0.0

    grid_plot
end

display(fig)
save("./out_new/london_grid_plot.pdf", grid_plot, pt_per_unit = 1)
