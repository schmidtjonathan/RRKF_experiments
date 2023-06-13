using JLD2
# using TuePlots
using CairoMakie
using LaTeXStrings

CairoMakie.activate!(type = "svg")
cd("/home/jschmidt/.julia/dev/SpectralKF/experiments/LA")

loaded_LA_results = load("./out_new/LA_error_data.jld2")
# loaded_setup = load("./out/setup_dict.jld2")


include("../plot_theme.jl")


WIDTH, HEIGHT = FULL_WIDTH, 0.9HALF_HEIGHT




fig = begin
    grid_plot = Figure(;
        resolution=(WIDTH, HEIGHT),
        figure_padding=1,
    )
    ax_rmse_truth = Axis(
        grid_plot[1, 1],
        xtrimspine=(true, false),
        ytrimspine=false,
        topspinevisible = false,
        rightspinevisible = false,
        xgridvisible = false,
        ygridvisible = false,
        xticks=loaded_LA_results["nval_list"][begin:2:end],
        xlabel="Low-rank dimension",
        aspect=1.2,
        # ylabel="Error",
        title=rich(
            rich("A.  ", font="Times New Roman bold", fontsize=BASE_FONTSIZE-1), "vs. ground truth  "
        ),
        titlegap=2.0,
        titlesize=BASE_FONTSIZE-2,
        titlealign=:center,
    )
    ax_rmse_kf = Axis(
        grid_plot[1, 2],
        xtrimspine=(true, false),
        ytrimspine=(true, false),
        topspinevisible = false,
        rightspinevisible = false,
        xgridvisible = false,
        ygridvisible = false,
        xticks=loaded_LA_results["nval_list"][begin:2:end],
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
        xtrimspine=(true, false),
        ytrimspine=(true, false),
        topspinevisible = false,
        rightspinevisible = false,
        xgridvisible = false,
        ygridvisible = false,
        xticks=loaded_LA_results["nval_list"][begin:2:end],
        xlabel="Low-rank dimension",
        aspect=1.2,
        title=rich(
            rich("C.  ", font="Times New Roman bold", fontsize=BASE_FONTSIZE-1), "vs. KF covariance"
        ),
        titlegap=2.0,
        titlesize=BASE_FONTSIZE-2,
        titlealign=:center,
        # ylabel="Frobenius distance"
    )

    kf_legend_handle = lines!(ax_rmse_truth, loaded_LA_results["nval_list"], loaded_LA_results["kf"]["rmse_to_truth"], color=COLORS[4], label="KF")
    # lines!(ax_rmse_truth, loaded_LA_results["nval_list"], loaded_LA_results["rrkf"]["rmse_to_truth"])
    rrkf_legend_handle = scatterlines!(ax_rmse_truth, loaded_LA_results["nval_list"], loaded_LA_results["rrkf"]["rmse_to_truth"], label="SKF", marker=:diamond, color=COLORS[3])
    # lines!(ax_rmse_truth, loaded_LA_results["nval_list"], loaded_LA_results["enkf"]["rmse_to_truth"])
    enkf_legend_handle = scatterlines!(ax_rmse_truth, loaded_LA_results["nval_list"], loaded_LA_results["enkf"]["rmse_to_truth"], label="EnKF", marker=:rect, color=COLORS[2])
    # lines!(ax_rmse_truth, loaded_LA_results["nval_list"], loaded_LA_results["etkf"]["rmse_to_truth"])
    etkf_legend_handle = scatterlines!(ax_rmse_truth, loaded_LA_results["nval_list"], loaded_LA_results["etkf"]["rmse_to_truth"], label="etkf", color=COLORS[1])

    ylims!(ax_rmse_truth, low=0.0)

    # lines!(ax_rmse_kf, loaded_LA_results["nval_list"], loaded_LA_results["rrkf"]["rmse_to_kf"])
    scatterlines!(ax_rmse_kf, loaded_LA_results["nval_list"], loaded_LA_results["rrkf"]["rmse_to_kf"], label="SKF", marker=:diamond, color=COLORS[3])
    # lines!(ax_rmse_kf, loaded_LA_results["nval_list"], loaded_LA_results["enkf"]["rmse_to_kf"])
    scatterlines!(ax_rmse_kf, loaded_LA_results["nval_list"], loaded_LA_results["enkf"]["rmse_to_kf"], label="EnKF", marker=:rect, color=COLORS[2])
    # lines!(ax_rmse_kf, loaded_LA_results["nval_list"], loaded_LA_results["etkf"]["rmse_to_kf"])
    scatterlines!(ax_rmse_kf, loaded_LA_results["nval_list"], loaded_LA_results["etkf"]["rmse_to_kf"], label="etkf", color=COLORS[1])

    # lines!(ax_cov_dist, loaded_LA_results["nval_list"], loaded_LA_results["rrkf"]["cov_distance"])
    scatterlines!(ax_cov_dist, loaded_LA_results["nval_list"], loaded_LA_results["rrkf"]["cov_distance"], label="SKF", marker=:diamond, color=COLORS[3])
    # lines!(ax_cov_dist, loaded_LA_results["nval_list"], loaded_LA_results["enkf"]["cov_distance"])
    scatterlines!(ax_cov_dist, loaded_LA_results["nval_list"], loaded_LA_results["enkf"]["cov_distance"], label="EnKF", marker=:rect, color=COLORS[2])
    # lines!(ax_cov_dist, loaded_LA_results["nval_list"], loaded_LA_results["etkf"]["cov_distance"])
    scatterlines!(ax_cov_dist, loaded_LA_results["nval_list"], loaded_LA_results["etkf"]["cov_distance"], label="etkf", color=COLORS[1])

    # axislegend(ax_rmse_truth, [rrkf_legend_handle, enkf_legend_handle, etkf_legend_handle, kf_legend_handle], ["ours", "EnKF", "ETKF", "KF"], position=(2.5, -10.25))
    Legend(grid_plot[1, 4], [rrkf_legend_handle, enkf_legend_handle, etkf_legend_handle, kf_legend_handle], ["ours", "EnKF", "ETKF", "KF"])

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
    #     "vs. ground truth",
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

    # ylims!(ax_rmse_truth, (-0.1, 2.5))
    # ylims!(ax_rmse_kf, (-0.1, 2.5))
    # ylims!(ax_cov_dist, (-0.1, 1.1))

    xlims!(ax_rmse_truth, high=105)
    xlims!(ax_rmse_kf, high=105)
    xlims!(ax_cov_dist, high=105)

    # rowgap!(grid_plot.layout, 1, 1.0)
    # rowgap!(grid_plot.layout, 2, 1.0)
    # rowgap!(grid_plot.layout, 3, 8.0)
    # colgap!(grid_plot.layout, 1, 0.0)
    # colgap!(grid_plot.layout, 2, 0.0)
    # colgap!(grid_plot.layout, 3, 10.0)
    # colgap!(grid_plot.layout, 4, 0.0)
    # colgap!(grid_plot.layout, 5, 10.0)
    # colgap!(grid_plot.layout, 6, 0.0)

    # resize_to_layout!(grid_plot)\
    colsize!(grid_plot.layout, 1, Aspect(1, 1.15))
    colsize!(grid_plot.layout, 2, Aspect(1, 1.15))
    colsize!(grid_plot.layout, 3, Aspect(1, 1.15))
    colsize!(grid_plot.layout, 4, Aspect(1, 0.4))
    colgap!(grid_plot.layout, 15.0)
    # colgap!(grid_plot.layout, 3, 20.0)
    ax_rmse_truth.ylabel = "RMSE"
    # ax_rmse_kf.ylabel = "RMSE to KF"
    ax_cov_dist.ylabel = "Frobenius distance"
    ax_rmse_truth.xlabelpadding=0.0
    ax_rmse_kf.xlabelpadding=0.0
    ax_cov_dist.xlabelpadding=0.0

    # resize_to_layout!(grid_plot)

    grid_plot
end

display(fig)
save("./out_new/LA_grid_plot.pdf", grid_plot, pt_per_unit = 1)
