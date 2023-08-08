using JLD2
# using TuePlots
using CairoMakie
using Statistics
using BenchmarkTools
using LaTeXStrings

CairoMakie.activate!(type = "svg")
cd("./experiments/err_runtime")

loaded_LA_results = load("./out/LA_err_runtime.jld2")
# loaded_setup = load("./out/setup_dict.jld2")


include("../plot_theme.jl")


WIDTH, HEIGHT = FULL_WIDTH, FULL_HEIGHT

to_seconds(tm) = tm / 1e9


fig = begin
    grid_plot = Figure(;
        resolution=(WIDTH, HEIGHT),
        figure_padding=1,
    )

    ax_rmse_kf = Axis(
        grid_plot[1, 1],
        xtrimspine=(false, false),
        ytrimspine=(false, false),
        topspinevisible = false,
        rightspinevisible = false,
        xgridvisible = false,
        ygridvisible = false,
        # xticks=loaded_LA_results["nval_list"][begin:4:end],
        # xlabel="Low-rank dim.",
        aspect=1.0,
        # ylabel="RMSE",
        title="Error vs. runtime",
        titlegap=2.0,
        titlesize=BASE_FONTSIZE-2,
        titlealign=:center,
    )

    rrkf_rmse_to_kf = loaded_LA_results["rrkf"]["rmse_to_kf"]
    enkf_rmse_to_kf = loaded_LA_results["enkf"]["rmse_to_kf"]
    etkf_rmse_to_kf = loaded_LA_results["etkf"]["rmse_to_kf"]

    rrkf_cov_distance = loaded_LA_results["rrkf"]["cov_distance"]
    enkf_cov_distance = loaded_LA_results["enkf"]["cov_distance"]
    etkf_cov_distance = loaded_LA_results["etkf"]["cov_distance"]




    enkf_legend_handle = scatterlines!(
        ax_rmse_kf,
        collect(map(to_seconds ∘ time ∘ mean, loaded_LA_results["enkf"]["bench"])),
        enkf_rmse_to_kf,
        marker=:rect, color=COLORS[2]
    )



    etkf_legend_handle = scatterlines!(
        ax_rmse_kf,
        collect(map(to_seconds ∘ time ∘ mean, loaded_LA_results["etkf"]["bench"])),
        etkf_rmse_to_kf,
        color=COLORS[1]
    )

    rrkf_legend_handle = scatterlines!(
        ax_rmse_kf,
        collect(map(to_seconds ∘ time ∘ mean, loaded_LA_results["rrkf"]["bench"])),
        rrkf_rmse_to_kf,
        marker=:diamond, color=COLORS[3]
    )



    # scatterlines!(
    #     ax_cov_dist,
    #     loaded_LA_results["nval_list"],
    #     enkf_cov_distance,
    #     marker=:rect, color=COLORS[2]
    # )


    # scatterlines!(
    #     ax_cov_dist,
    #     loaded_LA_results["nval_list"],
    #     etkf_cov_distance,
    #     label="etkf", color=COLORS[1]
    # )

    # scatterlines!(
    #     ax_cov_dist,
    #     loaded_LA_results["nval_list"],
    #     rrkf_cov_distance,
    #     marker=:diamond, color=COLORS[3]
    # )


    # CairoMakie.scatter!(
    #     ax_rmse_kf,
    #     loaded_LA_results["nval_list"][end-4:end],
    #     loaded_LA_results["rrkf"]["rmse_to_kf"][end-4:end],
    #     marker=:diamond,
    #     color=COLORS[4],
    #     strokecolor=:black,
    #     strokewidth=0.5,
    #     zorder=10,
    #     # glowwidth=20,
    #     # glowcolor=PN_COLORS[1],
    # )
    # CairoMakie.scatter!(
    #     ax_cov_dist,
    #     loaded_LA_results["nval_list"][end-4:end],
    #     loaded_LA_results["rrkf"]["cov_distance"][end-4:end],
    #     marker=:diamond,
    #     color=COLORS[4],
    #     strokecolor=:black,
    #     strokewidth=0.5,
    #     zorder=10,
    #     # glowwidth=20,
    #     # glowcolor=PN_COLORS[1],
    # )



    # xlims!(ax_rmse_kf, high=110)
    # xlims!(ax_cov_dist, high=110)

    ax_rmse_kf.ylabel = "RMSE"
    ax_rmse_kf.xlabelpadding=0.0

    Label(grid_plot[1, :, Bottom()], "Wall time [sec] ", fontsize=BASE_FONTSIZE-1, font="Times New Roman regular", halign = :center, valign=:top, padding=(0, 0, 0, 15))

    # resize_to_layout!(grid_plot)

    grid_plot
end

display(fig)
save("./out/LA_err_runtime.pdf", grid_plot, pt_per_unit = 1)
