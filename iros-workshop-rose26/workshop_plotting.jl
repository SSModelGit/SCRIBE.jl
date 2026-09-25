using Match: @match
using Plots

const WORKSHOP_TOPOLOGY_COLORS = Dict(
    :sparse => "#D55E00",
    :moderate => "#0072B2",
    :dense => "#009E73",
)

workshop_plot_style(profile=:paper) = @match profile begin
    :paper => (
        width=1400,
        wide_height=590,
        spatial_height=740,
        titlefontsize=16,
        guidefontsize=15,
        tickfontsize=12,
        legendfontsize=12,
        annotationfontsize=12,
        linewidth=2.8,
        markersize=5.5,
        ribbonalpha=0.17,
    )
    :poster => (
        width=2000,
        wide_height=850,
        spatial_height=1060,
        titlefontsize=22,
        guidefontsize=20,
        tickfontsize=17,
        legendfontsize=17,
        annotationfontsize=17,
        linewidth=4.0,
        markersize=8.0,
        ribbonalpha=0.19,
    )
    _ => throw(ArgumentError("Use the `paper` or `poster` rendering profile."))
end

function workshop_panel!(
    panel;
    profile=:paper,
    left_margin=7Plots.mm,
    right_margin=3Plots.mm,
    bottom_margin=6Plots.mm,
    top_margin=4Plots.mm,
)
    style = workshop_plot_style(profile)
    plot!(
        panel;
        titlefontsize=style.titlefontsize,
        guidefontsize=style.guidefontsize,
        tickfontsize=style.tickfontsize,
        legendfontsize=style.legendfontsize,
        gridalpha=0.13,
        foreground_color_grid=:gray75,
        left_margin,
        right_margin,
        bottom_margin,
        top_margin,
    )
    panel
end

function save_workshop_figure(figure, output_base)
    png_path = output_base * ".png"
    savefig(figure, png_path)
    println("Figure saved at $png_path")
    png_path
end
