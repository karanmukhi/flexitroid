import numpy as np
from itertools import permutations
import math
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy.spatial import ConvexHull
import plotly.io as pio

# pio.templates.default = "plotly_dark"
import flexitroid.utils as utils


def plot_flex(vertices):
    if np.shape(vertices)[1] == 2:
        plot2d(vertices)
    elif np.shape(vertices)[1] == 3:
        plot3d(vertices)
    else:
        raise ValueError("Cannot plot flex with T > 3")


def plot2d(vertices):
    points = vertices
    # Calculate convex hull

    if np.shape(vertices)[0] > 2:
        hull = ConvexHull(points)
        hull_vertices = np.append(hull.vertices, hull.vertices[0])  # Close the loop
        trace_hull = go.Scatter(
            x=points[hull_vertices, 0],
            y=points[hull_vertices, 1],
            mode="lines+markers",
            fill="toself",  # Fill the area inside the convex hull
            fillcolor="rgba(0,100,80,0.2)",  # Set fill color and opacity
            line=dict(color="blue"),
        )
    else:
        trace_hull = go.Scatter(
            x=vertices[:, 0],
            y=vertices[:, 1],
        )

    # Create figure
    fig = go.Figure(data=[trace_hull])
    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
    )
    # Show the figure
    fig.show()


def plot3d_surface(vertices):
    points_3d = vertices

    fig = go.Figure()
    fig.add_trace(
        go.Mesh3d(
            x=points_3d[:, 0],
            y=points_3d[:, 1],
            z=points_3d[:, 2],
            color="blue",
            opacity=0.5,
            showscale=False,
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=points_3d[:, 0],
            y=points_3d[:, 1],
            z=points_3d[:, 2],
            mode="markers",
            marker=dict(color="red", size=8),
            name="Data Points",
        )
    )

    # Create figure
    fig.show()


def plot3d(vertices):
    # Sample data points in 3D
    points_3d = vertices

    fig = go.Figure()
    fig.add_trace(
        go.Mesh3d(
            x=points_3d[:, 0],
            y=points_3d[:, 1],
            z=points_3d[:, 2],
            color="blue",
            opacity=0.2,
            alphahull=0,
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=points_3d[:, 0],
            y=points_3d[:, 1],
            z=points_3d[:, 2],
            mode="markers",
            marker=dict(color="red", size=4),
            name="Data Points",
        )
    )

    # Create layout
    layout_3d = go.Layout(
        title="Convex Hull in 3D",
        scene=dict(
            xaxis=dict(title="X-axis"),
            yaxis=dict(title="Y-axis"),
            zaxis=dict(title="Z-axis"),
        ),
    )

    # Create figure
    fig.show()


def plot3d_ambient(polytopes, gamma=None):
    # Sample data points in 3D
    fig = go.Figure()
    for vertices in polytopes:
        fig.add_trace(
            go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                opacity=0.2,
                alphahull=0,
                showlegend=True,
            )
        )

        if np.shape(vertices)[0] == np.shape(vertices)[1] + 1:
            vertices = utils.visit_edges(vertices)
            fig.add_trace(
                go.Scatter3d(
                    x=vertices[:, 0],
                    y=vertices[:, 1],
                    z=vertices[:, 2],
                    mode="lines+markers",
                    line=dict(color="black", width=2),
                    marker=dict(size=5, color="red"),
                )
            )
        else:
            fig.add_trace(
                go.Scatter3d(
                    x=vertices[:, 0],
                    y=vertices[:, 1],
                    z=vertices[:, 2],
                    mode="markers",
                    marker=dict(size=5),
                )
            )

    # Create figure
    fig.show()
    fig.write_html("output.html")


def plot3d_multi(polytopes, gamma=None, normal=None):
    # Sample data points in 3D
    fig = go.Figure()
    for vertices in polytopes:
        fig.add_trace(
            go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                opacity=0.2,
                alphahull=0,
                showlegend=True,
            )
        )

        if np.shape(vertices)[0] == np.shape(vertices)[1] + 1:
            vertices = utils.visit_edges(vertices)
            fig.add_trace(
                go.Scatter3d(
                    x=vertices[:, 0],
                    y=vertices[:, 1],
                    z=vertices[:, 2],
                    mode="lines+markers",
                    line=dict(color="black", width=2),
                    marker=dict(size=5, color="red"),
                )
            )
        else:
            fig.add_trace(
                go.Scatter3d(
                    x=vertices[:, 0],
                    y=vertices[:, 1],
                    z=vertices[:, 2],
                    mode="markers",
                    marker=dict(size=5),
                )
            )

    # Create figure
    fig.show()
    fig.write_html("output.html")


def plot_lifted(A, simplex, gamma, p, opt):
    fig = go.Figure()

    T = A.shape[1]
    points_3d = A
    fig.add_trace(
        go.Mesh3d(
            x=points_3d[:, 0],
            y=points_3d[:, 1],
            z=points_3d[:, 2],
            color="blue",
            opacity=0.2,
            showscale=False,
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=points_3d[:, 0],
            y=points_3d[:, 1],
            z=points_3d[:, 2],
            mode="markers",
            marker=dict(color="red", size=4),
            name="Permutahedron",
        )
    )

    points_3d = simplex

    fig.add_trace(
        go.Mesh3d(
            x=points_3d[:, 0],
            y=points_3d[:, 1],
            z=points_3d[:, 2],
            color="blue",
            opacity=0.2,
            showscale=False,
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=points_3d[:, 0],
            y=points_3d[:, 1],
            z=points_3d[:, 2],
            mode="markers",
            marker=dict(color="green", size=4),
            name="Delta",
        )
    )

    points_3d = gamma

    fig.add_trace(
        go.Mesh3d(
            x=points_3d[:, 0],
            y=points_3d[:, 1],
            z=points_3d[:, 2],
            color="blue",
            opacity=0.2,
            showscale=False,
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=points_3d[:, 0],
            y=points_3d[:, 1],
            z=points_3d[:, 2],
            mode="lines+markers",
            line=dict(color="greenyellow", width=4),
            marker=dict(color="chartreuse", size=5),
            name="Gamma",
        )
    )

    points_3d = p.reshape(1, T)
    fig.add_trace(
        go.Scatter3d(
            x=points_3d[:, 0],
            y=points_3d[:, 1],
            z=points_3d[:, 2],
            mode="markers",
            marker=dict(color="blue", size=4),
            name="Real Opt",
        )
    )

    points_3d = opt.reshape(1, T)
    fig.add_trace(
        go.Scatter3d(
            x=points_3d[:, 0],
            y=points_3d[:, 1],
            z=points_3d[:, 2],
            mode="markers",
            marker=dict(color="purple", size=4),
            name="Constrained Opt",
        )
    )

    normal = -utils.find_normal(utils.project(gamma))
    p_gamma = utils.project(gamma)
    norm_data = utils.lift(np.array([p_gamma[0], p_gamma[0] + normal]))

    fig.add_trace(
        go.Scatter3d(
            x=norm_data[:, 0],
            y=norm_data[:, 1],
            z=norm_data[:, 2],
            mode="lines",
            line=dict(color="green", width=4),
            name="Normal",
        )
    )
    fig.show()


def plot2d_multi(polytopes, p_gamma=None, plot_vertices=True):
    data = []
    for vertices in polytopes:
        points = vertices
        # Calculate convex hull

        if np.shape(vertices)[0] > 2:
            hull = ConvexHull(points)

            # Get hull vertices
            hull_vertices = np.append(hull.vertices, hull.vertices[0])  # Close the loop

            if plot_vertices:
                # Create scatter trace for the convex hull
                trace_hull = go.Scatter(
                    x=points[hull_vertices, 0],
                    y=points[hull_vertices, 1],
                    mode="lines+markers",
                    fill="toself",  # Fill the area inside the convex hull
                    fillcolor="rgba(0,100,80,0.1)",  # Set fill color and opacity
                    line=dict(color="rgba(0, 0, 0, 0.5)"),
                    # line=None
                )
            else:
                trace_hull = go.Scatter(
                    x=points[hull_vertices, 0],
                    y=points[hull_vertices, 1],
                    mode="lines+markers",
                    fill="toself",  # Fill the area inside the convex hull
                    fillcolor=f"rgba(0,100,80, {1/len(polytopes)})",  # Set fill color and opacity
                    line=dict(color="rgba(0, 0, 0, 0.0005)"),
                    # line=None
                )
        else:
            trace_hull = go.Scatter(
                x=vertices[:, 0],
                y=vertices[:, 1],
            )

        data.append(trace_hull)
        fig = go.Figure(data=data)
        if p_gamma is not None:
            normal = utils.find_normal(p_gamma)

            norm_data = np.array([p_gamma[0], p_gamma[0] + normal])

            norm_plot = go.Scatter(x=norm_data[:, 0], y=norm_data[:, 1])
            fig.add_trace(norm_plot)

    # Create figure

    fig.update_layout(
        yaxis=dict(
            scaleanchor="x",  # Anchors y-axis scaling to x-axis
            scaleratio=1      # Ensures 1:1 aspect ratio
        ),
        width=600,
        height=600
    )
    # Show the figure
    fig.show()
