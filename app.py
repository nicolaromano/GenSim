import dash
from dash import html, dcc

from dash.dependencies import Input, Output
import plotly.graph_objects as go
from simulation import Simulation

sim = Simulation(100, 100, [10,  # Food
                            25,  # Pray
                            50])  # Predator
app = dash.Dash(__name__)

app.layout = html.Div(
    children=[
        html.H1("Genetic Algorithm Simulation of Agents"),
        html.Div(id="epoch", children="Epoch: 0"),
        html.Div(
            children=[
                dcc.Graph(id="simulation-graph", className="graph"),
                dcc.Graph(id="num-agents-graph", className="graph")
            ],
            style={"display": "flex"}
        ),
        dcc.Interval(id="simulation-interval",
                     interval=100,
                     n_intervals=0)
    ]
)

@app.callback(
    Output("simulation-graph", "figure"),
    Input("simulation-interval", "n_intervals")
)
def updateplot(n):
    sim.step()

    return go.Figure(
        data=[
            go.Scatter(
                x=[agent.x for agent in sim.agents],
                y=[agent.y for agent in sim.agents],
                mode="markers",
                marker=dict(size=[10 if agent.type == "Predator" else 5 if agent.type == "Food" else 8 for agent in sim.agents],
                            color=["red" if agent.type == "Predator" else "orange" if agent.type ==
                                   "Food" else "blue" for agent in sim.agents],
                            opacity=[min(1.0, agent.energy / 100)
                                     for agent in sim.agents]
                            )
            )
        ],
        layout=go.Layout(
            xaxis=dict(range=[0, sim.width],
                       showline=False,
                       showgrid=False,
                       showticklabels=False,
                       zeroline=False
                       ),
            yaxis=dict(
                range=[0, sim.height],
                showline=False,
                showgrid=False,
                showticklabels=False,
                zeroline=False
            ),
            height=800,
            width=800
        )
    )


@app.callback(
    Output("num-agents-graph", "figure"),
    Input("simulation-interval", "n_intervals")
)
def update_num_agents(n):
    """
    Updates the plot showing the number of agents of each type

    This is a line plot with epoch on the x axis and number of agents on the y axis. It has a line for each agent type.
    """
    return go.Figure(
        data=[
            go.Scatter(
                x=sim.history["Epoch"],
                y=sim.history["Food"],
                mode="lines",
                name="Food"
            ),
            go.Scatter(
                x=sim.history["Epoch"],
                y=sim.history["Pray"],
                mode="lines",
                name="Pray"
            ),
            go.Scatter(
                x=sim.history["Epoch"],
                y=sim.history["Predator"],
                mode="lines",
                name="Predator"
            )
        ],
        layout=go.Layout(
            xaxis=dict(title="Epoch"),
            yaxis=dict(title="Number of Agents"),
            height=400,
            width=800
        )
    )


@app.callback(
    Output("epoch", "children"),
    Input("simulation-interval", "n_intervals")
)
def update_epoch(n):
    return f"Epoch: {sim.current_epoch}"


if __name__ == "__main__":
    app.run_server(debug=True)
