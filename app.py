import dash
from dash import html, dcc
import numpy as np
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from simulation import Simulation

external_stylesheets = ['app_styles.css']

palette = {
    "Food": "orange",
    "Prey": "blue",
    "Predator": "red"
}

start_food = 100
start_prey = 50
start_predator = 5
sim = Simulation(width=50, height=50,
                 num_agents=[start_food, start_prey, start_predator])
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(
    children=[
        html.H1("Genetic Algorithm Simulation of Agents"),
        html.Div(
            children=[
                html.Button("Restart simulation",
                            id="restart-button", className="restart-btn")
            ]),
        html.Div(
            children=[
                dcc.Graph(id="simulation-graph", className="graph"),
                html.Div(
                    children=[
                        dcc.Graph(id="num-agents-graph", className="graph"),
                        dcc.Markdown(id="log", className="log")
                    ]
                )
            ],
            style={"display": "flex"}
        ),
        dcc.Interval(id="simulation-interval",
                     interval=200,
                     n_intervals=0)
    ]
)


@ app.callback(
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
                            color=[palette[agent.type]
                                   for agent in sim.agents],
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
                range=[0, sim.height + 2],
                showline=False,
                showgrid=False,
                showticklabels=False,
                zeroline=False
            ),
            height=800,
            width=800,
            annotations=[
                go.layout.Annotation(
                    x=0,
                    y=sim.height + 1,
                    xref="x",
                    yref="y",
                    xanchor="left",
                    text=f"Epoch: {sim.current_epoch}",
                    showarrow=False,
                    font=dict(size=20, family="Fira Sans")
                )
            ]
        )
    )


@ app.callback(
    Output("restart-button", "n_clicks"),
    Output("simulation-interval", "n_intervals"),
    Input("restart-button", "n_clicks")
)
def restart_simulation(n):
    global sim
    sim = Simulation(50, 50, [100,  # Food
                              50,  # Prey
                              50])  # Predator
    return 0, 0


@ app.callback(
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
                name="Food",
                line=dict(color=palette["Food"])
            ),
            go.Scatter(
                x=sim.history["Epoch"],
                y=sim.history["Prey"],
                mode="lines",
                name="Prey",
                line=dict(color=palette["Prey"])
            ),
            go.Scatter(
                x=sim.history["Epoch"],
                y=sim.history["Predator"],
                mode="lines",
                name="Predator",
                line=dict(color=palette["Predator"])
            )],
        layout=go.Layout(
            xaxis=dict(title="Epoch"),
            yaxis=dict(title="Number of Agents"),
            height=400,
            width=800,
            annotations=[
                go.Annotation(
                    x=0,
                    y=0,
                    xref="x",
                    yref="y",
                    xanchor="left",
                    text=f"{sim.history['Prey'][-1]} preys, {sim.history['Predator'][-1]} predators, {sim.history['Food'][-1]} food",
                    showarrow=False,
                    font=dict(size=15, color="gray", family="Fira Sans")
                )
            ]
        )
    )

@app.callback(
    Output("log", "children"),
    Input("simulation-interval", "n_intervals")
)

def update_log(n):
    """
    Updates the log with the current epoch and number of agents of each type
    """
    return "\n".join(sim.log_messages)


if __name__ == "__main__":
    app.run_server(debug=True)
