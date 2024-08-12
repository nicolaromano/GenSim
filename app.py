import dash
from dash import html, dcc
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from simulation import Simulation

external_stylesheets = ['app_styles.css']

palette = {
    "Food": "orange",
    "Prey": "blue",
    "Predator": "red"
}

sim = Simulation()
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
                ),                
                dcc.Markdown(id="top-agents", className="top-agents")
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
def updateplot(n: int) -> go.Figure:
    """
    Updates the plot showing the agents in the simulation
    """
    sim.step()
    
    alive_agents = sim.get_alive_agents()
    top_agents = sim.get_top_agents()
    
    return go.Figure(
        data=[
            go.Scatter(
                x=[agent.x for agent in alive_agents],
                y=[agent.y for agent in alive_agents],
                mode="markers",
                marker=dict(size=[10 if agent.type == "Predator" else 5 if agent.type == "Food" else 8 for agent in alive_agents],
                            color=[palette[agent.type]
                                   for agent in alive_agents],
                            opacity=[min(1.0, agent.energy / 100)
                                     for agent in alive_agents]
                            )
            ),
            # Draw a star on the top 5 agents
            go.Scatter(
                x=[agent.x for agent in top_agents],
                y=[agent.y for agent in top_agents],
                mode="markers",
                marker=dict(size=10,
                            color="green",
                            symbol="star",
                            opacity=1
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
            showlegend=False,
            annotations=[
                go.layout.Annotation(
                    x=0,
                    y=sim.height-1,
                    xref="x",
                    yref="y",
                    xanchor="left",
                    text=f"Epoch: {sim.current_epoch}<br>Generation: {sim.current_generation}",
                    showarrow=False,
                    font=dict(size=14, family="Fira Sans")
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
    sim = Simulation()
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
                x=sim.history[sim.current_generation]["Epoch"],
                y=sim.history[sim.current_generation]["Food"],
                mode="lines",
                name="Food",
                line=dict(color=palette["Food"])
            ),
            go.Scatter(
                x=sim.history[sim.current_generation]["Epoch"],
                y=sim.history[sim.current_generation]["Prey"],
                mode="lines",
                name="Prey",
                line=dict(color=palette["Prey"])
            ),
            go.Scatter(
                x=sim.history[sim.current_generation]["Epoch"],
                y=sim.history[sim.current_generation]["Predator"],
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
                go.layout.Annotation(
                    x=0,
                    y=0,
                    xref="x",
                    yref="y",
                    xanchor="left",
                    text=f"{sim.history[sim.current_generation]['Prey'][-1]} preys, {sim.history[sim.current_generation]['Predator'][-1]} predators, {sim.history[sim.current_generation]['Food'][-1]} food",
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
    return "\n".join(sim.log_messages[-5:])

@app.callback(
    Output("top-agents", "children"),
    Input("simulation-interval", "n_intervals")
)
def update_top_agents(n):
    """
    Updates the markdown with the top agents of the current generation
    """
    top_agents = sim.get_top_agents()
    text = f"## Top agents of generation {sim.current_generation}\n"
    
    for agent in top_agents:
        if agent.alive:
            text += f"### {agent.id}\n"
        else:
            text += f"### {agent.id} (dead)\n"
        text += f"Energy: {agent.energy} - Age: {agent.age} - Food eaten: {agent.food_eaten}\n\n"
        text += f"Fitness: {agent.get_fitness()}\n\n"
        
    return text


if __name__ == "__main__":
    app.run_server(debug=True)
