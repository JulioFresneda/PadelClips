import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler

from padelLynxPackage.Game import *

def scale_data(data):
    min_val = np.min(data)
    max_val = np.max(data)
    return 2 * (data - min_val) / (max_val - min_val) - 1





def players_to_dataframe(game, clean = True):
    dfs = []
    for i in range(4):
        x_positions, y_positions, frames, _ = game.get_player_features(i)
        df = {'frame': frames, 'player_' + str(i) + '_x': x_positions, 'player_' + str(i) + '_y':y_positions}
        df = pd.DataFrame(df)
        df.set_index('frame', inplace=True)
        dfs.append(df)

    combined_df = pd.concat(dfs, axis=1)



    if clean:
        combined_df = combined_df.dropna()

    print(combined_df.head())

    return combined_df






def print_player_heatmap(game: Game, player_number):
    x_positions, y_positions, _, _ = game.get_player_features(player_number)

    # Scale x and y positions
    #x_scaled = scale_data(x_positions)
    #y_scaled = scale_data(y_positions)

    # Plotting the heatmap
    plt.figure(figsize=(8, 6))
    heatmap, xedges, yedges = np.histogram2d(x_positions, y_positions, bins=50)



    plt.imshow(heatmap.T, extent=[-0, 1, -0, 1], origin='lower', cmap='viridis')
    plt.colorbar(label='Counts')
    plt.title('Heatmap of Player Positions')
    plt.xlabel('Scaled X Position')
    plt.ylabel('Scaled Y Position')
    plt.show()






def cluster_positions_elbow_method(game):
    X = players_to_dataframe(
        game)  # Ensure this function returns a properly formatted DataFrame suitable for clustering
    silhouette_scores = []
    k_values = range(2, 20)  # Define range of k

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(X)
        score = silhouette_score(X, kmeans.labels_)
        silhouette_scores.append(score)

    plt.figure(figsize=(10, 4))
    plt.plot(k_values, silhouette_scores, 'bx-')  # Corrected to use k_values instead of 10
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score for each k')
    plt.show()

def cluster_positions(game, n_clusters):
    X = players_to_dataframe(game)
    players = ['player_0', 'player_1', 'player_2', 'player_3']
    features = [f"{player}_x" for player in players] + [f"{player}_y" for player in players]

    scaler = MinMaxScaler()
    #X[features] = scaler.fit_transform(X[features])

    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(X[features])
    X['cluster'] = kmeans.labels_

    # Print cluster counts
    cluster_counts = X['cluster'].value_counts()
    print(cluster_counts)

    # Set up a figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))  # Adjust layout/grid dimensions based on the number of players
    axes = axes.flatten()  # Flatten the axes array for easier iteration

    # Different colormaps for each player can be used if desired
    colormaps = ['Reds', 'Blues', 'Greens', 'Purples']

    for idx, player in enumerate(players):
        ax = axes[idx]
        cmap = plt.get_cmap(colormaps[idx])
        scatter = ax.scatter(X[f'{player}_x'], X[f'{player}_y'], c=X['cluster'], cmap=cmap, label=f'{player}',
                             alpha=0.6, edgecolor='k')
        ax.set_title(f'Position Clustering for {player}')
        ax.set_xlabel('X Position (scaled)')
        ax.set_ylabel('Y Position (scaled)')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.legend()

    # Adding a single colorbar for the figure
    fig.colorbar(scatter, ax=axes, orientation='vertical', fraction=.02)
    plt.tight_layout()
    plt.show()


