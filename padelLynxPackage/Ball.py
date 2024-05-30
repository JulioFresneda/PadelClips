class Ball:
    def get_ball_features(self, add_nan=False):
        ball_positions = {}
        for frame in self.frames:
            if frame.ball_position != None:
                ball_positions[frame.frame_number] = frame.ball_position
            elif add_nan:
                ball_positions[frame.frame_number] = ObjectPosition(0, np.nan, np.nan, np.nan, np.nan)

        return self.extract_features(ball_positions)

    def extract_features(self, positions):
        x_positions = np.array([pos.x for pos in positions.values()])
        y_positions = np.array([pos.y for pos in positions.values()])
        size = np.array([pos.size for pos in positions.values()])
        frame_numbers = np.array([frame for frame in positions.keys()])

        velocities = np.diff(np.stack((x_positions, y_positions), axis=1), axis=0)

        return x_positions, y_positions, frame_numbers, size

    def cluster_ball_features(self):
        x_positions, y_positions, frame_numbers, velocities = self.extract_features(self.ball_positions)

        # positions = np.stack((x_positions, y_positions), axis=-1)

        seed = 0
        np.random.seed(seed)

        X_train = TimeSeriesResampler(sz=40).fit_transform(y_positions)
        sz = X_train.shape[1]

        # Euclidean k-means
        print("Euclidean k-means")
        km = TimeSeriesKMeans(n_clusters=9, verbose=True, random_state=seed)
        y_pred = km.fit_predict(X_train)

        plt.figure()
        for yi in range(9):
            plt.subplot(3, 3, yi + 1)
            for xx in X_train[y_pred == yi]:
                plt.plot(xx.ravel(), "k-", alpha=.2)
            plt.plot(km.cluster_centers_[yi].ravel(), "r-")
            plt.xlim(0, sz)
            plt.ylim(-1, 1)
            plt.text(0.55, 0.85, 'Cluster %d' % (yi + 1),
                     transform=plt.gca().transAxes)
            if yi == 1:
                plt.title("Euclidean $k$-means")

        plt.tight_layout()
        plt.show()