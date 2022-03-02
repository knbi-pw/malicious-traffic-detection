class Story:

    def __init__(self, batch_size, shuffle, reps, history):
        super().__init__()
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.reps = reps
        self.history = history


class Stories:
    def __init__(self, batch_size, shuffle, reps, histories: list):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.reps = reps
        self.history = self.mean_histories(histories)

    @staticmethod
    def mean_histories(histories: list):
        history = {}

        if histories and 'loss' in histories[0]:
            epochs = len(histories[0]['loss'])
            history = dict.fromkeys(histories[0].keys(), [0.0]*epochs)
            # history = defaultdict(lambda: [0.0 * epochs])

            for h in histories:
                for key in h:
                    history[key] = [x + y for x, y in zip(history[key], h[key])]

            for key in history:
                history[key] = [element / len(histories) for element in history[key]]

        return history
