from keras.callbacks import *
import time


class ProgbarLogger(Callback):
    """Callback that prints metrics to stdout.

    # Arguments
        count_mode: One of "steps" or "samples".
            Whether the progress bar should
            count samples seens or steps (batches) seen.

    # Raises
        ValueError: In case of invalid `count_mode`.
    """

    def __init__(self, count_mode='samples'):
        super(ProgbarLogger, self).__init__()
        if count_mode == 'samples':
            self.use_steps = False
        elif count_mode == 'steps':
            self.use_steps = True
        else:
            raise ValueError('Unknown `count_mode`: ' + str(count_mode))

    def on_train_begin(self, logs=None):
        self.verbose = self.params['verbose'] or 1
        self.epochs = self.params['epochs']

    def on_epoch_begin(self, epoch, logs=None):
        if self.verbose:
            print('Epoch %d/%d' % (epoch + 1, self.epochs))
            if self.use_steps:
                target = self.params['steps']
            else:
                target = self.params['samples']
            self.target = target
            self.progbar = Progbar(target=self.target,
                                   verbose=self.verbose)
        self.seen = 0

    def on_batch_begin(self, batch, logs=None):
        if self.seen < self.target:
            self.log_values = []

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        batch_size = logs.get('size', 0)
        if self.use_steps:
            self.seen += 1
        else:
            self.seen += batch_size

        self.log_values.append(('time', time.time()))
        for k in self.params['metrics']:
            if k in logs:
                self.log_values.append((k, logs[k]))

        # Skip progbar update for the last batch;
        # will be handled by on_epoch_end.
        if self.verbose and self.seen < self.target:
            self.progbar.update(self.seen, self.log_values)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for k in self.params['metrics']:
            if k in logs:
                self.log_values.append((k, logs[k]))
        if self.verbose:
            self.progbar.update(self.seen, self.log_values, force=True)
