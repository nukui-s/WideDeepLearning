import math

import numpy
import pandas
import tensorflow as tf


class DNNLinearCombinedEstimator(object):

    estimator_cls = None
    prediction_key = None

    def __init__(self, linear_dataproc=None, dnn_dataproc=None, linear_optimizer='Ftrl', dnn_hidden_units=(100,),
                 dnn_optimizer='Adagrad', dnn_activation_fn=tf.nn.relu, dnn_dropout=None,
                 num_epochs=1, shuffle=True, batch_size=128, num_threads=1, **kwargs):
        self.linear_dataproc = linear_dataproc
        self.dnn_dataproc = dnn_dataproc
        self.num_epochs = num_epochs
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.num_threads = num_threads
        self.dnn_hidden_units = dnn_hidden_units
        self.dnn_optimizer = dnn_optimizer
        self.dnn_activation_fn = dnn_activation_fn
        self.dnn_dropout = dnn_dropout
        self.model = None
        self.other_args = kwargs

    def fit(self, x_data, t_data):

        assert isinstance(x_data, pandas.DataFrame)
        assert isinstance(t_data, pandas.Series)

        model_args = {'dnn_hidden_units': self.dnn_hidden_units, 'dnn_optimizer': self.dnn_optimizer,
                      'dnn_activation_fn': self.dnn_activation_fn, 'dnn_dropout': self.dnn_dropout}

        if self.linear_dataproc:
            self.linear_dataproc.set_columns(x_data)
            model_args['linear_feature_columns'] = self.linear_dataproc.get_feature_columns()

        if self.dnn_dataproc:
            self.dnn_dataproc.set_columns(x_data)
            model_args['dnn_feature_columns'] = self.dnn_dataproc.get_feature_columns()

        self.model = self.estimator_cls(**model_args, **self.other_args)

        input_fn = self.get_input_fn(x_data, t_data)
        self.model.train(input_fn)
        return self

    def predict(self, x_data):
        input_fn = self.get_input_fn(x_data)
        return numpy.hstack([res[self.prediction_key] for res in self.model.predict(input_fn)])

    def get_input_fn(self, x, y=None):
        x = self.cast_category_into_primitive(x)
        return tf.estimator.inputs.pandas_input_fn(
                    x, y, batch_size=self.batch_size, num_epochs=self.num_epochs,
                    shuffle=self.shuffle, target_column='_target', num_threads=self.num_threads)

    def cast_category_into_primitive(self, x_data):
        _x_data = pandas.DataFrame(index=x_data.index)
        for col in x_data:
            if x_data[col].dtype.name == 'category':
                _x_data[col] = x_data[col].astype(x_data[col].cat.categories.dtype)
            else:
                _x_data[col] = x_data[col]
        return _x_data


class DNNLinearCombinedClassifier(DNNLinearCombinedEstimator):

    estimator_cls = tf.estimator.DNNLinearCombinedClassifier
    prediction_key = 'logistic'


class DNNLinearCombinedRegressor(DNNLinearCombinedEstimator):

    estimator_cls = tf.estimator.DNNLinearCombinedRegressor
    prediction_key = 'predictions'


class LinearClassifier(DNNLinearCombinedClassifier):

    def __init__(self, dataproc, linear_optimizer='Ftrl',
                 num_epochs=1, shuffle=True, batch_size=128, num_threads=1, **kwargs):
        super().__init__(linear_dataproc=dataproc, linear_optimizer=linear_optimizer,
                         num_epochs=num_epochs, shuffle=shuffle, batch_size=batch_size,
                         num_threads=num_threads, **kwargs)


class LinearRegressor(DNNLinearCombinedRegressor):

    def __init__(self, dataproc, linear_optimizer='Ftrl',
                 num_epochs=1, shuffle=True, batch_size=128, num_threads=1, **kwargs):
        super().__init__(linear_dataproc=dataproc, linear_optimizer=linear_optimizer,
                         num_epochs=num_epochs, shuffle=shuffle, batch_size=batch_size,
                         num_threads=num_threads, **kwargs)


class DNNClassifier(DNNLinearCombinedClassifier):

    def __init__(self, dataproc, linear_optimizer='Ftrl',
                 num_epochs=1, shuffle=True, batch_size=128, num_threads=1, **kwargs):
        super().__init__(linear_dataproc=dataproc, linear_optimizer=linear_optimizer,
                         num_epochs=num_epochs, shuffle=shuffle, batch_size=batch_size,
                         num_threads=num_threads, **kwargs)


class DNNClassifier(DNNLinearCombinedClassifier):

    def __init__(self, dataproc, dnn_hidden_units=(100,),
                 dnn_optimizer='Adagrad', dnn_activation_fn=tf.nn.relu, dnn_dropout=None,
                 num_epochs=1, shuffle=True, batch_size=128, num_threads=1, **kwargs):
        super().__init__(dnn_dataproc=dataproc, dnn_hidden_units=dnn_hidden_units,
                         dnn_optimizer=dnn_optimizer, dnn_activation_fn=dnn_activation_fn,
                         dnn_dropout=dnn_dropout, num_epochs=num_epochs, shuffle=shuffle,
                         batch_size=batch_size, num_threads=num_threads, **kwargs)


class DNNRegressor(DNNLinearCombinedRegressor):

    def __init__(self, dataproc, dnn_hidden_units=(100,),
                 dnn_optimizer='Adagrad', dnn_activation_fn=tf.nn.relu, dnn_dropout=None,
                 num_epochs=1, shuffle=True, batch_size=128, num_threads=1, **kwargs):
        super().__init__(dnn_dataproc=dataproc, dnn_hidden_units=dnn_hidden_units,
                         dnn_optimizer=dnn_optimizer, dnn_activation_fn=dnn_activation_fn,
                         dnn_dropout=dnn_dropout, num_epochs=num_epochs, shuffle=shuffle,
                         batch_size=batch_size, num_threads=num_threads, **kwargs)
