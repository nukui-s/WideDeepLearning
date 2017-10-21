import numpy
import tensorflow as tf


class DNNLinearCombinedEstimator(object):

    estimator_cls = None
    prediction_key = None

    def __init__(self, dataproc, linear_optimizer='Ftrl', dnn_hidden_units=None, dnn_optimizer='Adagrad',
                 dnn_activation_fn=tf.nn.relu, dnn_dropout=None, n_epoch=1, **kwargs):
        self.dataproc = dataproc
        self.n_epoch = n_epoch
        self.prediction_key = prediction_key
        self.model = None,

    def fit(self, x_data, t_data):
        slef.dataproc.set_columns(x_data)

        self.model = self.estimator_cls(
                        linear_feature_columns=self.dataproc.get_linear_columns(),
                        linear_optimizer=linear_optimizer,
                        dnn_feautre_columns=self.dataproc.get_dnn_column(),
                        dnn_optimizer=dnn_optimizer,
                        dnn_activation_fn=dnn_activation_fn,
                        dnn_dropout=dnn_dropout,
                        **kwargs)

        for k in range(self.n_epoch):
            print('---------- {}th epoch ----------'.format(k))
            input_fn = self.dataproc.get_input_fn(x_data, t_data)
            self.model.train(input_fn)
        return self

    def predict(self, x_data):
        input_fn = self.dataproc.get_input_fn(x_data)
        return numpy.hstack([res[self.prediction_key] for res in self.model.predict(input_fn)])


class DNNLinearCombinedClassifier(DNNLinearCombinedEstimator):

    estimator_cls = tf.DNNLinearCombinedClassifier
    prediction_key = 'logistic'


class DNNLinearCombinedRegressor(DNNLinearCombinedEstimator):

    estimator_cls = tf.DNNLinearCombinedRegressor
    prediction_key = 'predictions'
