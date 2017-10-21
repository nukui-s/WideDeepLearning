import numpy
import tensorflow as tf


class DataProcessor(object):

    def __init__(self, numeric_features=None, vocab_features=None, embedding_features=None, hash_features=None,
                 embedding_dims=None, hash_bucket_sizes=None, dnn_features=None, linear_features=None):
        self.numeric_features = numeric_features or []
        self.vocab_features = vocab_features or []
        self.hash_features = hash_features or []
        self.hash_bucket_sizes = hash_bucket_sizes or {}
        self.embedding_features = embedding_features or []
        self.embedding_dims = embedding_dims or {}
        self.column_names = (self.numeric_features +
                             self.vocab_features +
                             self.hash_features +
                             self.embedding_features)

    def set_columns(self, x_data):
        self.columns = []

        for feature in self.numeric_features:
            column = tf.feature_column.numeric_column(feature)
            self.columns.append(column)

        for feature in self.vocab_features:
            categories = x_data[feature].categories.tolist()
            column = tf.feature_column.categorical_column_with_vocabulary_list(
                        feature, categories)
            self.columns.append(column)

        for feature in self.hash_features:
            bucket_size = self.hash_bucket_sizes[feature]
            if isinstance(x_data[feature].iloc[0], str):
                dtype = tf.string
            else:
                dtype = tf.integer
            column = tf.feature_column.categorical_column_with_hash_bucket(
                         feature, hash_bucket_size=bucket_size, dtype=dtype)
            self.columns.append(column)

        for feature in self.embedding_features:
            categories = x_data[feature].categories.tolist()
            voc_col = tf.feature_column.categorical_column_with_vocabulary_list(
                            feature, categories)
            embed_dim = self.embed_dim.get(feature, numpy.log2(len(categories)))
            column = tf.embedding_column(voc_col, embed_dim)
            self.columns.append(column)

    def get_columns(self):
        return self.columns

    def get_linear_columns(self):
        if self.linear_features is None:
            return self.columns
        else:
            return [col for name, col in zip(self.column_names, self.columns) if name in self.linear_features]

    def get_dnn_columns(self):
        if self.dnn_features is None:
            return self.columns
        else:
            return [col for name, col in zip(self.column_names, self.columns) if name in self.dnn_features]
