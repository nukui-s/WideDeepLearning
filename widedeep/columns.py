import itertools

import numpy
import tensorflow as tf


class FeatureColumns(object):
    """Define feature columns used in train"""

    def __init__(self, numeric_features=None, vocab_features=None, embedding_features=None, hash_features=None,
                 embedding_dims=None, hash_bucket_sizes=None):
        """

        Parameters
        ----------
        numeric_features : list of str
            column name of numeric features
        vocab_features : list of str
            column name of vocabrary features
        embedding_features : list of str
            column name of embedding features
        hash_features : list of str
            column name of hash features
        embedding_dims : list of int or dict
            embedding dimmension for embedding_features
            len(embedding_dims) must be equal to len(embedding_features)
        hash_bucket_sizes : list of int or dict
            hash buckets sizes used by hash features
            len(hash_bucket_sizes) must be equal to len(hash_features)

        """
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
        self.columns = None

        if embedding_features is not None and len(embedding_features) != len(embedding_dims):
            raise ValueError("embedding dims must be defined for all embedding features")

        if hash_features is not None and len(hash_features) != len(hash_bucket_sizes):
            raise ValueError("hash bucket size must be defined for all hash features")

        if isinstance(self.embedding_dims, list):
            self.embedding_dims = {k: v for k, v in zip(self.embedding_features, self.embedding_dims)}

        if isinstance(self.hash_bucket_sizes, list):
            self.hash_bucket_sizes = {k: v for k, v in zip(self.hash_features, self.hash_bucket_sizes)}

    def set_columns(self, x_data):
        """set up columns by input data

        Parameters
        ----------
        x_data : pandas.DataFrame
            input data used for training model

        """
        self.columns = []

        for feature in self.numeric_features:
            column = tf.feature_column.numeric_column(feature)
            self.columns.append(column)

        for feature in self.vocab_features:
            if isinstance(x_data[feature].iloc[0], list):
                # multi-hot
                categories = list(sorted(set(itertools.chain.from_iterable(x_data[feature]))))
            else:
                categories = x_data[feature].cat.categories.tolist()
            column = tf.feature_column.categorical_column_with_vocabulary_list(feature, categories)
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
            if isinstance(x_data[feature].iloc[0], list):
                # multi-hot
                categories = list(sorted(set(itertools.chain.from_iterable(x_data[feature]))))
            else:
                categories = x_data[feature].cat.categories.tolist()
            voc_col = tf.feature_column.categorical_column_with_vocabulary_list(
                            feature, categories)
            embed_dim = self.embedding_dims.get(feature, numpy.log2(len(categories)))
            column = tf.feature_column.embedding_column(voc_col, embed_dim)
            self.columns.append(column)

    def get_feature_columns(self):
        """get feature columns using training TF model

        Returns
        -------
        list
            list of feature column

        """
        if self.columns is None:
            raise ValueError("set_columns() must be run before get_feature_columns()")
        return self.columns
