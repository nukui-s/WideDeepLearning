import itertools

import numpy
import tensorflow as tf


class FeatureSet(object):
    """Set of features"""

    def __init__(self, features):
        """

        Parameters
        ----------
        features : list of Feature
            list of features

        """
        self.features = features

    def set_columns(self, x_data):
        """set up columns by input data

        Parameters
        ----------
        x_data : pandas.DataFrame
            input data used for training model

        """
        self.columns = []

        for feature in self.features:
            column = feature.get_tf_column(x_data)
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


class Feature(object):
    """Base Feature class"""

    def __init__(self, name):
        """

        Parameters
        ----------
        name : str
            name of feature

        """
        self.name = name

    def get_tf_column(self, x_data):
        """get TensorFlow feature column

        Parameters
        ----------
        x_data : pandas.DataFrame
            input data for training

        Returns
        -------
        tensorflow feature column

        """
        raise NotImplementedError


class NumericFeature(Feature):
    """Numeric feature"""

    def get_tf_column(self, x_data):
        """get TensorFlow feature column

        Parameters
        ----------
        x_data : pandas.DataFrame
            input data for training

        Returns
        -------
        tensorflow feature column

        """
        return tf.feature_column.numeric_column(self.name)


class CategoricalFeature(Feature):
    """Categorical feature"""

    def get_tf_column(self, x_data):
        """get TensorFlow feature column

        Parameters
        ----------
        x_data : pandas.DataFrame
            input data for training

        Returns
        -------
        tensorflow feature column

        """
        assert x_data[self.name].dtype.name == 'category'
        categories = x_data[self.name].cat.categories.tolist()
        return tf.feature_column.categorical_column_with_vocabulary_list(self.name, categories)


class IndicatorFeature(Feature):
    """Indicator feature"""

    def get_tf_column(self, x_data):
        """get TensorFlow feature column

        Parameters
        ----------
        x_data : pandas.DataFrame
            input data for training

        Returns
        -------
        tensorflow feature column

        """
        assert x_data[self.name].dtype.name == 'category'
        categories = x_data[self.name].cat.categories.tolist()
        cat_column = tf.feature_column.categorical_column_with_vocabulary_list(self.name, categories)
        return tf.feature_column.indicator_column(cat_column)


class EmbeddingFeature(Feature):
    """Embedding feature"""

    def __init__(self, name, embed_dim=None):
        """

        Parameters
        ----------
        name : str
            name of column
        embed_dim : int
            embedding dimension

        """
        self.name = name
        self.embed_dim = embed_dim

    def get_tf_column(self, x_data):
        """get TensorFlow feature column

        Parameters
        ----------
        x_data : pandas.DataFrame
            input data for training

        Returns
        -------
        tensorflow feature column

        """
        assert x_data[self.name].dtype.name == 'category'
        categories = x_data[self.name].cat.categories.tolist()
        embed_dim = self.embed_dim if self.embed_dim else int(numpy.log2(len(categories)))
        cat_column = tf.feature_column.categorical_column_with_vocabulary_list(self.name, categories)
        embed_column = tf.feature_column.embedding_column(cat_column, embed_dim)
        return embed_column


class HashIndicatorFeature(Feature):
    """Hash feature"""

    def __init__(self, name, hash_bucket_size=100):
        """

        Parameters
        ----------
        name : str
            name of column
        hash_bucket_size : int
            hash bucket size

        """
        self.name = name
        self.hash_bucket_size = hash_bucket_size

    def get_tf_column(self, x_data):
        """get TensorFlow feature column

        Parameters
        ----------
        x_data : pandas.DataFrame
            input data for training

        Returns
        -------
        tensorflow feature column

        """
        dtype = tf.string if isinstance(x_data[self.name].iloc[0], str) else tf.int64
        hash_col = tf.feature_column.categorical_column_with_hash_bucket(self.name, self.hash_bucket_size, dtype=dtype)
        return tf.feature_column.indicator_column(hash_col)


class HashEmbeddingFeature(Feature):
    """Embedding feature for hashing"""

    def __init__(self, name, embed_dim=None, hash_bucket_size=100):
        """

        Parameters
        ----------
        name : str
            name of column
        embed_dim : int
            embedding dimension
        hash_bucket_size : int
            hash bucket size

        """
        self.name = name
        self.embed_dim = embed_dim
        self.hash_bucket_size = hash_bucket_size

    def get_tf_column(self, x_data):
        """get TensorFlow feature column

        Parameters
        ----------
        x_data : pandas.DataFrame
            input data for training

        Returns
        -------
        tensorflow feature column

        """
        dtype = tf.string if isinstance(x_data[self.name].iloc[0], str) else tf.int64
        hash_column = tf.feature_column.categorical_column_with_hash_bucket(self.name, self.hash_bucket_size, dtype=dtype)
        unique_size = len(set(x_data[self.name]))
        embed_dim = self.embed_dim if self.embed_dim else int(numpy.log2(unique_size))
        embed_column = tf.feature_column.embedding_column(hash_column, embed_dim)
        return embed_column
