# recommender_model.py

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dot, Add, Flatten
import tensorflow as tf


@tf.keras.utils.register_keras_serializable()
class RecommenderNet(Model):
    def __init__(self, num_users, num_place, embedding_size, **kwargs):
        super(RecommenderNet, self).__init__(**kwargs)
        self.num_users = num_users
        self.num_place = num_place
        self.embedding_size = embedding_size
        
        self.user_embedding = Embedding(num_users, embedding_size,
                                        embeddings_initializer='he_normal',
                                        embeddings_regularizer=tf.keras.regularizers.l2(1e-6))
        self.place_embedding = Embedding(num_place, embedding_size,
                                         embeddings_initializer='he_normal',
                                         embeddings_regularizer=tf.keras.regularizers.l2(1e-6))
        self.user_bias = Embedding(num_users, 1)
        self.place_bias = Embedding(num_place, 1)
    
    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        place_vector = self.place_embedding(inputs[:, 1])
        
        user_bias = self.user_bias(inputs[:, 0])
        place_bias = self.place_bias(inputs[:, 1])
        
        dot_user_place = Dot(axes=1)([user_vector, place_vector])
        x = Add()([dot_user_place, user_bias, place_bias])
        return Flatten()(x)

    def get_config(self):
        config = super(RecommenderNet, self).get_config()
        config.update({
            'num_users': self.num_users,
            'num_place': self.num_place,
            'embedding_size': self.embedding_size,
        })
        return config
