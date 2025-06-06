import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs

# 1. Carregar e pré-processar os dados
ratings = tfds.load("movielens/100k-ratings", split="train")
movies = tfds.load("movielens/100k-movies", split="train")

# Garantir que os tensores sejam escalares (shape=[]), para evitar erros de batching
ratings = ratings.map(
    lambda x: {
        "movie_title": tf.reshape(x["movie_title"], []),
        "user_id": tf.reshape(x["user_id"], []),
    }
)

movies = movies.map(lambda x: {"movie_title": tf.reshape(x["movie_title"], [])})

# 2. Preparar vocabulários
movie_titles = movies.map(lambda x: x["movie_title"])
user_ids = ratings.map(lambda x: x["user_id"])

unique_movie_titles = list(set(title.numpy().decode("utf-8") for title in movie_titles))
unique_user_ids = list(set(uid.numpy().decode("utf-8") for uid in user_ids))


# 3. Modelos de usuário e filme
class MovieModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.embedding = tf.keras.Sequential(
            [
                tf.keras.layers.StringLookup(
                    vocabulary=unique_movie_titles, mask_token=None
                ),
                tf.keras.layers.Embedding(len(unique_movie_titles) + 1, 32),
            ]
        )

    def call(self, titles):
        return self.embedding(titles)


class UserModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.embedding = tf.keras.Sequential(
            [
                tf.keras.layers.StringLookup(
                    vocabulary=unique_user_ids, mask_token=None
                ),
                tf.keras.layers.Embedding(len(unique_user_ids) + 1, 32),
            ]
        )

    def call(self, user_ids):
        return self.embedding(user_ids)


# 4. Modelo de recomendação
class MovieRetrievalModel(tfrs.models.Model):
    def __init__(self, user_model, movie_model):
        super().__init__()
        self.movie_model = movie_model
        self.user_model = user_model
        self.task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=movies.batch(128).map(
                    lambda x: (x["movie_title"], movie_model(x["movie_title"]))
                )
            )
        )

    def compute_loss(self, features, training=False):
        user_embeddings = self.user_model(features["user_id"])
        movie_embeddings = self.movie_model(features["movie_title"])
        return self.task(user_embeddings, movie_embeddings)


# 5. Treinamento
cached_ratings = ratings.shuffle(100_000).batch(8192).cache()

model = MovieRetrievalModel(UserModel(), MovieModel())
model.compile(optimizer=tf.keras.optimizers.Adagrad(0.5))
model.fit(cached_ratings, epochs=3)

# 6. Recomendações
index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)
index.index_from_dataset(
    movies.batch(100).map(
        lambda x: (x["movie_title"], model.movie_model(x["movie_title"]))
    )
)

# Exibe recomendações para um usuário específico
user_id = "42"
scores, titles = index(tf.constant([user_id]))
print(f"\n🎬 Filmes recomendados para o usuário {user_id}:\n")
for title in titles[0, :5].numpy():
    print(f" - {title.decode('utf-8')}")

# Verificação de versão
# print(f"\n✅ TensorFlow: {tf.__version__}")
# print(f"✅ TFRS: {tfrs.__version__}")
