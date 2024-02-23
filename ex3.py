from collections import Counter, defaultdict
from typing import Optional
from lidstone import UnigramLidstoneModel
from math import log, exp
import matplotlib.pyplot as plt
import pandas as pd

EPSILON: float = 1e-6
LIDSTONE_CONSTANT: float = 0.1
K: int = 10
THRESHOLD: float = 10.0


class EM:
    def __init__(self, articles: dict[int, list[str]],
                 num_of_clusters: int,
                 num_of_words: int,
                 vocabulary: int,
                 epsilon: float = EPSILON,
                 k: int = K,
                 lidstone_constant: float = LIDSTONE_CONSTANT) -> None:

        self.articles: dict[int, list[str]] = articles
        self.num_of_articles: int = len(articles)
        self.num_of_clusters: int = num_of_clusters
        self.clusters: list[list[int]] = [[] for _ in range(self.num_of_clusters)]
        self.alpha: dict[int, float] = defaultdict(float)
        self.epsilon: float = epsilon
        self.k: int = k
        self.lidstone_constant: float = lidstone_constant
        self.vocabulary: int = vocabulary
        self.num_of_words: int = num_of_words

        self.p: list[Optional[UnigramLidstoneModel]] = [None] * self.num_of_clusters
        self.w_t: list[list[float]] = [[] for _ in range(self.num_of_clusters)]
        self.m: list[float] = [0.0] * self.num_of_articles
        self.exp_z: list[list[float]] = [[] for _ in range(self.num_of_articles)]
        self.likelihood: float = 0.0

    def initialization(self) -> None:
        for article_ord in self.articles.keys():
            self.clusters[article_ord % self.num_of_clusters].append(article_ord)

        for article in range(self.num_of_articles):
            for cluster in range(self.num_of_clusters):
                if article in self.clusters[cluster]:
                    self.w_t[cluster].append(1.0)
                else:
                    self.w_t[cluster].append(0.0)

        self.compute_alpha()
        self.normalize_alpha()
        self.compute_p_ik()

    def compute_alpha(self) -> None:
        for i in range(self.num_of_clusters):
            self.alpha[i] = sum(self.w_t[i]) / self.num_of_articles

            if self.alpha[i] < self.epsilon:
                self.alpha[i] = self.epsilon

    def normalize_alpha(self) -> None:
        sum_alpha: int = sum(self.alpha.values())

        for i in range(self.num_of_clusters):
            self.alpha[i] /= sum_alpha

    def compute_p_ik(self) -> None:
        list_of_words: list[str] = list()

        for i, cluster in enumerate(self.clusters):
            for article in cluster:
                list_of_words += self.articles[article]

            self.p[i] = UnigramLidstoneModel(self.lidstone_constant, list_of_words, self.vocabulary)
            list_of_words.clear()

    def e_step(self) -> None:
        z: list[float] = [0.0] * self.num_of_clusters

        new_clusters: list[list[int]] = [[] for _ in range(self.num_of_clusters)]

        all_w_t: list[list[float]] = []
        all_m: list[float] = []
        all_exp_z: list[list[float]] = []

        for article in range(self.num_of_articles):

            for cluster in range(self.num_of_clusters):
                temp: float = 0.0

                for word, count in Counter(self.articles[article]).items():
                    temp += log(self.p[cluster].lidstone(word)) * count

                z[cluster] = log(self.alpha[cluster]) + temp

            m: float = max(z)
            exp_z: list[float] = [exp(z[i] - m) for i in range(self.num_of_clusters)]
            w_t: list[float] = [0.0] * self.num_of_clusters

            for i in range(self.num_of_clusters):
                if z[i] - m < -self.k:
                    w_t[i] = 0.0
                else:
                    w_t[i] = exp_z[i] / sum([exp_z[j] for j in  range(self.num_of_clusters) if z[j] - m >= -self.k])

            new_clusters[w_t.index(max(w_t))].append(article)
            all_w_t.append(w_t)
            all_m.append(m)
            all_exp_z.append(exp_z)

        self.clusters = new_clusters
        self.w_t = all_w_t
        self.m = all_m
        self.exp_z = all_exp_z

    def m_step(self) -> None:
        self.compute_alpha()
        self.normalize_alpha()
        self.compute_p_ik()

    def likelihood_calc(self) -> None:
        self.likelihood: float = 0.0
        for i in range(self.num_of_articles):
            self.likelihood += self.m[i] + log(sum([e for e in self.exp_z[i] if e - self.m[i] >= -self.k]))

    def perplexity(self) -> float:
        return exp((-self.likelihood / self.num_of_words))


def main() -> None:
    # parsing file

    article_to_words: dict[int, list[str]] = dict()
    all_words: list[str] = []

    with open('develop.txt', 'r') as file:
        next(file)
        next(file)

        for i, article in enumerate(file):
            if i % 4 != 0:
                continue

            words: list[str] = article.split(' ')[:-1]

            article_to_words[i // 4] = words
            all_words += words

    # filter rare words

    words_count = Counter(all_words)

    all_words = list(filter(lambda x: words_count[x] > 3, all_words))
    vocabulary_s: int = len(set(all_words))
    set_all_words = set(all_words)

    for i in range(len(article_to_words)):
        article_to_words[i] = list(filter(lambda x: x in set_all_words, article_to_words[i]))

    # EM clustering

    em = EM(article_to_words, 9, len(all_words), vocabulary_s)
    em.initialization()

    last_likelihood: float = float('-inf')
    all_likelihoods: list[float] = []
    all_perplexities: list[float] = []
    iteration: int = 0

    while True:
        em.e_step()
        em.m_step()
        em.likelihood_calc()

        iteration += 1

        all_likelihoods.append(em.likelihood)
        all_perplexities.append(em.perplexity())

        print(f'Log-Likelihood: {all_likelihoods[-1]}')

        if all_likelihoods[-1] - last_likelihood < THRESHOLD:
            break

        last_likelihood = all_likelihoods[-1]

    # create plots

    # log-likelihood plot
    plt.plot(range(1, iteration + 1), all_likelihoods)
    plt.xlabel('Iteration')
    plt.ylabel('Log-Likelihood')
    plt.title('Log-Likelihood vs. Iteration')
    plt.show()

    # perplexity plot
    plt.plot(range(1, iteration + 1), all_perplexities)
    plt.xlabel('Iteration')
    plt.ylabel('Perplexity')
    plt.title('Perplexity vs. Iteration')
    plt.show()

    # confusion matrix
    # TODO

    topics: list[str] = []

    with open('topics.txt', 'r') as file:
        for topic in file:
            topics.append(topic.strip())

    clusters_id_sorted: list[int] = sorted(range(len(em.clusters)), key=lambda x: len(em.clusters[x]), reverse=True)
    rows: list[int] = [x + 1 for x in clusters_id_sorted]

    matrix_data: list[list[int]] = [[0] * len(topics) for _ in range(len(em.clusters))]

    for i in range(len(matrix_data)):
        for j in range(len(matrix_data[i])):
            # matrix_data[i][j] =
            ...

    matrix_df = pd.DataFrame(matrix_data, index=rows, columns=topics)


if __name__ == '__main__':
    main()
