from collections import Counter, defaultdict
from typing import Optional
from lidstone import UnigramLidstoneModel
from math import log, exp
import matplotlib.pyplot as plt
import pandas as pd
import re

EPSILON: float = 1e-6
LIDSTONE_CONSTANT: float = 0.12
K: int = 10
THRESHOLD: float = 10.0
SECTION_NUM_LINES: int = 4


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


def to_png_file_name(y_label, plot_type):
    file_name = y_label.lower().replace(" ", "_")
    return f"{file_name}_{plot_type}.png"

def save_plot(values, y_label):
    plt.figure(figsize=(10, 5))
    plt.plot(values)
    plt.xlabel('Iterations')
    plt.ylabel(y_label)
    plt.title(f"{y_label} vs Iteration")
    file_name = to_png_file_name(y_label, "plot")
    plt.savefig(file_name)
    print(f"{y_label} plot image: {file_name}")
    plt.close()

def save_histograms(topics, matrix_data, clusters_id_sorted, predictions):
    fig, axs = plt.subplots(5, 2, figsize=(20, 20))
    for i, ax in enumerate(axs.flat):
        if i >= 9:
            ax.axis('off')
            continue
        cid = clusters_id_sorted[i]
        values = matrix_data[i][:-1]
        ax.bar(topics, values)
        ax.set_xticks(topics)  # redundant but deprecates warnings
        ax.set_xticklabels(topics, fontweight='bold', fontsize='large')

        ax.set_title(f'Cluster {cid + 1} ({predictions[cid]})', fontweight='bold', fontsize='xx-large')

    plt.tight_layout()
    image_name = to_png_file_name("clusters", "histogram")
    fig.savefig(image_name, dpi=300)  # higher resolution
    print(f"Saved histogram image: {image_name}")
    plt.close(fig)

def main():
    # parsing file

    article_to_words: dict[int, list[str]] = dict()
    article_to_topics: dict[int, list[str]] = dict()
    all_words: list[str] = []

    header_re = re.compile(r"<TRAIN\s+\d+\s+((\S+\s*)+)>\s*")
    with open('develop.txt', 'r') as file:
        for i, line in enumerate(file):
            article = i // SECTION_NUM_LINES
            mod = i % SECTION_NUM_LINES
            if mod == 0:
                # article header line
                match: re.Match = header_re.match(line)
                if not match:
                    print("line #{} expected to be a header but not follow header format")
                    return 1
                g1: str = match.group(1)
                topics: list[str] = g1.split()
                article_to_topics[article]= topics
            elif mod == 2:
                # article content line
                words: list[str] = line.split(' ')[:-1]

                article_to_words[article] = words
                all_words += words

    # filter rare words

    words_count = Counter(all_words)

    all_words = list(filter(lambda x: words_count[x] > 3, all_words))
    vocabulary_s: int = len(set(all_words))
    set_all_words = set(all_words)
    print(f"voabulary size: {len(set_all_words)}")

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

        print(f'{iteration}) (Log)-Likelihood: {all_likelihoods[-1]:.3f}, Perplexity: {all_perplexities[-1]:.3f}')

        if all_likelihoods[-1] - last_likelihood < THRESHOLD:
            break
        
        last_likelihood = all_likelihoods[-1]

    # create plots

    save_plot(all_likelihoods, "Log Likelihood")
    save_plot(all_perplexities, "Perplexity")

    # parse topics file

    topics = []
    with open("topics.txt", "r") as fp:
        for line in fp:
            line = line.strip()
            if line:
                topics.append(line)
    topic_to_column: dict[str, int] = {topic: idx for idx, topic in enumerate(topics)}

    # set confusion matrix

    matrix_data = []
    n_rows = len(em.clusters)
    n_cols = len(em.clusters) + 1  # +1 for cluster size 
    for i in range(n_rows):
        matrix_data.append([0] * n_cols)

    clusters_id_sorted: list[int] = sorted(range(len(em.clusters)), key=lambda x: len(em.clusters[x]), reverse=True)
    cluster_id_to_row: dict[int, int] = {cid: idx for idx, cid in enumerate(clusters_id_sorted)}
    article_to_cluster: dict[int, int] = {}
    for cid, cluster in enumerate(em.clusters):
        for article in cluster:
            article_to_cluster[article] = cid
    for article, atopics in article_to_topics.items():
        cid = article_to_cluster[article]
        row = cluster_id_to_row[cid]
        for topic in atopics:
            col = topic_to_column[topic]
            matrix_data[row][col] += 1

    # set predictions based on most common topic

    predictions: dict[int, str] = {}
    for i, cid in enumerate(clusters_id_sorted):
        counts = matrix_data[i][:-1]
        max_pos = counts.index(max(counts))
        predictions[cid] = topics[max_pos]

    # plot histograms based on confusion matrix

    save_histograms(topics, matrix_data, clusters_id_sorted, predictions)

    # add cluster size to matrix
    for cid, row in cluster_id_to_row.items():
        matrix_data[row][-1] = len(em.clusters[cid])

    # log confusion matrix

    rows: list[int] = [x + 1 for x in clusters_id_sorted]
    columns = [*topics, "cluster size"]
    matrix_df = pd.DataFrame(matrix_data, index=rows, columns=columns)
    print(matrix_df)

    # calculate model's accuracy
    correct_assignments: int = 0
    total_assignments: int = 0
    for cid, cluster in enumerate(em.clusters):
        for article in cluster:
            real_topics = article_to_topics[article]
            if predictions[cid] in real_topics:
                correct_assignments += 1
            total_assignments += 1
    if total_assignments:
        print(f"Model's accuracy: {correct_assignments/total_assignments:.3f}")

if __name__ == '__main__':
    import sys
    sys.exit(main())
