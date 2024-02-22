# Name: Reuven Chiche, ID: 328944517
import math
from collections import Counter


class UnigramLidstoneModel:
    """
    This class represents a unigram lidstone model.
    """

    def __init__(self, constant: float, events: list[str], vocabulary: int) -> None:
        """
        Initialize an unigram lidstone model.
        :param constant: The constant of the model.
        :param events: The events that the model will be trained on.
        :param vocabulary: The vocabulary size.
        """

        self.constant: float = constant
        self.events: Counter[str] = Counter(events)
        self.len_of_events: int = len(events)
        self.vocabulary: int = vocabulary

        # Caching the denominator of the lidstone probability for faster calculation.
        self.lidstone_denominator: float = float(self.len_of_events + float(self.constant * self.vocabulary))

    def lidstone(self, input_word: str) -> float:
        """
        Calculate the lidstone probability of a given word.
        :param input_word: The word to calculate the probability of.
        :return: The lidstone probability of the given word.
        """

        return float(self.events[input_word] + self.constant) / self.lidstone_denominator

    def perplexity(self, dataset: list[str]) -> float:
        """
        Calculate the perplexity measure of the model on a given dataset.
        :param dataset: The dataset to calculate the perplexity on.
        :return: The perplexity measure of the model on the given dataset.
        """

        # Dict to store the log of the lidstone probability of each event for caching.
        dic: dict[str, float] = dict()
        sum_log: float = 0.0

        for event in dataset:
            if event in dic:
                sum_log += dic[event]
            else:
                dic[event] = math.log(self.lidstone(event), 2)
                sum_log += dic[event]

        return 2 ** ((-1 / len(dataset)) * sum_log)

    def set_constant(self, const: float) -> None:
        """
        Set the constant - λ of the model.
        :param const: The constant to set.
        :return: None.
        """

        self.constant = const
        self.lidstone_denominator = float(self.len_of_events + float(self.constant * self.vocabulary))

    def debug(self) -> None:
        """
        Calculate the sum of each event probability.
        :return: None.
        """

        sum_of_model: float = 0.0

        for event in self.events.keys():
            sum_of_model += self.lidstone(event)

        sum_of_model += (self.vocabulary - len(set(self.events))) * self.lidstone("unseen-word")

        print(f'@Debug - Constant of the model: {self.constant} probability sum: {sum_of_model}')


def find_best_constant(events_of_training: list[str], events_of_validation: list[str], vocabulary: int) -> float:
    """
    Find the best constant - λ for the model.
    :param events_of_training: Training dataset.
    :param events_of_validation: Validation dataset.
    :param vocabulary: The vocabulary size.
    :return: The best constant for the model.
    """

    min_perplexity: float = float('inf')
    best_constant: float = 0.0

    model: UnigramLidstoneModel = UnigramLidstoneModel(0.01, events_of_training, vocabulary)

    while model.constant <= 2.00:
        perplexity: float = model.perplexity(events_of_validation)
        if perplexity < min_perplexity:
            min_perplexity = perplexity
            best_constant = model.constant

        model.set_constant(model.constant + 0.01)

    return float('{:.2f}'.format(best_constant))
