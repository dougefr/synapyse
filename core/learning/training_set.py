__author__ = 'Douglas Eric Fonseca Rodrigues'


class TrainingSet:
    def __init__(self):
        self.rows = []
        """:type : list[core.learning.training_set.TrainingSetRow] """

    def append(self, input_pattern, ideal_output):
        """
        :type input_pattern: list[double]
        :type ideal_output: list[double]
        """
        self.rows.append(TrainingSetRow(input_pattern, ideal_output))

    def __getitem__(self, item):
        """
        :type item: int
        """
        return self.rows[item]

    def __len__(self):
        return len(self.rows)


class TrainingSetRow:
    def __init__(self, input_pattern=None, ideal_output=None):
        """
        :type input_pattern: list[double]
        :type ideal_output: list[double]
        """
        self.pattern = input_pattern
        self.ideal_output = ideal_output