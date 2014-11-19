__author__ = 'Douglas Eric Fonseca Rodrigues'


class TrainingSet:
    def __init__(self):
        self.rows = []
        """:type : list[core.learning.training_set.TrainingSetRow] """

    def append(self, input_pattern, ideal_output=None):
        """
        :type input_pattern: list[float]
        :type ideal_output: list[float]
        """
        self.rows.append(TrainingSetRow(input_pattern, ideal_output))

        return self

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
        :type input_pattern: list[float]
        :type ideal_output: list[float]
        """
        self.input_pattern = input_pattern if input_pattern is not None else []
        self.ideal_output = ideal_output if ideal_output is not None else []