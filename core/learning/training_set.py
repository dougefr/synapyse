__author__ = 'Douglas'


class TrainingSet:
    def __init__(self):
        self.rows = []
        """:type : list[TrainingSetRow] """

    def append(self, input_pattern, ideal_output):
        """
        :type input_pattern: list[double]
        :type ideal_output: list[double]
        """

        self.rows.append(TrainingSetRow(input_pattern, ideal_output))

    def __getitem__(self, item):
        return self.rows[item]

    def __len__(self):
        return len(self.rows)


class TrainingSetRow:
    def __init__(self, input_pattern=None, ideal_output=None):
        self.pattern = input_pattern
        """:type : list[bool]"""
        self.ideal_output = ideal_output
        """:type : list[bool]"""
