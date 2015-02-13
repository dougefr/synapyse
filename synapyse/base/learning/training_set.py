__author__ = 'Douglas Eric Fonseca Rodrigues'


class TrainingSet:
    def __init__(self, input_count, output_count=None):
        """
        :type input_count: int
        :type output_count: int
        """

        self.rows = []
        """:type : list[core.learning.training_set.TrainingSetRow] """

        self.input_count = input_count
        self.output_count = output_count

    def append(self, input_pattern, ideal_output=None):
        """
        :type input_pattern: list[float]
        :type ideal_output: list[float]
        """
        if self.input_count != len(input_pattern):
            raise ValueError
        elif ideal_output is None and self.output_count is not None:
            raise ValueError
        elif ideal_output is not None and self.output_count is None:
            raise ValueError
        elif ideal_output is not None and self.output_count != len(ideal_output):
            raise ValueError
        else:
            self.rows.append(TrainingSetRow(input_pattern, ideal_output))

        return self

    def __getitem__(self, item):
        """
        :type item: int
        """
        return self.rows[item]

    def __len__(self):
        return len(self.rows)

    def import_from_file(self, file_name, separator):
        """
        :type file_name: str
        :type separator: str
        """
        file = open(file_name)

        for line in file:
            line_split = line.split(separator)

            training_set_row = TrainingSetRow()

            for i in range(self.input_count):
                training_set_row.input_pattern.append(float(line_split[i]))

            for i in range(self.output_count):
                training_set_row.ideal_output.append(float(line_split[self.input_count + i]))

            self.rows.append(training_set_row)

        file.close()

        return self

    def normalize(self):
        """
        Sets the values of training set to values in the range from 0 to 1 using the follow formula:
        Xn = (X - Xmin)/(Xmax - Xmin)

        Where:

        X – value that should be normalized
        Xn – normalized value
        Xmin – minimum value of X
        Xmax – maximum value of X

        Reference: http://neuroph.sourceforge.net/tutorials/MusicClassification/
        music_classification_by_genre_using_neural_networks.html
        """

        # Find the Xmin and Xmax
        x_min = dict(input_pattern=[None] * self.input_count, ideal_output=[None] * self.output_count)
        x_max = dict(input_pattern=[None] * self.input_count, ideal_output=[None] * self.output_count)

        for training_set_row in self:

            for i in range(len(training_set_row.input_pattern)):
                if x_min['input_pattern'][i] is None or x_min['input_pattern'][i] > training_set_row.input_pattern[i]:
                    x_min['input_pattern'][i] = training_set_row.input_pattern[i]
                if x_max['input_pattern'][i] is None or x_max['input_pattern'][i] < training_set_row.input_pattern[i]:
                    x_max['input_pattern'][i] = training_set_row.input_pattern[i]

            for i in range(len(training_set_row.ideal_output)):
                if x_min['ideal_output'][i] is None or x_min['ideal_output'][i] > training_set_row.ideal_output[i]:
                    x_min['ideal_output'][i] = training_set_row.ideal_output[i]
                if x_max['ideal_output'][i] is None or x_max['ideal_output'][i] < training_set_row.ideal_output[i]:
                    x_max['ideal_output'][i] = training_set_row.ideal_output[i]

        # Normalizes
        for training_set_row in self:

            for i in range(len(training_set_row.input_pattern)):
                if x_max['input_pattern'][i] == x_min['input_pattern'][i]:
                    training_set_row.input_pattern[i] = 0
                else:
                    training_set_row.input_pattern[i] = \
                        (training_set_row.input_pattern[i] - x_min['input_pattern'][i]) / \
                        (x_max['input_pattern'][i] - x_min['input_pattern'][i])

            for i in range(len(training_set_row.ideal_output)):
                if x_max['ideal_output'][i] == x_min['ideal_output'][i]:
                    training_set_row.ideal_output[i] = 0
                else:
                    training_set_row.ideal_output[i] = \
                        (training_set_row.ideal_output[i] - x_min['ideal_output'][i]) / \
                        (x_max['ideal_output'][i] - x_min['ideal_output'][i])

        return self


class TrainingSetRow:
    def __init__(self, input_pattern=None, ideal_output=None):
        """
        :type input_pattern: list[float]
        :type ideal_output: list[float]
        """
        self.input_pattern = input_pattern if input_pattern is not None else []
        self.ideal_output = ideal_output if ideal_output is not None else []