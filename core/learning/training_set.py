__author__ = "Douglas Eric Fonseca Rodrigues"


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

    def import_from_file(self, file_name, input_count, output_count, separator):
        """
        :type file_name: str
        :type input_count: int
        :type output_count: int
        :type separator: str
        """
        file = open(file_name)

        for line in file:
            line_split = line.split(separator)

            training_set_row = TrainingSetRow()

            for i in range(input_count):
                training_set_row.input_pattern.append(float(line_split[i]))

            for i in range(output_count):
                training_set_row.ideal_output.append(float(line_split[input_count + i]))

            self.rows.append(training_set_row)

        file.close()

        return self


class TrainingSetRow:
    def __init__(self, input_pattern=None, ideal_output=None):
        """
        :type input_pattern: list[float]
        :type ideal_output: list[float]
        """
        self.input_pattern = input_pattern if input_pattern is not None else []
        self.ideal_output = ideal_output if ideal_output is not None else []