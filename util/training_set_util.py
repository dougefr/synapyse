from core.learning.training_set import TrainingSet, TrainingSetRow

__author__ = 'Douglas Eric Fonseca Rodrigues'


def import_from_file(file_name, input_count, output_count, separator):
    """
    :type file_name: str
    :type input_count: int
    :type output_count: int
    :type separator: str
    """
    training_set = TrainingSet()

    file = open(file_name, "r")

    for line in file:
        line_split = line.split(separator)

        training_set_row = TrainingSetRow()

        for i in range(input_count):
            training_set_row.input_pattern.append(float(line_split[i]))

        for i in range(output_count):
            training_set_row.ideal_output.append(float(line_split[input_count + i]))

        training_set.rows.append(training_set_row)

    file.close()

    return training_set