from matplotlib.pyplot import subplots, show
from pandas import cut
from seaborn import factorplot

from read_write import read_train_data


def print_value_examples_for_each_column(data):
    # For each column, print the possible values
    for column in data:
        values = data[column].unique()
        try:
            values = sorted(values)
        except TypeError:
            """ Not all columns can be sorted; leave the unsortable ones as-is """
        if len(values) > 10:
            examples = "{} ... ({})".format(
                ", ".join(str(v) for v in values[:10]),
                len(values)
            )
        else:
            examples = ", ".join(str(v) for v in values)
        print(f"{column}: {examples}")
    print()


AGE_BINS_START = (0, 3, 8, 12, 18, 25, 32, 40, 50, 60, 70, 120)
AGE_BIN_LABELS = tuple("{}-{}".format(AGE_BINS_START[i], AGE_BINS_START[1 + 1] - 1) for i in range(len(AGE_BINS_START) - 1))


def create_age_bins(data):
    return cut(data["Age"], bins=AGE_BINS_START)


def plot_survival_per_bin(data, bin_column, normalize=True):
    groups = data.groupby(bin_column).agg(
        total=("Survived", "count"),
        survived=("Survived", "sum"),
    )
    if normalize:
        survived = groups["survived"] / groups["total"]
        died = (groups["total"] - groups["survived"]) / groups["total"]
    else:
        survived = groups["survived"]
        died = groups["total"] - groups["survived"]
    fig, ax = subplots(tight_layout=True)
    ax.bar(AGE_BIN_LABELS, died.tolist(), label="Died")
    ax.bar(AGE_BIN_LABELS, survived.tolist(), bottom=died.tolist(), label="Survived")
    ax.legend()

def main():
    """
    This functions creates images to compare the effect of different input factors.
    """

    train_data = read_train_data()
    print_value_examples_for_each_column(train_data)

    # Add a new column that is the age group
    train_data["Age_group"] = create_age_bins(train_data)
    # Plot the survival for each age group
    plot_survival_per_bin(train_data, bin_column="Age_group")

    # Plot the effect of class per gender
    factorplot("Pclass", "Survived", "Sex", data=train_data, kind="bar")


if __name__ == '__main__':
    main()
    show()

