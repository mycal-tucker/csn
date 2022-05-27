# From https://github.com/guillaume-chevalier/Predict-if-salary-is-over-50k-with-Keras/blob/master/Predict%20if%20salary%20is%20%3E50k.ipynb
import numpy as np
from scipy import stats

inputs = (
    ("age", ("continuous",)),
    ("workclass", (
        "Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov", "Without-pay",
        "Never-worked")),
    ("fnlwgt", ("continuous",)),
    ("education", (
        "Bachelors", "Some-college", "11th", "HS-grad", "Prof-school", "Assoc-acdm", "Assoc-voc", "9th", "7th-8th",
        "12th", "Masters", "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool")),
    ("education-num", ("continuous",)),
    ("marital-status", (
        "Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed", "Married-spouse-absent",
        "Married-AF-spouse")),
    ("occupation", ("Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty",
                    "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving",
                    "Priv-house-serv", "Protective-serv", "Armed-Forces")),
    ("relationship", ("Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried")),
    ("race", ("White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black")),
    ("sex", ("Female", "Male")),
    ("capital-gain", ("continuous",)),
    ("capital-loss", ("continuous",)),
    ("hours-per-week", ("continuous",)),
    ("native-country", (
        "United-States", "Cambodia", "England", "Puerto-Rico", "Canada", "Germany", "Outlying-US(Guam-USVI-etc)",
        "India", "Japan", "Greece", "South", "China", "Cuba", "Iran", "Honduras", "Philippines", "Italy", "Poland",
        "Jamaica", "Vietnam", "Mexico", "Portugal", "Ireland", "France", "Dominican-Republic", "Laos", "Ecuador",
        "Taiwan", "Haiti", "Columbia", "Hungary", "Guatemala", "Nicaragua", "Scotland", "Thailand", "Yugoslavia",
        "El-Salvador", "Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands"))
)

age_idx = 0
race_idx = 8
sex_idx = 9
protected_idx = sex_idx

all_fields_shape = []
input_shape = []
for input_idx, input_field in enumerate(inputs):
    count = len(input_field[1])
    all_fields_shape.append(count)
    if input_idx == protected_idx:
        continue
    input_shape.append(count)
all_fields_dim = sum(all_fields_shape)
input_dim = sum(input_shape)
print("input_shape:", input_shape)
print("input_dim:", input_dim)
print("all shapes:", all_fields_shape)
print()


outputs = (0, 1)  # (">50K", "<=50K")
output_dim = 2  # len(outputs)
print("output_dim:", output_dim)
print()


def is_float(string):
    # credits: http://stackoverflow.com/questions/2356925/how-to-check-whether-string-might-be-type-cast-to-float-in-python
    try:
        float(string)
        return True
    except ValueError:
        return False


def find_means_for_continuous_types(X):
    means = []
    for col in range(len(X[0])):
        summ = 0
        count = 0.000000000000000000001
        for value in X[:, col]:
            if is_float(value):
                summ += float(value)
                count += 1
        means.append(summ / count)
    return means


def flatten_persons_inputs_for_model(person_inputs, means):
    float_inputs = []
    for i in range(len(all_fields_shape)):
        if i == protected_idx:
            continue
        features_of_this_type = all_fields_shape[i]
        is_feature_continuous = features_of_this_type == 1
        if is_feature_continuous:
            mean = means[i]
            if is_float(person_inputs[i]):
                scale_factor = 1 / (2 * mean)  # we prefer inputs mainly scaled from -1 to 1.
                float_inputs.append(float(person_inputs[i]) * scale_factor)
            else:
                assert False, "when does it come here?"
                float_inputs.append(mean)
        else:
            for j in range(features_of_this_type):
                feature_name = inputs[i][1][j]

                if feature_name == person_inputs[i]:
                    float_inputs.append(1.)
                else:
                    float_inputs.append(0)
    return float_inputs


def flatten_person_inputs_was_setup(person_inputs, distributions, person_idx):
    float_inputs = []
    float_idx = 0
    for i in range(len(all_fields_shape)):
        if i == protected_idx:
            continue
        features_of_this_type = all_fields_shape[i]
        is_feature_continuous = features_of_this_type == 1
        if is_feature_continuous:
            if is_float(person_inputs[i]):
                percentile = distributions[float_idx][person_idx]
                binned_percentile = int(5 * percentile / 100)
                one_hot_percentile = [0 for _ in range(5)]
                one_hot_percentile[binned_percentile] = 1.0
                float_inputs.extend(one_hot_percentile)
                float_idx += 1
        else:
            for j in range(features_of_this_type):
                feature_name = inputs[i][1][j]

                if feature_name == person_inputs[i]:
                    float_inputs.append(1.)
                else:
                    float_inputs.append(0)
    return float_inputs


def convert_continuous_to_categorical(distributions):
    float_inputs = []
    for i in range(len(all_fields_shape)):
        if i == protected_idx:
            continue
        feature_dist = list(distributions[:, i])
        if not is_float(feature_dist[0]):
            continue
        feature_dist = [float(j) for j in feature_dist]
        features_of_this_type = all_fields_shape[i]
        is_feature_continuous = features_of_this_type == 1
        if is_feature_continuous:
            percentiles = (stats.rankdata(feature_dist, 'min') - 1) / distributions.shape[0]
            float_inputs.append(percentiles)
        else:
            assert False, print("Not continuous?")
            for j in range(features_of_this_type):
                feature_name = inputs[i][1][j]

                if feature_name == person_inputs[i]:
                    float_inputs.append(1.)
                else:
                    float_inputs.append(0)
    return float_inputs


def prepare_data(raw_data, means, wass_setup=False, filter_entries=True):
    X = np.delete(raw_data, -1, axis=1)
    protected_data = raw_data[:, protected_idx: protected_idx + 1]
    y = raw_data[:, -1:]

    new_X = []
    bad_idxs = set()
    if wass_setup:
        percentiles = convert_continuous_to_categorical(X)
    for person in range(len(X)):
        # print("Running person", person, "of", len(X))
        person_data = X[person]
        race = person_data[race_idx]
        if wass_setup and filter_entries and race not in ['White', 'Black']:  # Wass only uses white or black
            bad_idxs.add(person)
            continue
        if wass_setup:  # Wass requires both different protected data (race + sex) and different formatting of inputs.
            formatted_X = flatten_person_inputs_was_setup(person_data, percentiles, person)
        else:
            formatted_X = flatten_persons_inputs_for_model(person_data, means)
        new_X.append(formatted_X)
    new_X = np.array(new_X)

    protected = []
    for p in range(len(protected_data)):
        elt = protected_data[p][0]
        # For sex, use following logic
        if not wass_setup:
            if 'F' in elt:
                protected.append((1, 0))
            else:
                protected.append((0, 1))
            continue
        # Using wass setup, so race and sex.
        # Also get the race idx
        race = raw_data[p][race_idx]
        code = 0
        if race == 'White':
            code += 0
        elif race == 'Black':
            code += 2
        elif filter_entries:
            continue
        if 'F' in elt:
            code += 1
        code_list = [0 for _ in range(4)]
        code_list[code] = 1
        protected.append(tuple(code_list))
        # For race, use following logic:
        # "White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"
        # if 'White' in elt:
        #     protected.append((1, 0, 0, 0, 0))
        # elif "Asian" in elt:
        #     protected.append((0, 1, 0, 0, 0))
        # elif "Amer" in elt:
        #     protected.append((0, 0, 1, 0, 0))
        # elif "Other" in elt:
        #     protected.append((0, 0, 0, 1, 0))
        # else:
        #     protected.append((0, 0, 0, 0, 1))
        # In MMD paper, sensitive value is age
        # For age, use following logic:
        # protected.append(protected_data[p] / protected_range)
        # Age is a continuous variable, but a bunch of the papers seem to treat it as binary.
        # Judging by their "random baseline" ata, it seems like they use the cutoff of 45.
        # Nope, it's age 65: http://proceedings.mlr.press/v28/zemel13.pdf
        # if protected_data[p] < 25:
        #     protected.append((1, 0))
        # else:
        #     protected.append((0, 1))
    new_p = np.array(protected)

    new_y = []
    for i in range(len(y)):
        if i in bad_idxs:
            continue
        elt = y[i][0]
        if '>' in elt:
            new_y.append((1, 0))
        else:  # y[i] == "<=50k":
            new_y.append((0, 1))
    new_y = np.array(new_y)
    return new_X, new_p, new_y


def get_adult_data(train_path, test_path, wass_setup=False):
    training_data = np.genfromtxt(train_path, delimiter=',', dtype=str, autostrip=True)
    print("Training data count:", len(training_data))
    test_data = np.genfromtxt(test_path, delimiter=',', dtype=str, autostrip=True)
    print("Test data count:", len(test_data))

    catted = np.concatenate((training_data, test_data), axis=0)
    means = find_means_for_continuous_types(catted)
    print("Mean values for data types (if continuous):", means)

    X_train, p_train, y_train = prepare_data(training_data, means, wass_setup=wass_setup)
    X_test, p_test, y_test = prepare_data(test_data, means, wass_setup=wass_setup, filter_entries=False)

    percent = sum([i[0] for i in y_train]) / len(y_train)
    print("Training data percentage that is >50k:", percent * 100, "%")
    percent = sum([i[0] for i in y_test]) / len(y_test)
    print("Testing data percentage that is >50k:", percent * 100, "%")
    percent = sum([i[0] for i in p_train]) / len(p_train)
    print("Training data percentage that is protected class 0:", percent * 100, "%")
    percent = sum([i[0] for i in p_test]) / len(p_test)
    print("Testing data percentage that is protected class 0:", percent * 100, "%")
    # print("Training protected value", np.mean(p_train))
    # print("Testing protected value", np.mean(p_test))

    from sklearn.linear_model import LinearRegression
    y_num = np.reshape(np.argmax(y_train, axis=1), (-1, 1))
    p_num = np.reshape(np.argmax(p_train, axis=1), (-1, 1))
    reg = LinearRegression().fit(y_num, p_num)
    print("Score", reg.score(y_num, p_num))
    print("Coeff", reg.coef_)
    return X_train, y_train, p_train, X_test, y_test, p_test
