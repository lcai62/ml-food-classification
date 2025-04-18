import re
from typing import Tuple

import numpy as np
import pandas as pd
import random
from collections import Counter

from matplotlib import pyplot as plt

RUNS = 100


STOPWORDS = {"the", "of"}


def label_encode_food(x: str) -> int:
    """
    mapping
    pizza: 0
    shawarma: 1
    sushi: 2
    """
    if isinstance(x, str):
        x_lower = x.lower()
        if "pizza" in x_lower:
            return 0
        elif "shawarma" in x_lower:
            return 1
        elif "sushi" in x_lower:
            return 2
    return -2  # this should never happen


# giant ass numbers map
written_numbers_map = {
    "zero": 0, "one": 1, "first": 1, "two": 2, "second": 2, "three": 3, "third": 3,
    "four": 4, "fourth": 4, "five": 5, "fifth": 5, "six": 6, "sixth": 6, "seven": 7, "eighth": 8,
    "eight": 8, "nine": 9, "ninth": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "twenty-one": 21, "twenty one": 21, "twenty-two": 22, "twenty two": 22,
    "twenty-three": 23, "twenty three": 23, "twenty-four": 24, "twenty four": 24,
    "twenty-five": 25, "twenty five": 25, "twenty-six": 26, "twenty six": 26,
    "twenty-seven": 27, "twenty seven": 27, "twenty-eight": 28, "twenty eight": 28,
    "twenty-nine": 29, "twenty nine": 29,
    "thirty": 30, "thirty-one": 31, "thirty one": 31, "thirty-two": 32, "thirty two": 32,
    "thirty-three": 33, "thirty three": 33, "thirty-four": 34, "thirty four": 34,
    "thirty-five": 35, "thirty five": 35, "thirty-six": 36, "thirty six": 36,
    "thirty-seven": 37, "thirty seven": 37, "thirty-eight": 38, "thirty eight": 38,
    "thirty-nine": 39, "thirty nine": 39,
    "forty": 40, "forty-one": 41, "forty one": 41, "forty-two": 42, "forty two": 42,
    "forty-three": 43, "forty three": 43, "forty-four": 44, "forty four": 44,
    "forty-five": 45, "forty five": 45, "forty-six": 46, "forty six": 46,
    "forty-seven": 47, "forty seven": 47, "forty-eight": 48, "forty eight": 48,
    "forty-nine": 49, "forty nine": 49,
    "fifty": 50, "fifty-one": 51, "fifty one": 51, "fifty-two": 52, "fifty two": 52,
    "fifty-three": 53, "fifty three": 53, "fifty-four": 54, "fifty four": 54,
    "fifty-five": 55, "fifty five": 55, "fifty-six": 56, "fifty six": 56,
    "fifty-seven": 57, "fifty seven": 57, "fifty-eight": 58, "fifty eight": 58,
    "fifty-nine": 59, "fifty nine": 59,
    "sixty": 60, "sixty-one": 61, "sixty one": 61, "sixty-two": 62, "sixty two": 62,
    "sixty-three": 63, "sixty three": 63, "sixty-four": 64, "sixty four": 64,
    "sixty-five": 65, "sixty five": 65, "sixty-six": 66, "sixty six": 66,
    "sixty-seven": 67, "sixty seven": 67, "sixty-eight": 68, "sixty eight": 68,
    "sixty-nine": 69, "sixty nine": 69,
    "seventy": 70, "seventy-one": 71, "seventy one": 71, "seventy-two": 72, "seventy two": 72,
    "seventy-three": 73, "seventy three": 73, "seventy-four": 74, "seventy four": 74,
    "seventy-five": 75, "seventy five": 75, "seventy-six": 76, "seventy six": 76,
    "seventy-seven": 77, "seventy seven": 77, "seventy-eight": 78, "seventy eight": 78,
    "seventy-nine": 79, "seventy nine": 79,
    "eighty": 80, "eighty-one": 81, "eighty one": 81, "eighty-two": 82, "eighty two": 82,
    "eighty-three": 83, "eighty three": 83, "eighty-four": 84, "eighty four": 84,
    "eighty-five": 85, "eighty five": 85, "eighty-six": 86, "eighty six": 86,
    "eighty-seven": 87, "eighty seven": 87, "eighty-eight": 88, "eighty eight": 88,
    "eighty-nine": 89, "eighty nine": 89,
    "ninety": 90, "ninety-one": 91, "ninety one": 91, "ninety-two": 92, "ninety two": 92,
    "ninety-three": 93, "ninety three": 93, "ninety-four": 94, "ninety four": 94,
    "ninety-five": 95, "ninety five": 95, "ninety-six": 96, "ninety six": 96,
    "ninety-seven": 97, "ninety seven": 97, "ninety-eight": 98, "ninety eight": 98,
    "ninety-nine": 99, "ninety nine": 99,
    "one hundred": 100, "hundred": 100
}


def parse_ingredient_count(text: str) -> float:
    """
    Takes in text and tries to extract the ingredient count
    """
    text_lower = str(text).lower().strip()

    # keep track of all numbers found
    found_nums = []

    # find all digit numbers with regex
    digit_matches = re.findall(r"\d+", text_lower)
    for dm in digit_matches:
        try:
            found_nums.append(float(dm))  # store as float
        except ValueError:
            pass  # shouldn't go here but just incase

    # search for written numbers
    for word, val in written_numbers_map.items():
        if re.search(rf"\b{re.escape(word)}\b", text_lower):
            found_nums.append(float(val))

    # take average of all numbers found (this works very accurately 95% of the time
    # since most people report 1 number or a range)
    result = np.nan
    if len(found_nums) >= 1:
        result = sum(found_nums) / len(found_nums)
    else:
        # no digit or textual numbers found

        # quercus uses * for bulleted lists, if we see this, assume that
        # the user make a bulleted list
        if "*" in text or "\n" in text_lower:
            lines = [line.strip() for line in text_lower.split("\n") if line.strip()]
            bullet_lines = [ln for ln in lines if ln.startswith("*") or ln.startswith("-")]
            if bullet_lines:
                result = len(bullet_lines)
            else:
                result = len(lines)  # fallback

        # the user made a comma separated list
        elif "," in text_lower:
            ingredients = [i.strip() for i in text_lower.split(",") if i.strip()]
            if len(ingredients) > 1:
                result = len(ingredients)

    # print(f"'{text}' => {result}")
    return result


def parse_price(text: str) -> float:
    """
    Takes in text and tries to extract the price
    """
    text_lower = str(text).lower().strip()

    # replace written numbers with digits
    for word, val in written_numbers_map.items():
        text_lower = re.sub(rf"\b{word}\b", str(val), text_lower)

    # extract numbers and take avg
    nums = re.findall(r"(\d+\.?\d*)", text_lower)
    nums = list(map(float, nums))

    if len(nums) == 0:
        return np.nan  # no numbers (5 rows got here)

    # print(f"'{text}' => {sum(nums) / len(nums)}")

    return sum(nums) / len(nums)


def parse_hot_sauce(text):
    """
    options:
    "None"
    "I will have some of this food item with my hot sauce"
    "A little (mild)"
    "A moderate amount (medium)"
    "A lot (hot)"
    """

    text_lower = str(text).lower()

    if "none" in text_lower:
        return 0
    elif "mild" in text_lower:
        return 1
    elif "medium" in text_lower:
        return 2
    elif "lot" in text_lower:
        return 3
    elif "i will have some" in text_lower:
        return 4
    return np.nan  # shouldn't get here


DRINK_KEYWORDS = {
    "sushi": {"yuzu", "milk tea", "baijiu", "fish", "boba", "japanese", "tea", "matcha", "soju", "sake", "miso", "soup",
              "saporo", "ocha", "yakult", "ramune", "soy", "calpis"},
    "shawarma": {"leban", "lassi", "ayran", "laban", "barbican", "yogurt"},
    "pizza": {"powerade", "cocacola", "gatorade", "pops", "dew", "cococola", "jarritos", "pepper", "rootbeer",
              "carbonated", "soft", "dry", "7up", "coke", "cola", "pepsi", "pop", "root beer", "fanta", "soda",
              "sprite", "crush", "ginger", "gingerale", "ale", "brisk"},
    "water": {"sprindrift", "water"},
    "juice": {"fruit", "smoothie", "juice", "orange", "lemon", "mango", "apple", "nestea", "lemonade"},
    "alcohol": {"rum", "champagne", "wine", "beer", "alcohol", "cocktail"},
    "milk": {"dairy", "milk", "milkshake", "hot chocolate"},
    "other": {"coffee"},
    "none": {"none", "no"},
}


def parse_drink_category(text: str) -> str:
    """
    tokenize text and count instances, if multiple equal randomly choose one
    returns the category of drink
    """
    if not isinstance(text, str):
        return "other"

    text_lower = text.lower()

    tokens = re.findall(r"\b\w+\b", text_lower)

    # sum up total token category counts
    category_counts = Counter()
    for token in tokens:
        for category, keywords in DRINK_KEYWORDS.items():
            if token in keywords:
                category_counts[category] += 1

    if not category_counts:
        return "other"

    # choose top category
    max_count = max(category_counts.values())
    top_categories = [cat for cat, count in category_counts.items() if count == max_count]
    return random.choice(top_categories)


def tokenize(text):
    '''
    convert text to lower case and extract all tokens
    '''
    return re.findall(r"\b\w+\b", text.lower()) if isinstance(text, str) else []


def build_vocabulary(series, min_freq=2):
    '''
    series: pandas series, each row is a string
    returns sorted list of words that appear min freq times
    '''

    counter = Counter()
    for txt in series:
        counter.update(tokenize(txt))

    return sorted([word for word, freq in counter.items()
                   if freq >= min_freq and word not in STOPWORDS])


def binary_bow_features(series, vocab):
    '''
    returns dataframe, each row corresponds to a sample,
    each colunmn is a word in vocabulary
    1 if word appears 0 otherwise
    '''
    matrix = []
    for txt in series:
        tokens = set(tokenize(txt))
        row = [1 if word in tokens else 0 for word in vocab]
        matrix.append(row)
    return pd.DataFrame(matrix, columns=[f"Q5_word_{w}" for w in vocab])


def clean_data(df):
    """
    perform data cleaning, based on above
    """

    # strip names for simplicity
    df.columns = df.columns.str.strip()
    df.rename(columns={
        "Q1: From a scale 1 to 5, how complex is it to make this food? (Where 1 is the most simple, and 5 is the most complex)": "Q1",
        "Q2: How many ingredients would you expect this food item to contain?": "Q2",
        "Q3: In what setting would you expect this food to be served? Please check all that apply": "Q3",
        "Q4: How much would you expect to pay for one serving of this food item?": "Q4",
        "Q5: What movie do you think of when thinking of this food item?": "Q5",
        "Q6: What drink would you pair with this food item?": "Q6",
        "Q7: When you think about this food item, who does it remind you of?": "Q7",
        "Q8: How much hot sauce would you add to this food item?": "Q8"
    }, inplace=True)

    # id is unused
    if 'id' in df.columns:
        df.drop("id", axis=1, inplace=True)

    # convert to numbers
    df["Q1"] = pd.to_numeric(df["Q1"], errors="coerce")

    # parse ingredient count
    df["Q2_parsed"] = df["Q2"].apply(parse_ingredient_count)
    # print("after parsing q2:", df.shape, "nans in q2 parsed =", df["Q2_parsed"].isna().sum())
    # nan_q2 = df[df["Q2_parsed"].isna()]
    # print("\nrows where q2 parsed is nan:")
    # print(nan_q2[["Q2", "Q2_parsed"]])

    # parse price
    df["Q4_parsed"] = df["Q4"].apply(parse_price)
    # print("after parsing q4:", df.shape, "nans in q4 parsed =", df["Q4_parsed"].isna().sum())
    # nan_q4 = df[df["Q4_parsed"].isna()]
    # print("\nrows where q4 parsed is nan:")
    # print(nan_q4[["Q4", "q4 parsed"]])

    # parse drinks
    df["Q6_cat"] = df["Q6"].apply(parse_drink_category)
    # print("after parsing Q6:", df.shape)
    # uncategorized = df[df["Q6_cat"] == "other"]
    # print("\nuncategorized drinks:")
    # print(uncategorized[["Q6", "Q6_cat"]].to_string(index=False))

    df["Q8_spice"] = df["Q8"].apply(parse_hot_sauce)
    # print("after parsing Q8:", df.shape, "nans in q8 spice =", df["Q8_spice"].isna().sum())

    # bag of words for q5 movies
    vocab = build_vocabulary(df["Q5"], min_freq=2)
    bow_df = binary_bow_features(df["Q5"], vocab)
    df = pd.concat([df.reset_index(drop=True), bow_df.reset_index(drop=True)], axis=1)
    q5_cols = [c for c in df.columns if c.startswith("Q5_word_")]

    # drop rows without labels (all of them SHOULD have labels)
    if "Label" not in df.columns:
        raise ValueError("No 'Label' column found in data!")
    df = df.dropna(subset=["Label"])

    # Convert Label to 3 classes (0,1,2)
    df["Label_num"] = df["Label"].apply(label_encode_food)
    df = df.dropna(subset=["Label_num"])  # keep only valid 0/1/2

    # q3 multi select (one hot vectors)
    # create a new column for every setting, set to 1 if it was selected
    def split_list(x):
        # helper for q3
        if isinstance(x, str):
            return [v.strip() for v in x.split(',') if v.strip()]
        return []

    all_settings = set()
    for val in df["Q3"]:
        all_settings.update(split_list(val))
    all_settings = sorted(all_settings)

    for s in all_settings:
        df[f"setting_{s}"] = df["Q3"].apply(lambda txt: 1 if s in split_list(txt) else 0)

    # q7 multi select (one hot)
    # "Parents,Siblings,Friends,Teachers,Strangers,None
    # create a new column for every person, set to 1 if it was selected
    all_people = set()
    for val in df["Q7"]:
        all_people.update(split_list(val))
    all_people = sorted(all_people)

    for p in all_people:
        df[f"people_{p}"] = df["Q7"].apply(lambda txt: 1 if p in split_list(txt) else 0)

    # convert drinks to one hot vectors
    temp_drinks = pd.get_dummies(df["Q6_cat"], prefix="drink").astype(int)
    df = pd.concat([df, temp_drinks], axis=1)

    # get the final feature set
    # q1: int
    # q2_parsed: float
    # q4_parsed: float
    # q8_spice: int
    #
    #
    feature_columns = ["Q1", "Q2_parsed", "Q4_parsed", "Q8_spice"]
    setting_cols = [c for c in df.columns if c.startswith("setting_")]
    people_cols = [c for c in df.columns if c.startswith("people_")]
    drink_cols = [c for c in df.columns if c.startswith("drink_")]

    X = df[feature_columns + setting_cols + people_cols + drink_cols + q5_cols].copy()
    y = df["Label_num"].copy()

    # if there are any rows with nan, drop them
    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    y = y.loc[X.index]

    return X, y


def train_test_split(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_seed: int = 42) \
        -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    '''
    our own train_test_split function
    splits feature matrix X and targets y into training and test sets
    '''

    # np.random.seed(random_seed)

    # create random array of indices
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    # number of tests
    n_test = int(len(X) * test_size)

    # self explanatory
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]

    X_train = X.iloc[train_idx].values
    X_test = X.iloc[test_idx].values
    y_train = y.iloc[train_idx].values
    y_test = y.iloc[test_idx].values

    return X_train, X_test, y_train, y_test


def custom_standard_scaler(X_train, X_test):
    '''
    normalizes our training and test data
    '''

    # convert to float (gives me errors if i dont)
    X_train = X_train.astype(float)
    X_test = X_test.astype(float)

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0, ddof=1)
    std_no_zero = np.where(std == 0, 1, std)  # avoid div 0

    # z = (x-mean) / std
    X_train_scaled = (X_train - mean) / std_no_zero
    X_test_scaled = (X_test - mean) / std_no_zero

    return X_train_scaled, X_test_scaled


def one_hot_encode(y, num_classes=3):
    """
    convert array of labels to one hot
    """
    N = len(y)
    T = np.zeros((N, num_classes), dtype=float)
    for i in range(N):
        T[i, y[i]] = 1.0
    return T


def softmax(Z):
    # softmax(z) = softmax(z+c), so we shift down by the highest to avoid overflow
    Z_shifted = Z - np.max(Z, axis=1, keepdims=True)
    expZ = np.exp(Z_shifted)
    sum = np.sum(expZ, axis=1, keepdims=True)
    return expZ / sum


def pred_softmax(W, X):
    """
    linear model prediction + softmax activation function
    """
    Z = X @ W
    return softmax(Z)


REG = 0.005


def cross_entropy_loss(W, X, T, reg=REG):
    """
    Lce = -tlog(y)
    returns scalar
    """
    Y = pred_softmax(W, X)
    buf = 1e-10  # prevent log 0

    loss = -np.mean(np.sum(T * np.log(Y + buf), axis=1))
    l2 = reg * 0.5 * np.sum(W ** 2)
    return l2 + loss


def grad_softmax(W, X, T, reg=REG):
    """
    grad = X^T (Y - T)/N
    """
    N = X.shape[0]
    Y = pred_softmax(W, X)

    dW = (X.T @ (Y - T)) / N
    dW += reg * W

    return dW


def accuracy(W, X, t):
    """
    predict, pick highest probability, then compare with label
    """
    Y = pred_softmax(W, X)
    predictions = np.argmax(Y, axis=1)
    return np.mean(predictions == t)


def solve_via_softmax_gd(X_train, t_train, X_valid, t_valid,
                         alpha, niter, verbose=False):
    '''
    Given `alpha` - the learning rate
          `niter` - the number of iterations of gradient descent to run
          `X_train` - the data matrix to use for training
          `t_train` - the target vector to use for training
          `X_valid` - the data matrix to use for validation
          `t_valid` - the target vector to use for validation
          `plot` - whether to track statistics and plot the training curve

    returns W, learned weight matrix
    '''

    K = 3  # classes

    # convert labels to one hot
    T_train = one_hot_encode(t_train, K)
    T_valid = one_hot_encode(t_valid, K)

    D = X_train.shape[1]
    W = np.zeros((D, K))

    if verbose:
        print(f"Starting training: alpha={alpha}, niter={niter}")

    for i in range(niter):
        dW = grad_softmax(W, X_train, T_train)
        W -= alpha * dW

        if verbose and (i + 1) % max(1, (niter // 5)) == 0:
            tr_loss = cross_entropy_loss(W, X_train, T_train)
            va_loss = cross_entropy_loss(W, X_valid, T_valid)
            tr_acc = accuracy(W, X_train, t_train)
            va_acc = accuracy(W, X_valid, t_valid)
            print(f"Iter {i + 1}/{niter}: "
                  f"train_loss={tr_loss:.3f}, val_loss={va_loss:.3f}, "
                  f"train_acc={tr_acc:.3f}, val_acc={va_acc:.3f}")

    return W


def main():
    # the options have to be set to avoid from "None" being interpreted as python None
    df_raw = pd.read_csv("data/cleaned_data_combined_modified.csv",
                         keep_default_na=False,
                         na_filter=False)

    # clean
    X, y = clean_data(df_raw)

    train_accuracies = []
    val_accuracies = []
    test_accuracies = []

    for run in range(RUNS):
        print(f"\n======= run {run + 1}/{RUNS} =======")

        # train test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_seed=42 + run)

        # scale
        X_train_sc, X_test_sc = custom_standard_scaler(X_train, X_test)

        # train validation split
        X_train2, X_val, y_train2, y_val = train_test_split(
            pd.DataFrame(X_train_sc), pd.Series(y_train),
            test_size=0.2, random_seed=22 + run
        )

        W = solve_via_softmax_gd(X_train2, y_train2, X_val, y_val,
                                 alpha=0.005, niter=2000)

        # evaluate
        train_acc = accuracy(W, X_train2, y_train2)
        val_acc = accuracy(W, X_val, y_val)
        test_acc = accuracy(W, X_test_sc, y_test)

        print(f"training acc: {train_acc:.4f}, validation acc: {val_acc:.4f}, test acc: {test_acc:.4f}")

        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        test_accuracies.append(test_acc)

    print(f"\n=== combined accuracy over {RUNS} runs ===")
    print(f"training: {np.mean(train_accuracies):.4f}")
    print(f"validation: {np.mean(val_accuracies):.4f}")
    print(f"test: {np.mean(test_accuracies):.4f}")


    plt.figure(figsize=(8, 4))
    plt.scatter([1] * RUNS, test_accuracies, alpha=0.8, edgecolors='k')
    plt.xticks([1], ["Test Accuracy"])
    plt.ylabel("Accuracy")
    plt.ylim(0.8, 1)  # <- updated range
    plt.title("Test Accuracy Across 100 Independent Runs")
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("test_accuracy_stripplot.png")
    print("Strip plot saved as test_accuracy_stripplot.png")






if __name__ == "__main__":
    main()
