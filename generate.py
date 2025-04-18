import re
import numpy as np
import pandas as pd
import random
from collections import Counter

RUNS = 1

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

    text_lower = str(text).lower().strip()


    found_nums = []

    digit_matches = re.findall(r"\d+", text_lower)
    for dm in digit_matches:
        try:
            found_nums.append(float(dm))
        except ValueError:
            pass


    for word, val in written_numbers_map.items():
        if re.search(rf"\b{re.escape(word)}\b", text_lower):
            found_nums.append(float(val))

    result = np.nan
    if len(found_nums) >= 1:
        result = sum(found_nums) / len(found_nums)
    else:

        if "*" in text or "\n" in text_lower:
            lines = [line.strip() for line in text_lower.split("\n") if line.strip()]
            bullet_lines = [ln for ln in lines if ln.startswith("*") or ln.startswith("-")]
            if bullet_lines:
                result = len(bullet_lines)
            else:
                result = len(lines)  # fallback
        elif "," in text_lower:
            ingredients = [i.strip() for i in text_lower.split(",") if i.strip()]
            if len(ingredients) > 1:
                result = len(ingredients)

    return result


def parse_price(text: str) -> float:
    text_lower = str(text).lower().strip()
    for word, val in written_numbers_map.items():
        text_lower = re.sub(rf"\b{word}\b", str(val), text_lower)

    nums = re.findall(r"(\d+\.?\d*)", text_lower)
    nums = list(map(float, nums))

    if len(nums) == 0:
        return np.nan
    return sum(nums) / len(nums)


def parse_hot_sauce(text: str) -> float:
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
    return np.nan


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
    if not isinstance(text, str):
        return "other"

    text_lower = text.lower()
    tokens = re.findall(r"\b\w+\b", text_lower)

    from collections import Counter
    category_counts = Counter()
    for token in tokens:
        for category, keywords in DRINK_KEYWORDS.items():
            if token in keywords:
                category_counts[category] += 1

    if not category_counts:
        return "other"

    max_count = max(category_counts.values())
    top_categories = [cat for cat, ccount in category_counts.items() if ccount == max_count]
    return random.choice(top_categories)


def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower()) if isinstance(text, str) else []


def build_vocabulary(series, min_freq=2):
    counter = Counter()
    for txt in series:
        counter.update(tokenize(txt))
    return sorted([word for word, freq in counter.items() if freq >= min_freq])


def binary_bow_features(series, vocab):
    matrix = []
    for txt in series:
        tokens = set(tokenize(txt))
        row = [1 if word in tokens else 0 for word in vocab]
        matrix.append(row)
    return pd.DataFrame(matrix, columns=[f"Q5_word_{w}" for w in vocab])


def split_list(x):
    if isinstance(x, str):
        return [v.strip() for v in x.split(",") if v.strip()]
    return []


def prepare_training_data(df_raw):


    df = df_raw.copy()
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

    if 'id' in df.columns:
        df.drop("id", axis=1, inplace=True)


    df["Q1"] = pd.to_numeric(df["Q1"], errors="coerce")


    df["Q2_parsed"] = df["Q2"].apply(parse_ingredient_count)
    df["Q4_parsed"] = df["Q4"].apply(parse_price)
    df["Q6_cat"] = df["Q6"].apply(parse_drink_category)
    df["Q8_spice"] = df["Q8"].apply(parse_hot_sauce)


    vocab = build_vocabulary(df["Q5"], min_freq=2)
    bow_df = binary_bow_features(df["Q5"], vocab)
    df = pd.concat([df.reset_index(drop=True), bow_df.reset_index(drop=True)], axis=1)


    if "Label" not in df.columns:
        raise ValueError("No 'Label' column found in data!")
    df["Label_num"] = df["Label"].apply(label_encode_food)
    df = df.dropna(subset=["Label_num"])


    all_settings = set()
    for val in df["Q3"]:
        all_settings.update(split_list(val))
    all_settings = sorted(all_settings)

    for s in all_settings:
        df[f"setting_{s}"] = df["Q3"].apply(lambda txt: 1 if s in split_list(txt) else 0)


    all_people = set()
    for val in df["Q7"]:
        all_people.update(split_list(val))
    all_people = sorted(all_people)

    for p in all_people:
        df[f"people_{p}"] = df["Q7"].apply(lambda txt: 1 if p in split_list(txt) else 0)


    drink_dummies = pd.get_dummies(df["Q6_cat"], prefix="drink").astype(int)
    df = pd.concat([df, drink_dummies], axis=1)

    # final features
    feature_columns = ["Q1", "Q2_parsed", "Q4_parsed", "Q8_spice"]
    setting_cols = [c for c in df.columns if c.startswith("setting_")]
    people_cols = [c for c in df.columns if c.startswith("people_")]
    drink_cols = [c for c in df.columns if c.startswith("drink_")]
    bow_cols = [c for c in df.columns if c.startswith("Q5_word_")]

    X = df[feature_columns + setting_cols + people_cols + drink_cols + bow_cols].copy()
    y = df["Label_num"].copy()


    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    y = y.loc[X.index]


    metadata = {
        "feature_columns": feature_columns,
        "setting_cols": setting_cols,
        "people_cols": people_cols,
        "drink_cols": drink_cols,
        "bow_cols": bow_cols,
        "vocab": vocab,
        "all_settings": all_settings,
        "all_people": all_people,
        "unique_drinks": sorted(drink_dummies.columns),
    }
    return X, y, metadata


def train_test_split(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_seed: int = 42):
    indices = np.arange(len(X))
    np.random.seed(random_seed)
    np.random.shuffle(indices)

    n_test = int(len(X) * test_size)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]

    X_train = X.iloc[train_idx].values
    X_test = X.iloc[test_idx].values
    y_train = y.iloc[train_idx].values
    y_test = y.iloc[test_idx].values
    return X_train, X_test, y_train, y_test


def custom_standard_scaler(X_train, X_test):
    X_train = X_train.astype(float)
    X_test = X_test.astype(float)

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0, ddof=1)
    std_no_zero = np.where(std == 0, 1, std)

    X_train_sc = (X_train - mean) / std_no_zero
    X_test_sc = (X_test - mean) / std_no_zero
    return X_train_sc, X_test_sc, mean, std_no_zero


def one_hot_encode(y, num_classes=3):
    N = len(y)
    T = np.zeros((N, num_classes), dtype=float)
    for i in range(N):
        T[i, y[i]] = 1.0
    return T

def softmax(Z):
    Z_shifted = Z - np.max(Z, axis=1, keepdims=True)
    expZ = np.exp(Z_shifted)
    return expZ / np.sum(expZ, axis=1, keepdims=True)

REG = 0.005
def cross_entropy_loss(W, X, T, reg=REG):
    Y = softmax(X @ W)
    buf = 1e-10
    loss = -np.mean(np.sum(T * np.log(Y + buf), axis=1))
    l2 = reg * 0.5 * np.sum(W ** 2)
    return l2 + loss

def grad_softmax(W, X, T, reg=REG):
    N = X.shape[0]
    Y = softmax(X @ W)
    dW = (X.T @ (Y - T)) / N
    dW += reg * W
    return dW

def accuracy(W, X, t):
    Y = softmax(X @ W)
    preds = np.argmax(Y, axis=1)
    return np.mean(preds == t)

def solve_via_softmax_gd(X_train, y_train, X_val, y_val,
                         alpha=0.005, niter=2000, verbose=True):
    K = 3
    T_train = one_hot_encode(y_train, K)
    T_val = one_hot_encode(y_val, K)

    D = X_train.shape[1]
    W = np.zeros((D, K))

    for i in range(niter):
        dW = grad_softmax(W, X_train, T_train)
        W -= alpha * dW

        if verbose and (i + 1) % max(1, (niter // 5)) == 0:
            tr_loss = cross_entropy_loss(W, X_train, T_train)
            va_loss = cross_entropy_loss(W, X_val, T_val)
            tr_acc = accuracy(W, X_train, y_train)
            va_acc = accuracy(W, X_val, y_val)
            print(f"Iter {i + 1}/{niter}: "
                  f"train_loss={tr_loss:.3f}, val_loss={va_loss:.3f}, "
                  f"train_acc={tr_acc:.3f}, val_acc={va_acc:.3f}")

    return W


def main():
    df_raw = pd.read_csv("data/cleaned_data_combined_modified.csv",
                         keep_default_na=False,
                         na_filter=False)

    X_df, y_series, metadata = prepare_training_data(df_raw)


    X_train, X_test, y_train, y_test = train_test_split(X_df, y_series, test_size=0.2, random_seed=42)

    X_train2, X_val, y_train2, y_val = train_test_split(pd.DataFrame(X_train), pd.Series(y_train),
                                                        test_size=0.2, random_seed=99)


    X_train2_sc, X_val_sc, mean, std_no_zero = custom_standard_scaler(X_train2, X_val)
    X_test_sc = (X_test - mean) / std_no_zero


    W = solve_via_softmax_gd(X_train2_sc, y_train2, X_val_sc, y_val,
                             alpha=0.005, niter=2000, verbose=True)


    train_acc = accuracy(W, X_train2_sc, y_train2)
    val_acc = accuracy(W, X_val_sc, y_val)
    test_acc = accuracy(W, X_test_sc, y_test)
    print(f"\ntrain_acc={train_acc:.4f}, val_acc={val_acc:.4f}, test_acc={test_acc:.4f}")

    # save parameters
    np.savez("model_params.npz",
             W=W,
             mean=mean,
             std_no_zero=std_no_zero,
             feature_columns=metadata["feature_columns"],
             setting_cols=metadata["setting_cols"],
             people_cols=metadata["people_cols"],
             drink_cols=metadata["drink_cols"],
             bow_cols=metadata["bow_cols"],
             vocab=metadata["vocab"],
             all_settings=metadata["all_settings"],
             all_people=metadata["all_people"],
             unique_drinks=metadata["unique_drinks"])

    print("\nsaved")

if __name__ == "__main__":
    main()
