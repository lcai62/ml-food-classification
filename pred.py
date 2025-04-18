import re
import numpy as np
import pandas as pd
import random
from collections import Counter



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



def prepare_prediction_data(df_raw, model_info):

    df = df_raw.copy()
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


    for colname in ["Q1","Q2","Q3","Q4","Q5","Q6","Q7","Q8"]:
        if colname not in df.columns:
            df[colname] = ""


    df["Q1"] = pd.to_numeric(df["Q1"], errors="coerce")


    df["Q2_parsed"] = df["Q2"].apply(parse_ingredient_count)
    df["Q4_parsed"] = df["Q4"].apply(parse_price)
    df["Q6_cat"] = df["Q6"].apply(parse_drink_category)
    df["Q8_spice"] = df["Q8"].apply(parse_hot_sauce)

    # bag of words for q5 movies
    vocab = model_info["vocab"]
    bow_matrix = []
    for txt in df["Q5"]:
        tokens = set(tokenize(txt))
        row = [1 if w in tokens else 0 for w in vocab]
        bow_matrix.append(row)
    bow_df = pd.DataFrame(bow_matrix, columns=[f"Q5_word_{w}" for w in vocab])
    df = pd.concat([df.reset_index(drop=True), bow_df.reset_index(drop=True)], axis=1)

    def split_list(x):
        return [v.strip() for v in x.split(",") if isinstance(x, str) and v.strip()]

    # q3 multi select (one hot vectors)
    # create a new column for every setting, set to 1 if it was selected
    for s in model_info["all_settings"]:
        df[f"setting_{s}"] = df["Q3"].apply(lambda txt: 1 if s in split_list(txt) else 0)

    # q7 multi select (one hot)
    # "Parents,Siblings,Friends,Teachers,Strangers,None
    # create a new column for every person, set to 1 if it was selected
    for p in model_info["all_people"]:
        df[f"people_{p}"] = df["Q7"].apply(lambda txt: 1 if p in split_list(txt) else 0)

    # convert drinks to one hot vectors
    drink_dummies = pd.get_dummies(df["Q6_cat"], prefix="drink").astype(int)

    for dcol in model_info["unique_drinks"]:
        if dcol not in drink_dummies.columns:
            drink_dummies[dcol] = 0
    drink_dummies = drink_dummies[model_info["unique_drinks"]]

    df = pd.concat([df, drink_dummies], axis=1)

    # final feature set
    feat_cols = (model_info["feature_columns"] +
                 model_info["setting_cols"] +
                 model_info["people_cols"] +
                 model_info["drink_cols"] +
                 model_info["bow_cols"])

    for col in feat_cols:
        if col not in df.columns:
            df[col] = 0

    X = df[feat_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)  # fill any missing
    return X


def softmax(Z):
    # softmax(z) = softmax(z+c), so we shift down by the highest to avoid overflow
    Z_shifted = Z - np.max(Z, axis=1, keepdims=True)
    expZ = np.exp(Z_shifted)
    return expZ / np.sum(expZ, axis=1, keepdims=True)

def predict_all(csv_filename: str):

    data = np.load("model_params.npz", allow_pickle=True)
    W = data["W"]
    mean = data["mean"]
    std_no_zero = data["std_no_zero"]
    model_info = {
        "feature_columns": data["feature_columns"].tolist(),
        "setting_cols": data["setting_cols"].tolist(),
        "people_cols": data["people_cols"].tolist(),
        "drink_cols": data["drink_cols"].tolist(),
        "bow_cols": data["bow_cols"].tolist(),
        "vocab": data["vocab"].tolist(),
        "all_settings": data["all_settings"].tolist(),
        "all_people": data["all_people"].tolist(),
        "unique_drinks": data["unique_drinks"].tolist()
    }


    df_new = pd.read_csv(csv_filename, keep_default_na=False, na_filter=False)
    X_new = prepare_prediction_data(df_new, model_info)


    X_new = X_new.values.astype(float)
    X_new_sc = (X_new - mean) / std_no_zero


    Y_proba = softmax(X_new_sc @ W)
    predictions = np.argmax(Y_proba, axis=1)

    label_map = {0: "Pizza", 1: "Shawarma", 2: "Sushi"}
    predicted_labels = [label_map[p] for p in predictions]
    return predicted_labels


if __name__ == "__main__":
    preds = predict_all("data/cleaned_data_combined.csv")
    print("Predictions:", preds)

