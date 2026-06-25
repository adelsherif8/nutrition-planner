import re
import json
import base64
import torch
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
from groq import Groq
from transformers import pipeline
import requests

from google.colab import drive
drive.mount('/content/drive')

GROQ_API_KEY  = "<YOUR_GROQ_API_KEY>"
HF_TOKEN      = "<YOUR_HF_TOKEN>"
HF_MODEL      = "meta-llama/Llama-3.2-1B-Instruct"
VISION_MODEL  = "meta-llama/llama-4-scout-17b-16e-instruct"

client = Groq(api_key=GROQ_API_KEY)

print(f"Loading {HF_MODEL} locally...")
hf_pipe = pipeline(
    "text-generation",
    model=HF_MODEL,
    token=HF_TOKEN,
    torch_dtype="auto",
    device_map="auto"
)
print("Model loaded.")

DATA_DIR      = "FoodData_Central_sr_legacy_food_csv_2018-04"
KEY_NUTRIENTS = {1003: "protein", 1004: "fat", 1005: "carbs", 1008: "calories"}
MEAL_SPLITS   = {"breakfast": 0.25, "lunch": 0.35, "dinner": 0.30, "snack": 0.10}

# Agent 1 User Profile Collector

def lbs_to_kg(lbs): return round(lbs * 0.453592, 1)
def ft_in_to_cm(feet, inches=0): return round(feet * 30.48 + inches * 2.54, 1)

def parse_imperial(text):
    t = text.lower()
    result = {}
    import re as _re
    m = _re.search(r'(\d+(?:\.\d+)?)\s*(lbs?|pounds?)', t)
    if m:
        result["weight_kg"] = lbs_to_kg(float(m.group(1)))
        result["_weight_note"] = f"{m.group(1)} lbs → {result['weight_kg']} kg"
    m = _re.search(r'(\d+)\s*(?:ft|feet|\')\s*(\d+)\s*(?:in|inches|\")?', t)
    if m:
        result["height_cm"] = ft_in_to_cm(int(m.group(1)), int(m.group(2)))
        result["_height_note"] = f"{m.group(1)}'{m.group(2)}\" → {result['height_cm']} cm"
    else:
        m = _re.search(r'(\d+)\s*(?:ft|feet|\')', t)
        if m:
            result["height_cm"] = ft_in_to_cm(int(m.group(1)))
    return result

class UserProfileAgent:

    def __init__(self):
        self.profile = {
            "age": None, "gender": None, "weight_kg": None,
            "height_cm": None, "activity_level": None, "goal": None
        }

    def extract_from_text(self, text):
        t = text.lower()
        imperial = parse_imperial(text)
        if "weight_kg" in imperial:
            self.profile["weight_kg"] = imperial["weight_kg"]
            print(f"  {imperial['_weight_note']}")
        if "height_cm" in imperial:
            self.profile["height_cm"] = imperial["height_cm"]
            if "_height_note" in imperial:
                print(f"  {imperial['_height_note']}")

        def find(pattern):
            m = re.search(pattern, t)
            return float(m.group(1)) if m else None

        if not self.profile["weight_kg"]:
            w = find(r'(\d+(?:\.\d+)?)\s*(kg|kgs|kilogram)')
            if w: self.profile["weight_kg"] = float(w)
        if not self.profile["height_cm"]:
            h = find(r'(\d+(?:\.\d+)?)\s*(cm|centimeter)')
            if h: self.profile["height_cm"] = float(h)

        age = find(r'(\d+)\s*(year|years|yo|y\.o)')
        if age: self.profile["age"] = int(age)

        if any(w in t for w in ["male", "man", " guy"]):
            self.profile["gender"] = "male"
        elif any(w in t for w in ["female", "woman", "girl"]):
            self.profile["gender"] = "female"

        if any(w in t for w in ["lose", "fat", "cut", "slim", "shred"]):
            self.profile["goal"] = "weight_loss"
        elif any(w in t for w in ["gain", "muscle", "bulk", "mass"]):
            self.profile["goal"] = "muscle_gain"
        elif any(w in t for w in ["maintain", "stay"]):
            self.profile["goal"] = "maintenance"

        if any(w in t for w in ["sedentary", "no exercise", "inactive", "desk"]):
            self.profile["activity_level"] = "sedentary"
        elif any(w in t for w in ["light", "walk", "once a week"]):
            self.profile["activity_level"] = "light"
        elif any(w in t for w in ["moderate", "3x", "sometimes"]):
            self.profile["activity_level"] = "moderate"
        elif any(w in t for w in ["active", "daily", "intense", "athlete", "5x"]):
            self.profile["activity_level"] = "active"

    def _ask(self, prompt, cast=str):
        while True:
            val = input(prompt).strip()
            if not val:
                print("  Please enter a value.")
                continue
            try:
                return cast(val)
            except:
                print("  Invalid — try again.")

    def _ask_measure(self, prompt):
        while True:
            raw = input(prompt).strip()
            if not raw:
                print("  Please enter a value.")
                continue
            imp = parse_imperial(raw)
            if imp:
                return list(imp.values())[0]
            stripped = re.sub(r"[^\d.]", "", raw)
            if stripped:
                return float(stripped)
            print("  Could not parse — try again (e.g. 80kg or 180lbs).")

    def ask_missing(self):
        if self.profile["age"] is None:
            self.profile["age"] = self._ask("Age: ", int)
        if self.profile["gender"] is None:
            self.profile["gender"] = self._ask("Sex (male/female): ")
        if self.profile["weight_kg"] is None:
            self.profile["weight_kg"] = self._ask_measure("Weight (e.g. 80kg or 180lbs): ")
        if self.profile["height_cm"] is None:
            self.profile["height_cm"] = self._ask_measure("Height (e.g. 175cm or 5'11): ")
        if self.profile["activity_level"] is None:
            print("  Options: sedentary / light / moderate / active")
            self.profile["activity_level"] = self._ask("Activity: ")
        if self.profile["goal"] is None:
            print("  Options: weight_loss / muscle_gain / maintenance")
            self.profile["goal"] = self._ask("Goal: ")

    def run(self):
        print("Describe yourself (or press Enter to fill in manually):")
        text = input("> ")
        if text.strip():
            self.extract_from_text(text)
            extracted = {k: v for k, v in self.profile.items() if v is not None and not k.startswith("_")}
            if extracted:
                print(f"  Got: {extracted}")
        self.ask_missing()
        return self.profile

def agent1_to_agent2(profile):
    return {
        "age": profile["age"], "sex": profile["gender"],
        "height_cm": profile["height_cm"], "weight_kg": profile["weight_kg"],
        "activity_level": profile["activity_level"], "goal": profile["goal"]
    }

# Agent 2 Macro Calculator & Validator

VALID_SEXES          = {"male", "female"}
VALID_ACTIVITY_LEVELS = {"sedentary", "light", "moderate", "active"}
VALID_GOALS          = {"weight_loss", "maintenance", "muscle_gain"}
VALID_DIET_TYPES     = {"balanced", "vegetarian", "vegan", "keto", "low_carb"}

ACTIVITY_NORM_MAP = {
    "none": "sedentary", "no exercise": "sedentary", "desk job": "sedentary",
    "inactive": "sedentary", "couch": "sedentary",
    "light": "light", "walks": "light", "walking": "light",
    "1x week": "light", "1-2x week": "light", "light exercise": "light",
    "moderate": "moderate", "gym 3x": "moderate", "gym 3x/week": "moderate",
    "3x week": "moderate", "3-4x week": "moderate", "sometimes": "moderate",
    "regular": "moderate",
    "active": "active", "very active": "active", "daily": "active",
    "5x week": "active", "5-6x week": "active", "athlete": "active", "intense": "active",
}

GOAL_NORM_MAP = {
    "lose weight": "weight_loss", "lose fat": "weight_loss", "cut": "weight_loss",
    "cutting": "weight_loss", "slim down": "weight_loss", "fat loss": "weight_loss",
    "shred": "weight_loss", "get shredded": "weight_loss", "calorie deficit": "weight_loss",
    "maintain": "maintenance", "maintain weight": "maintenance", "stay same": "maintenance",
    "keep weight": "maintenance", "healthy": "maintenance",
    "gain muscle": "muscle_gain", "build muscle": "muscle_gain", "bulk": "muscle_gain",
    "bulking": "muscle_gain", "gain mass": "muscle_gain", "get bigger": "muscle_gain",
    "hypertrophy": "muscle_gain",
}

DIET_NORM_MAP = {
    "balanced": "balanced", "normal": "balanced", "everything": "balanced", "omnivore": "balanced",
    "vegetarian": "vegetarian", "veg": "vegetarian", "no meat": "vegetarian",
    "vegan": "vegan", "plant based": "vegan", "plant-based": "vegan",
    "keto": "keto", "ketogenic": "keto",
    "low carb": "low_carb", "low-carb": "low_carb", "low carbohydrate": "low_carb",
}

FIELD_DEFAULTS = {"medical_conditions": [], "diet_type": "balanced"}

@dataclass
class UserProfile:
    age: int
    sex: str
    height_cm: float
    weight_kg: float
    activity_level: str
    goal: str
    medical_conditions: list = field(default_factory=list)
    diet_type: str = "balanced"

@dataclass
class ValidationResult:
    valid: bool
    profile: Optional[UserProfile]
    errors: list
    warnings: list
    normalizations: list

def _normalize_string(value):
    return re.sub(r"\s+", " ", str(value).strip().lower())

def _lookup_norm_map(value, norm_map):
    return norm_map.get(_normalize_string(value))

def normalize_activity(raw):
    cleaned = _normalize_string(raw)
    if cleaned in VALID_ACTIVITY_LEVELS: return cleaned, True
    return _lookup_norm_map(cleaned, ACTIVITY_NORM_MAP), False

def normalize_goal(raw):
    cleaned = _normalize_string(raw)
    if cleaned in VALID_GOALS: return cleaned, True
    return _lookup_norm_map(cleaned, GOAL_NORM_MAP), False

def normalize_diet(raw):
    cleaned = _normalize_string(raw)
    if cleaned in VALID_DIET_TYPES: return cleaned, True
    return _lookup_norm_map(cleaned, DIET_NORM_MAP), False

def validate_input(raw_input):
    errors, warnings, normalizations = [], [], []
    data = {**FIELD_DEFAULTS, **raw_input}

    for key in ["age", "sex", "height_cm", "weight_kg", "activity_level", "goal"]:
        if key not in data or data[key] is None or str(data[key]).strip() == "":
            errors.append(f"Missing required field: '{key}'")
    if errors:
        return ValidationResult(False, None, errors, warnings, normalizations)

    try:
        age = int(data["age"])
        if not (1 <= age <= 120): errors.append(f"'age' out of range: {age}")
    except: errors.append(f"'age' must be integer, got: {data['age']!r}"); age = None

    sex_raw = _normalize_string(str(data["sex"]))
    sex = sex_raw if sex_raw in VALID_SEXES else None
    if sex is None: errors.append(f"'sex' must be male/female, got: {data['sex']!r}")

    try:
        height_cm = float(data["height_cm"])
        if not (50 <= height_cm <= 250): errors.append(f"'height_cm' out of range: {height_cm}")
    except: errors.append(f"'height_cm' must be numeric"); height_cm = None

    try:
        weight_kg = float(data["weight_kg"])
        if not (20 <= weight_kg <= 300): errors.append(f"'weight_kg' out of range: {weight_kg}")
    except: errors.append(f"'weight_kg' must be numeric"); weight_kg = None

    activity, already = normalize_activity(str(data["activity_level"]))
    if activity is None:
        activity = {"__llm_needed__": str(data["activity_level"])}
        warnings.append("activity_level unrecognised — LLM fallback")
    elif not already:
        normalizations.append(f"activity_level: '{data['activity_level']}' → '{activity}'")

    goal, already = normalize_goal(str(data["goal"]))
    if goal is None:
        goal = {"__llm_needed__": str(data["goal"])}
        warnings.append("goal unrecognised — LLM fallback")
    elif not already:
        normalizations.append(f"goal: '{data['goal']}' → '{goal}'")

    diet, already = normalize_diet(str(data.get("diet_type", "balanced")))
    if diet is None:
        diet = "balanced"
        warnings.append("diet_type unrecognised — defaulting to 'balanced'")
    elif not already:
        normalizations.append(f"diet_type: '{data.get('diet_type')}' → '{diet}'")

    med = data.get("medical_conditions", [])
    if not isinstance(med, list):
        med = [m.strip() for m in med.split(",")] if isinstance(med, str) and med.strip() else []

    if errors:
        return ValidationResult(False, None, errors, warnings, normalizations)

    return ValidationResult(
        valid=True,
        profile=UserProfile(age=age, sex=sex, height_cm=height_cm, weight_kg=weight_kg,
                            activity_level=activity, goal=goal,
                            medical_conditions=med, diet_type=diet),
        errors=errors, warnings=warnings, normalizations=normalizations
    )

def llm_semantic_normalization(field_name, raw_value):
    allowed = {
        "activity_level": "sedentary, light, moderate, active",
        "goal": "weight_loss, maintenance, muscle_gain"
    }
    prompt = (f'Map "{raw_value}" to one of: {allowed[field_name]}.\n'
              f'Return ONLY: {{"value":"..."}}')
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    content = re.sub(r"```json|```", "", response.choices[0].message.content).strip()
    try:
        return json.loads(content)["value"]
    except:
        return None

def resolve_llm_fields(profile):
    if isinstance(profile.goal, dict):
        profile.goal = llm_semantic_normalization("goal", profile.goal["__llm_needed__"])
    if isinstance(profile.activity_level, dict):
        profile.activity_level = llm_semantic_normalization("activity_level", profile.activity_level["__llm_needed__"])
    return profile

def safety_check(profile):
    bmi = profile.weight_kg / ((profile.height_cm / 100) ** 2)
    if profile.age < 18:   return {"status": "blocked", "reason": "Underage"}
    if bmi < 18.5:         return {"status": "blocked", "reason": "Underweight"}
    if bmi > 35:           return {"status": "blocked", "reason": "Severely obese"}
    if profile.medical_conditions: return {"status": "blocked", "reason": "Medical condition present"}
    return {"status": "safe", "bmi": round(bmi, 2)}

def generate_explanation(profile, targets):
    prompt = (f"Explain in 1-2 sentences why these targets make sense. No medical advice.\n"
              f"User: {profile.age}yo {profile.sex}, {profile.weight_kg}kg, goal: {profile.goal}\n"
              f"Targets: {targets['daily_calories']} kcal | {targets['macros']['protein_g']}g protein | "
              f"{targets['macros']['fat_g']}g fat | {targets['macros']['carbs_g']}g carbs")
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return response.choices[0].message.content.strip()

CALCULATOR_TOOLS = [
    {"type": "function", "function": {
        "name": "calculate_bmr",
        "description": "Calculate Basal Metabolic Rate using Mifflin-St Jeor formula",
        "parameters": {"type": "object", "required": ["weight_kg","height_cm","age","sex"],
            "properties": {
                "weight_kg": {"type": "number"}, "height_cm": {"type": "number"},
                "age":       {"type": "integer"}, "sex": {"type": "string", "enum": ["male","female"]}
            }}}},
    {"type": "function", "function": {
        "name": "calculate_tdee",
        "description": "Multiply BMR by activity factor to get Total Daily Energy Expenditure",
        "parameters": {"type": "object", "required": ["bmr","activity_level"],
            "properties": {
                "bmr": {"type": "number"},
                "activity_level": {"type": "string", "enum": ["sedentary","light","moderate","active"]}
            }}}},
    {"type": "function", "function": {
        "name": "apply_goal",
        "description": "Adjust TDEE by goal: weight_loss -400, muscle_gain +300, maintenance 0",
        "parameters": {"type": "object", "required": ["tdee","goal"],
            "properties": {
                "tdee": {"type": "number"},
                "goal": {"type": "string", "enum": ["weight_loss","muscle_gain","maintenance"]}
            }}}},
    {"type": "function", "function": {
        "name": "calculate_macros",
        "description": "Calculate protein, fat, carbs from daily calories and user weight/goal",
        "parameters": {"type": "object", "required": ["calories","weight_kg","goal"],
            "properties": {
                "calories":  {"type": "number"}, "weight_kg": {"type": "number"},
                "goal":      {"type": "string", "enum": ["weight_loss","muscle_gain","maintenance"]}
            }}}}
]

_TOOL_FN_MAP = {
    "calculate_bmr":    lambda weight_kg, height_cm, age, sex: round(
        10*weight_kg + 6.25*height_cm - 5*age + (5 if sex=="male" else -161), 1),
    "calculate_tdee":   lambda bmr, activity_level: round(
        bmr * {"sedentary":1.2,"light":1.375,"moderate":1.55,"active":1.725}[activity_level], 1),
    "apply_goal":       lambda tdee, goal: round(
        max(tdee + {"weight_loss":-400,"muscle_gain":300,"maintenance":0}[goal], 1200), 1),
    "calculate_macros": lambda calories, weight_kg, goal: {
        "protein_g": round(weight_kg*(2.0 if goal=="muscle_gain" else 1.8 if goal=="weight_loss" else 1.4)),
        "fat_g":     round(calories*0.25/9),
        "carbs_g":   round((calories - weight_kg*(2.0 if goal=="muscle_gain" else 1.8 if goal=="weight_loss" else 1.4)*4 - calories*0.25)/4)
    }
}

def _direct_calculate(profile):
    bmr  = round(10*profile.weight_kg + 6.25*profile.height_cm - 5*profile.age + (5 if profile.sex=="male" else -161))
    tdee = round(bmr * {"sedentary":1.2,"light":1.375,"moderate":1.55,"active":1.725}[profile.activity_level])
    cal  = round(max(tdee + {"weight_loss":-400,"muscle_gain":300,"maintenance":0}[profile.goal], 1200))
    prot = round(profile.weight_kg * (2.0 if profile.goal=="muscle_gain" else 1.8 if profile.goal=="weight_loss" else 1.4))
    fat  = round(cal * 0.25 / 9)
    carbs = round((cal - prot*4 - cal*0.25) / 4)
    return {"bmr":bmr,"tdee":tdee,"daily_calories":cal,"macros":{"protein_g":prot,"fat_g":fat,"carbs_g":carbs}}

def run_calculator_with_tools(profile) -> dict:
    direct = _direct_calculate(profile)
    try:
        messages = [{"role": "user", "content": (
            f"Calculate nutrition targets step by step using the tools.\n"
            f"Call: calculate_bmr → calculate_tdee → apply_goal → calculate_macros\n\n"
            f"Profile: age={profile.age}, sex={profile.sex}, weight={profile.weight_kg}kg, "
            f"height={profile.height_cm}cm, activity={profile.activity_level}, goal={profile.goal}"
        )}]
        tool_results = {}
        for _ in range(6):
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=messages,
                tools=CALCULATOR_TOOLS,
                tool_choice="auto"
            )
            msg = response.choices[0].message
            if not msg.tool_calls:
                break
            messages.append(msg)
            for tc in msg.tool_calls:
                fn_name   = tc.function.name
                fn_args   = json.loads(tc.function.arguments)
                fn_result = _TOOL_FN_MAP[fn_name](**fn_args)
                tool_results[fn_name] = fn_result
                print(f"  {fn_name} → {fn_result}")
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(fn_result)})
        cal    = tool_results.get("apply_goal") or tool_results.get("calculate_tdee", 0)
        macros = tool_results.get("calculate_macros", {})
        if 1200 <= cal <= 5000 and macros.get("protein_g", 0) > 0:
            return {"bmr":   tool_results.get("calculate_bmr", direct["bmr"]),
                    "tdee":  tool_results.get("calculate_tdee", direct["tdee"]),
                    "daily_calories": round(cal), "macros": macros}
    except Exception:
        pass
    return direct

def run_agent2(raw_input):
    result = validate_input(raw_input)
    if not result.valid:
        return {"status": "error", "errors": result.errors}
    profile = resolve_llm_fields(result.profile)
    safety  = safety_check(profile)
    if safety["status"] != "safe":
        return {"status": "blocked", "reason": safety["reason"]}
    targets     = run_calculator_with_tools(profile)
    explanation = generate_explanation(profile, targets)
    return {"status": "success", "profile": profile, "targets": targets, "explanation": explanation}

# Agent 4 Food Database

USDA_API_KEY = "DEMO_KEY"

def _search_usda(food_name):
    try:
        r = requests.get(
            "https://api.nal.usda.gov/fdc/v1/foods/search",
            params={"query": food_name, "api_key": USDA_API_KEY,
                    "pageSize": 5, "dataType": "SR Legacy,Foundation"},
            timeout=8
        )
        foods = r.json().get("foods", [])
        if not foods: return None
        food = sorted(foods, key=lambda f: len(f.get("description","")))[0]
        nmap = {1003:"protein", 1004:"fat", 1005:"carbs", 1008:"calories"}
        macros = {nmap[n["nutrientId"]]: round(n.get("value",0),1)
                  for n in food.get("foodNutrients",[]) if n.get("nutrientId") in nmap}
        return {"food_name": food["description"].lower(), "source": "USDA", "per_100g": macros}
    except:
        return None

def build_food_db():
    foods          = pd.read_csv(f"/content/drive/MyDrive/FoodData_Central_sr_legacy_food_csv_2018-04/food.csv")
    food_nutrients = pd.read_csv(f"/content/drive/MyDrive/FoodData_Central_sr_legacy_food_csv_2018-04/food_nutrient.csv")
    filtered = food_nutrients[food_nutrients["nutrient_id"].isin(KEY_NUTRIENTS)].copy()
    filtered["nutrient_name"] = filtered["nutrient_id"].map(KEY_NUTRIENTS)
    pivoted = filtered.pivot_table(
        index="fdc_id", columns="nutrient_name", values="amount", aggfunc="first"
    ).reset_index()
    df = pivoted.merge(foods[["fdc_id", "description"]], on="fdc_id")
    df = df.dropna(subset=["calories", "protein", "fat", "carbs"])
    df = df[df["calories"] > 0].reset_index(drop=True)
    df["food_name"] = df["description"].str.lower()
    return df[["food_name", "calories", "protein", "carbs", "fat"]]

class FoodDatabaseAgent:

    def __init__(self, dataframe):
        self.df = dataframe.dropna(subset=["calories", "protein", "carbs", "fat"]).copy()

    def get_food(self, name):
        name = name.lower().strip()
        matches = self.df[self.df["food_name"].str.contains(name, na=False, regex=False)]
        if not matches.empty:
            return matches.iloc[0].to_dict()
        for word in name.split():
            if len(word) < 3: continue
            matches = self.df[self.df["food_name"].str.contains(word, na=False, regex=False)]
            if not matches.empty:
                return matches.iloc[0].to_dict()
        return None

    def calculate_macros(self, name, grams):
        food = self.get_food(name)
        if not food: return None
        f = grams / 100
        return {
            "food_name": food["food_name"], "grams": grams,
            "calories": round(food["calories"] * f, 1),
            "protein":  round(food["protein"]  * f, 1),
            "carbs":    round(food["carbs"]    * f, 1),
            "fat":      round(food["fat"]      * f, 1)
        }

    def calculate_meal(self, items):
        total   = {"calories": 0, "protein": 0, "carbs": 0, "fat": 0}
        details = []
        for name, grams in items:
            macros = self.calculate_macros(name, grams)
            if not macros: continue
            details.append(macros)
            for k in total: total[k] += macros[k]
        return {"items": details, "macros": {k: round(v, 1) for k, v in total.items()}}

print("Loading food database...")
food_df = build_food_db()
food_db = FoodDatabaseAgent(food_df)
print(f"{len(food_df)} foods loaded")

# Agent 3 Meal Planner

class MealPlanAgent:

    def __init__(self, food_db: FoodDatabaseAgent, pipe=None):
        self.db   = food_db
        self.pipe = pipe or hf_pipe

    def _call_llm(self, targets, profile):
        cal   = targets["daily_calories"]
        prot  = targets["macros"]["protein_g"]
        fat   = targets["macros"]["fat_g"]
        carbs = targets["macros"]["carbs_g"]
        p     = profile if isinstance(profile, dict) else profile.__dict__

        splits_text = "\n".join(
            f"  {meal}: {round(cal*pct)} kcal, {round(prot*pct)}g protein, "
            f"{round(fat*pct)}g fat, {round(carbs*pct)}g carbs"
            for meal, pct in MEAL_SPLITS.items()
        )

        example_format = (
            '{"breakfast":[{"food":"oats","grams":80},{"food":"egg whites","grams":200}],'
            '"lunch":[{"food":"chicken breast","grams":150},{"food":"rice","grams":130}],'
            '"dinner":[{"food":"turkey breast","grams":150},{"food":"sweet potato","grams":150}],'
            '"snack":[{"food":"banana","grams":100},{"food":"greek yogurt","grams":150}]}'
        )

        prompt = (
            f"You are a strict nutrition planner. Create a daily meal plan.\n\n"
            f"User: {p.get('sex','male')}, {p.get('weight_kg',70)}kg, "
            f"goal: {p.get('goal','maintenance')}, diet: {p.get('diet_type','balanced')}\n"
            f"Daily: {cal} kcal | {prot}g protein | {fat}g fat | {carbs}g carbs\n\n"
            f"Per meal:\n{splits_text}\n\n"
            f"Rules:\n- ONLY valid JSON, no markdown\n"
            f"- Use lean whole foods: chicken breast, turkey, egg whites, oats, rice, potato, banana, greek yogurt\n"
            f"- AVOID: peanut butter, nuts, oils, cheese, candy, baby food, processed snacks\n"
            f"- Keep grams realistic (max 300g per item)\n- Each meal: 2-3 foods\n\nFormat:\n{example_format}"
        )

        messages = [{"role": "user", "content": prompt}]
        output = self.pipe(messages, max_new_tokens=1024, temperature=0.4, do_sample=True)
        raw = output[0]["generated_text"][-1]["content"]
        raw = re.sub(r"```json|```", "", raw).strip()
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        if not m:
            raise ValueError(f"No JSON in response: {raw[:200]}")
        return json.loads(m.group(0))

    def _verify_meals(self, meal_json):
        verified = {}
        for meal, items in meal_json.items():
            if not isinstance(items, list):
                continue
            pairs = [(i["food"], i["grams"]) for i in items if isinstance(i, dict)]
            verified[meal] = self.db.calculate_meal(pairs)
        return verified

    def generate(self, agent2_output):
        targets   = agent2_output["targets"]
        profile   = agent2_output["profile"]
        meal_json = self._call_llm(targets, profile)
        verified  = self._verify_meals(meal_json)
        total     = {k: round(sum(m["macros"][k] for m in verified.values()), 1)
                     for k in ["calories", "protein", "carbs", "fat"]}
        return {"meals": verified, "total_macros": total}

# Substitution Agent

def parse_dislikes(feedback: str) -> list:
    prompt = f'Extract disliked foods from: "{feedback}"\nReturn ONLY a JSON array: ["food1","food2"]'
    r = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    try:
        return json.loads(re.sub(r"```json|```", "", r.choices[0].message.content).strip())
    except:
        return []

def apply_substitutions(meal_plan: dict, feedback: str, db) -> tuple:
    dislikes = parse_dislikes(feedback)
    if not dislikes:
        return meal_plan, []

    changes = []
    for meal_name, meal_data in meal_plan["meals"].items():
        new_items = []
        for item in meal_data["items"]:
            fn = item["food_name"].lower()
            hated = any(d.lower() in fn or fn in d.lower() for d in dislikes)
            if hated:
                prompt = (
                    f'Suggest 5 substitutes for "{item["food_name"]}" not in {dislikes}. '
                    f'Simple whole foods only.\nReturn ONLY JSON array: ["food1","food2","food3","food4","food5"]'
                )
                r = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3
                )
                try:
                    alts = json.loads(re.sub(r"```json|```", "", r.choices[0].message.content).strip())
                    found = False
                    for alt in alts:
                        if any(d.lower() in alt.lower() for d in dislikes): continue
                        rc = db.calculate_macros(alt, item["grams"])
                        if rc:
                            changes.append(f"{item['food_name']} → {rc['food_name']}")
                            new_items.append(rc)
                            found = True
                            break
                    if not found: new_items.append(item)
                except:
                    new_items.append(item)
            else:
                new_items.append(item)

        total = {k: round(sum(i[k] for i in new_items), 1) for k in ["calories","protein","carbs","fat"]}
        meal_plan["meals"][meal_name] = {"items": new_items, "macros": total}

    meal_plan["total_macros"] = {
        k: round(sum(m["macros"][k] for m in meal_plan["meals"].values()), 1)
        for k in ["calories","protein","carbs","fat"]
    }
    return meal_plan, changes

# Pipeline Orchestrator

class NutritionPipeline:

    def __init__(self):
        self.agent1 = UserProfileAgent()
        self.agent3 = MealPlanAgent(food_db)

    def run(self):
        raw_profile   = self.agent1.run()
        agent2_input  = agent1_to_agent2(raw_profile)
        agent2_result = run_agent2(agent2_input)

        if agent2_result["status"] == "error":
            print("Error:", agent2_result["errors"])
            return agent2_result
        if agent2_result["status"] == "blocked":
            print("Blocked:", agent2_result["reason"])
            return agent2_result

        t = agent2_result["targets"]
        print(f"Targets: {t['daily_calories']} kcal  |  P {t['macros']['protein_g']}g  F {t['macros']['fat_g']}g  C {t['macros']['carbs_g']}g")
        print(f"{agent2_result['explanation']}\n")

        meal_result = self.agent3.generate(agent2_result)
        self._print_meal_plan(meal_result, t)

        return {
            "profile":     agent2_result["profile"],
            "targets":     t,
            "explanation": agent2_result["explanation"],
            "meal_plan":   meal_result
        }

    def _print_meal_plan(self, meal_result, targets):
        for meal_name, meal_data in meal_result["meals"].items():
            m = meal_data["macros"]
            print(f"{meal_name.capitalize()}  —  {m['calories']} kcal  |  P {m['protein']}g  F {m['fat']}g  C {m['carbs']}g")
            for item in meal_data["items"]:
                print(f"  {item['food_name']}  {item['grams']}g  ·  {item['calories']} kcal")
            print()

        tot = meal_result["total_macros"]
        print(f"Total   : {tot['calories']} kcal  |  P {tot['protein']}g  F {tot['fat']}g  C {tot['carbs']}g")
        print(f"Target  : {targets['daily_calories']} kcal  |  P {targets['macros']['protein_g']}g  F {targets['macros']['fat_g']}g  C {targets['macros']['carbs_g']}g")


def run_pipeline_from_profile(raw_input: dict):
    agent2_result = run_agent2(raw_input)
    if agent2_result["status"] != "success":
        print(agent2_result)
        return

    agent3 = MealPlanAgent(food_db)
    meal_result = agent3.generate(agent2_result)

    t   = agent2_result["targets"]
    tot = meal_result["total_macros"]

    print(f"Target  : {t['daily_calories']} kcal  |  P {t['macros']['protein_g']}g  F {t['macros']['fat_g']}g  C {t['macros']['carbs_g']}g")
    print(f"Achieved: {tot['calories']} kcal  |  P {tot['protein']}g  F {tot['fat']}g  C {tot['carbs']}g")
    print(f"\n{agent2_result['explanation']}\n")

    for meal_name, meal_data in meal_result["meals"].items():
        m = meal_data["macros"]
        print(f"{meal_name.capitalize()}  —  {m['calories']} kcal  |  P {m['protein']}g  F {m['fat']}g  C {m['carbs']}g")
        for item in meal_data["items"]:
            print(f"  {item['food_name']}  {item['grams']}g  ·  {item['calories']} kcal")
        print()

# Agent 5 InBody Extractor

def extract_inbody_from_image(image_path: str) -> dict:
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    ext  = image_path.split(".")[-1].lower()
    mime = "image/png" if ext == "png" else "image/jpeg"
    prompt = (
        "Extract all body composition data from this InBody scan.\n"
        "Return ONLY a raw JSON object — no markdown, no explanation, no code fences.\n"
        "Use exactly these keys (set to null if not visible):\n"
        '{"weight_kg":null,"skeletal_muscle_mass_kg":null,"body_fat_mass_kg":null,'
        '"body_fat_percent":null,"bmi":null,"bmr_kcal":null,"visceral_fat_level":null,'
        '"total_body_water_L":null,"height_cm":null}'
    )
    response = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
            {"type": "text",      "text": prompt}
        ]}],
        temperature=0
    )
    raw = re.sub(r"```json|```", "", response.choices[0].message.content or "").strip()
    m = re.search(r'\{[^{}]*\}', raw, re.DOTALL)
    if m: raw = m.group(0)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {}
    defaults = {
        "weight_kg": None, "skeletal_muscle_mass_kg": None, "body_fat_mass_kg": None,
        "body_fat_percent": None, "bmi": None, "bmr_kcal": None,
        "visceral_fat_level": None, "total_body_water_L": None, "height_cm": None
    }
    return {**defaults, **data}

def run_inbody_pipeline(image_path: str, age: int, sex: str,
                        activity_level: str, goal: str):
    print("Extracting InBody scan...")
    inbody = extract_inbody_from_image(image_path)
    print(f"  weight={inbody.get('weight_kg')}kg  height={inbody.get('height_cm')}cm  "
          f"body_fat={inbody.get('body_fat_percent')}%  BMR={inbody.get('bmr_kcal')} kcal")

    raw = {
        "age": age, "sex": sex,
        "weight_kg":      inbody.get("weight_kg")  or 70,
        "height_cm":      inbody.get("height_cm")  or 170,
        "activity_level": activity_level, "goal": goal,
    }

    agent2_result = run_agent2(raw)
    if agent2_result["status"] != "success":
        print(agent2_result)
        return

    t = agent2_result["targets"]
    print(f"Targets: {t['daily_calories']} kcal  |  P {t['macros']['protein_g']}g  F {t['macros']['fat_g']}g  C {t['macros']['carbs_g']}g")

    agent3 = MealPlanAgent(food_db)
    plan   = agent3.generate(agent2_result)
    tot    = plan["total_macros"]
    print(f"Achieved: {tot['calories']} kcal  |  P {tot['protein']}g  F {tot['fat']}g  C {tot['carbs']}g\n")

    for meal_name, meal_data in plan["meals"].items():
        m = meal_data["macros"]
        print(f"{meal_name.capitalize()}  —  {m['calories']} kcal")
        for item in meal_data["items"]:
            print(f"  {item['food_name']}  {item['grams']}g  ·  {item['calories']} kcal")
        print()

    return {"inbody": inbody, "targets": t, "plan": plan}


if __name__ == "__main__":
    import os
    inbody_path = "/content/drive/MyDrive/FoodData_Central_sr_legacy_food_csv_2018-04/ibody/inbodyscan.jpg"
    if os.path.exists(inbody_path):
        result = run_inbody_pipeline(inbody_path, age=25, sex="male",
                                     activity_level="active", goal="muscle_gain")
    else:
        print("inbodyscan.jpg not found — place it in the same folder to test")

    pipeline = NutritionPipeline()
    result   = pipeline.run()
