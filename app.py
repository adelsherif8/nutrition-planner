import re
import json
import base64
import requests
import pandas as pd
import gradio as gr
from groq import Groq
from dataclasses import dataclass, field
from typing import Optional

GROQ_API_KEY  = "<YOUR_GROQ_API_KEY>"
USDA_API_KEY  = "DEMO_KEY"
DATA_DIR      = "FoodData_Central_sr_legacy_food_csv_2018-04"
VISION_MODEL  = "meta-llama/llama-4-scout-17b-16e-instruct"
KEY_NUTRIENTS = {1003: "protein", 1004: "fat", 1005: "carbs", 1008: "calories"}
MEAL_SPLITS   = {"breakfast": 0.25, "lunch": 0.35, "dinner": 0.30, "snack": 0.10}

client = Groq(api_key=GROQ_API_KEY)

MODELS = {
    "M1 — Consistent": {"model": "llama-3.1-8b-instant", "temp": 0.0},
    "M2 — Balanced":   {"model": "llama-3.1-8b-instant", "temp": 0.4},
    "M3 — Creative":   {"model": "llama-3.1-8b-instant", "temp": 0.9},
}

VALID_SEXES           = {"male", "female"}
VALID_ACTIVITY_LEVELS = {"sedentary", "light", "moderate", "active"}
VALID_GOALS           = {"weight_loss", "maintenance", "muscle_gain"}
VALID_DIET_TYPES      = {"balanced", "vegetarian", "vegan", "keto", "low_carb"}

ACTIVITY_NORM_MAP = {
    "none":"sedentary","no exercise":"sedentary","desk job":"sedentary","inactive":"sedentary",
    "light":"light","walks":"light","walking":"light","1x week":"light","1-2x week":"light",
    "moderate":"moderate","gym 3x":"moderate","gym 3x/week":"moderate","3x week":"moderate",
    "3-4x week":"moderate","sometimes":"moderate","regular":"moderate",
    "active":"active","very active":"active","daily":"active","5x week":"active",
    "5-6x week":"active","athlete":"active","intense":"active",
}
GOAL_NORM_MAP = {
    "lose weight":"weight_loss","lose fat":"weight_loss","cut":"weight_loss","cutting":"weight_loss",
    "slim down":"weight_loss","fat loss":"weight_loss","shred":"weight_loss",
    "get shredded":"weight_loss","calorie deficit":"weight_loss",
    "maintain":"maintenance","maintain weight":"maintenance","stay same":"maintenance",
    "keep weight":"maintenance","healthy":"maintenance",
    "gain muscle":"muscle_gain","build muscle":"muscle_gain","bulk":"muscle_gain",
    "bulking":"muscle_gain","gain mass":"muscle_gain","get bigger":"muscle_gain",
    "hypertrophy":"muscle_gain",
}
DIET_NORM_MAP = {
    "balanced":"balanced","normal":"balanced","everything":"balanced","omnivore":"balanced",
    "vegetarian":"vegetarian","veg":"vegetarian","no meat":"vegetarian",
    "vegan":"vegan","plant based":"vegan","plant-based":"vegan",
    "keto":"keto","ketogenic":"keto",
    "low carb":"low_carb","low-carb":"low_carb","low carbohydrate":"low_carb",
}

@dataclass
class UserProfile:
    age: int; sex: str; height_cm: float; weight_kg: float
    activity_level: str; goal: str
    medical_conditions: list = field(default_factory=list)
    diet_type: str = "balanced"

def lbs_to_kg(lbs): return round(lbs * 0.453592, 1)
def ft_in_to_cm(ft, inches=0): return round(ft * 30.48 + inches * 2.54, 1)

def parse_imperial(text):
    t, result = text.lower(), {}
    m = re.search(r'(\d+(?:\.\d+)?)\s*(lbs?|pounds?)', t)
    if m: result["weight_kg"] = lbs_to_kg(float(m.group(1)))
    m = re.search(r"(\d+)\s*(?:ft|feet|')\s*(\d+)\s*(?:in|inches|\")?", t)
    if m: result["height_cm"] = ft_in_to_cm(int(m.group(1)), int(m.group(2)))
    else:
        m = re.search(r"(\d+)\s*(?:ft|feet|')", t)
        if m: result["height_cm"] = ft_in_to_cm(int(m.group(1)))
    return result

def extract_profile_from_text(text):
    t, p = text.lower(), {}
    imp = parse_imperial(text)
    if "weight_kg" in imp: p["weight_kg"] = imp["weight_kg"]
    if "height_cm" in imp: p["height_cm"] = imp["height_cm"]
    def find(pat):
        m = re.search(pat, t)
        return float(m.group(1)) if m else None
    age = find(r'(\d+)\s*(year|years|yo|y\.o)')
    if age: p["age"] = int(age)
    if not p.get("weight_kg"):
        w = find(r'(\d+(?:\.\d+)?)\s*(kg|kgs)')
        if w: p["weight_kg"] = float(w)
    if not p.get("height_cm"):
        h = find(r'(\d+(?:\.\d+)?)\s*(cm)')
        if h: p["height_cm"] = float(h)
    if any(w in t for w in ["male","man"," guy"]): p["sex"] = "male"
    elif any(w in t for w in ["female","woman","girl"]): p["sex"] = "female"
    if any(w in t for w in ["lose","fat","cut","slim","shred"]): p["goal"] = "weight_loss"
    elif any(w in t for w in ["gain","muscle","bulk","mass"]): p["goal"] = "muscle_gain"
    elif any(w in t for w in ["maintain","stay"]): p["goal"] = "maintenance"
    if any(w in t for w in ["sedentary","no exercise","inactive","desk"]): p["activity_level"] = "sedentary"
    elif any(w in t for w in ["light","walk","once a week"]): p["activity_level"] = "light"
    elif any(w in t for w in ["moderate","3x","sometimes"]): p["activity_level"] = "moderate"
    elif any(w in t for w in ["active","daily","intense","5x","athlete"]): p["activity_level"] = "active"
    return p

def _ns(v): return re.sub(r"\s+", " ", str(v).strip().lower())
def _norm(v, m): return m.get(_ns(v))

def normalize_activity(raw):
    c = _ns(raw)
    return (c, True) if c in VALID_ACTIVITY_LEVELS else (_norm(c, ACTIVITY_NORM_MAP), False)

def normalize_goal(raw):
    c = _ns(raw)
    return (c, True) if c in VALID_GOALS else (_norm(c, GOAL_NORM_MAP), False)

def normalize_diet(raw):
    c = _ns(raw)
    return (c, True) if c in VALID_DIET_TYPES else (_norm(c, DIET_NORM_MAP), False)

def validate_and_build_profile(raw):
    data = {"medical_conditions": [], "diet_type": "balanced", **raw}
    for k in ["age","sex","height_cm","weight_kg","activity_level","goal"]:
        if not data.get(k): return None, f"Missing: {k}"
    try: age = int(data["age"])
    except: return None, "Age must be a number"
    sex = _ns(str(data["sex"]))
    if sex not in VALID_SEXES: return None, "Sex must be male or female"
    try: height_cm = float(data["height_cm"])
    except: return None, "Height must be numeric"
    try: weight_kg = float(data["weight_kg"])
    except: return None, "Weight must be numeric"
    activity, _ = normalize_activity(str(data["activity_level"]))
    if activity is None:
        activity = _llm_normalize("activity_level", str(data["activity_level"]))
    goal, _ = normalize_goal(str(data["goal"]))
    if goal is None:
        goal = _llm_normalize("goal", str(data["goal"]))
    diet, _ = normalize_diet(str(data.get("diet_type","balanced")))
    if diet is None: diet = "balanced"
    bmi = weight_kg / ((height_cm/100)**2)
    if age < 18: return None, "User is underage"
    if bmi < 18.5: return None, "BMI too low (underweight)"
    if bmi > 35: return None, "BMI too high (severely obese)"
    return UserProfile(age=age, sex=sex, height_cm=height_cm, weight_kg=weight_kg,
                       activity_level=activity, goal=goal,
                       medical_conditions=data.get("medical_conditions",[]),
                       diet_type=diet), None

def _llm_normalize(field, raw):
    allowed = {"activity_level":"sedentary, light, moderate, active",
               "goal":"weight_loss, maintenance, muscle_gain"}
    prompt = f'Map "{raw}" to one of: {allowed[field]}. Return ONLY: {{"value":"..."}}'
    r = client.chat.completions.create(model="llama-3.1-8b-instant",
        messages=[{"role":"user","content":prompt}], temperature=0)
    try: return json.loads(re.sub(r"```json|```","",r.choices[0].message.content))["value"]
    except: return list(VALID_ACTIVITY_LEVELS)[0] if field=="activity_level" else "maintenance"

CALC_TOOLS = [
    {"type":"function","function":{"name":"calculate_bmr","description":"Calculate BMR",
     "parameters":{"type":"object","required":["weight_kg","height_cm","age","sex"],
     "properties":{"weight_kg":{"type":"number"},"height_cm":{"type":"number"},
                   "age":{"type":"integer"},"sex":{"type":"string","enum":["male","female"]}}}}},
    {"type":"function","function":{"name":"calculate_tdee","description":"Calculate TDEE",
     "parameters":{"type":"object","required":["bmr","activity_level"],
     "properties":{"bmr":{"type":"number"},
                   "activity_level":{"type":"string","enum":["sedentary","light","moderate","active"]}}}}},
    {"type":"function","function":{"name":"apply_goal","description":"Adjust calories for goal",
     "parameters":{"type":"object","required":["tdee","goal"],
     "properties":{"tdee":{"type":"number"},
                   "goal":{"type":"string","enum":["weight_loss","muscle_gain","maintenance"]}}}}},
    {"type":"function","function":{"name":"calculate_macros","description":"Calculate macros",
     "parameters":{"type":"object","required":["calories","weight_kg","goal"],
     "properties":{"calories":{"type":"number"},"weight_kg":{"type":"number"},
                   "goal":{"type":"string","enum":["weight_loss","muscle_gain","maintenance"]}}}}}
]
_FN = {
    "calculate_bmr":    lambda weight_kg,height_cm,age,sex: round(10*weight_kg+6.25*height_cm-5*age+(5 if sex=="male" else -161),1),
    "calculate_tdee":   lambda bmr,activity_level: round(bmr*{"sedentary":1.2,"light":1.375,"moderate":1.55,"active":1.725}[activity_level],1),
    "apply_goal":       lambda tdee,goal: round(max(tdee+{"weight_loss":-400,"muscle_gain":300,"maintenance":0}[goal],1000),1),
    "calculate_macros": lambda calories,weight_kg,goal: {"protein_g":round(weight_kg*(2.0 if goal=="muscle_gain" else 1.8 if goal=="weight_loss" else 1.4)),"fat_g":round(calories*0.25/9),"carbs_g":round((calories-weight_kg*(2.0 if goal=="muscle_gain" else 1.8 if goal=="weight_loss" else 1.4)*4-calories*0.25)/4)}
}

def _direct_calculate(profile: UserProfile) -> dict:
    bmr    = round(10*profile.weight_kg + 6.25*profile.height_cm - 5*profile.age + (5 if profile.sex=="male" else -161))
    tdee   = round(bmr * {"sedentary":1.2,"light":1.375,"moderate":1.55,"active":1.725}[profile.activity_level])
    cal    = round(max(tdee + {"weight_loss":-400,"muscle_gain":300,"maintenance":0}[profile.goal], 1200))
    prot   = round(profile.weight_kg * (2.0 if profile.goal=="muscle_gain" else 1.8 if profile.goal=="weight_loss" else 1.4))
    fat    = round(cal * 0.25 / 9)
    carbs  = round((cal - prot*4 - cal*0.25) / 4)
    return {"bmr":bmr,"tdee":tdee,"daily_calories":cal,"macros":{"protein_g":prot,"fat_g":fat,"carbs_g":carbs}}

def run_calculator(profile: UserProfile) -> dict:
    direct = _direct_calculate(profile)         
    try:
        messages = [{"role":"user","content":(
            f"Calculate targets step by step: calculate_bmr → calculate_tdee → apply_goal → calculate_macros\n"
            f"age={profile.age}, sex={profile.sex}, weight={profile.weight_kg}kg, "
            f"height={profile.height_cm}cm, activity={profile.activity_level}, goal={profile.goal}"
        )}]
        tr = {}
        for _ in range(6):
            r = client.chat.completions.create(model="llama-3.1-8b-instant",
                messages=messages, tools=CALC_TOOLS, tool_choice="auto")
            msg = r.choices[0].message
            if not msg.tool_calls: break
            messages.append(msg)
            for tc in msg.tool_calls:
                res = _FN[tc.function.name](**json.loads(tc.function.arguments))
                tr[tc.function.name] = res
                messages.append({"role":"tool","tool_call_id":tc.id,"content":json.dumps(res)})
        cal    = tr.get("apply_goal") or tr.get("calculate_tdee", 0)
        macros = tr.get("calculate_macros", {})
        if 1200 <= cal <= 5000 and macros.get("protein_g", 0) > 0:
            return {"bmr":tr.get("calculate_bmr", direct["bmr"]),
                    "tdee":tr.get("calculate_tdee", direct["tdee"]),
                    "daily_calories":round(cal), "macros":macros}
    except Exception:
        pass
    return direct 

def generate_explanation(profile: UserProfile, targets: dict) -> str:
    prompt = (f"Explain in 2 sentences (no medical advice):\n"
              f"{profile.age}yo {profile.sex}, {profile.weight_kg}kg, goal:{profile.goal}\n"
              f"Targets: {targets['daily_calories']} kcal, {targets['macros']}")
    r = client.chat.completions.create(model="llama-3.1-8b-instant",
        messages=[{"role":"user","content":prompt}], temperature=0.3)
    return r.choices[0].message.content.strip()

def build_food_db():
    foods = pd.read_csv(f"{DATA_DIR}/food.csv")
    fn    = pd.read_csv(f"{DATA_DIR}/food_nutrient.csv")
    fn["nutrient_id"] = fn["nutrient_id"].astype(int)
    filtered = fn[fn["nutrient_id"].isin(KEY_NUTRIENTS)].copy()
    filtered["nutrient_name"] = filtered["nutrient_id"].map(KEY_NUTRIENTS)
    pivoted = filtered.pivot_table(index="fdc_id",columns="nutrient_name",values="amount",aggfunc="first").reset_index()
    df = pivoted.merge(foods[["fdc_id","description"]], on="fdc_id")
    df = df.dropna(subset=["calories","protein","fat","carbs"])
    df = df[df["calories"]>0].reset_index(drop=True)
    df["food_name"] = df["description"].str.lower()
    return df[["food_name","calories","protein","carbs","fat"]]

def _usda(name):
    try:
        r = requests.get("https://api.nal.usda.gov/fdc/v1/foods/search",
            params={"query":name,"api_key":USDA_API_KEY,"pageSize":5,"dataType":"SR Legacy,Foundation"},timeout=6)
        foods = r.json().get("foods",[])
        if not foods: return None
        food = sorted(foods, key=lambda f:len(f.get("description","")))[0]
        nmap = {1003:"protein",1004:"fat",1005:"carbs",1008:"calories"}
        m = {nmap[n["nutrientId"]]:round(n.get("value",0),1) for n in food.get("foodNutrients",[]) if n.get("nutrientId") in nmap}
        return {**m,"food_name":food["description"].lower()}
    except: return None

def _off(name):
    try:
        r = requests.get("https://world.openfoodfacts.org/cgi/search.pl",
            params={"search_terms":name,"json":1,"page_size":5,"fields":"product_name,nutriments"},timeout=6)
        ps = [p for p in r.json().get("products",[]) if p.get("nutriments",{}).get("energy-kcal_100g")]
        if not ps: return None
        p = sorted(ps,key=lambda x:len(x.get("product_name","")))[0]
        n = p["nutriments"]
        return {"calories":round(n.get("energy-kcal_100g",0),1),"protein":round(n.get("proteins_100g",0),1),
                "fat":round(n.get("fat_100g",0),1),"carbs":round(n.get("carbohydrates_100g",0),1),
                "food_name":p.get("product_name",name).lower()}
    except: return None

JUNK = ["baby","candies","candy","instant","dry mix","formula","infant"]

class FoodDB:
    def __init__(self, df):
        self.df = df.dropna(subset=["calories","protein","carbs","fat"]).copy()

    def _best(self, matches):
        clean = matches[~matches["food_name"].str.contains("|".join(JUNK),na=False)]
        pool  = clean if not clean.empty else matches
        return pool.loc[pool["food_name"].str.len().idxmin()].to_dict()

    def get(self, name):
        name = name.lower().strip()
        m = self.df[self.df["food_name"].str.contains(name,na=False,regex=False)]
        if not m.empty: return self._best(m)
        for w in name.split():
            if len(w)<3: continue
            m = self.df[self.df["food_name"].str.contains(w,na=False,regex=False)]
            if not m.empty: return self._best(m)
        u = _usda(name)
        if u: return u
        o = _off(name)
        if o: return o
        return None

    def macros(self, name, grams):
        food = self.get(name)
        if not food: return None
        f = grams/100
        return {"food_name":food["food_name"],"grams":grams,
                "calories":round(food["calories"]*f,1),"protein":round(food["protein"]*f,1),
                "carbs":round(food["carbs"]*f,1),"fat":round(food["fat"]*f,1)}

    def meal(self, items):
        total, details = {"calories":0,"protein":0,"carbs":0,"fat":0}, []
        for name, grams in items:
            m = self.macros(name, grams)
            if not m: continue
            details.append(m)
            for k in total: total[k] += m[k]
        return {"items":details,"macros":{k:round(v,1) for k,v in total.items()}}

print("Loading food database...")
try:
    food_df = build_food_db()
    db = FoodDB(food_df)
    print(f"Food database ready — {len(food_df)} foods")
except Exception as e:
    print(f"Warning: Could not load local food DB ({e}), will use APIs only")
    db = None

def generate_meal_plan(targets: dict, profile: UserProfile, model_variant: str = "M1 — Consistent") -> dict:
    cfg   = MODELS.get(model_variant, MODELS["M1 — Consistent"])
    cal   = targets["daily_calories"]
    prot  = targets["macros"]["protein_g"]
    fat   = targets["macros"]["fat_g"]
    carbs = targets["macros"]["carbs_g"]
    splits_text = "\n".join(
        f"  {meal}: {round(cal*pct)} kcal, {round(prot*pct)}g protein, "
        f"{round(fat*pct)}g fat, {round(carbs*pct)}g carbs"
        for meal, pct in MEAL_SPLITS.items()
    )
    example = ('{"breakfast":[{"food":"oats","grams":80},{"food":"egg whites","grams":200}],'
               '"lunch":[{"food":"chicken breast","grams":150},{"food":"rice","grams":130}],'
               '"dinner":[{"food":"turkey breast","grams":150},{"food":"sweet potato","grams":150}],'
               '"snack":[{"food":"banana","grams":100},{"food":"greek yogurt","grams":150}]}')
    prompt = (
        f"You are a strict nutrition planner. Create a daily meal plan.\n\n"
        f"User: {profile.sex}, {profile.weight_kg}kg, goal:{profile.goal}, diet:{profile.diet_type}\n"
        f"Daily: {cal} kcal | {prot}g protein | {fat}g fat | {carbs}g carbs\n\n"
        f"Per meal:\n{splits_text}\n\n"
        f"Rules:\n- ONLY valid JSON, no markdown\n"
        f"- Use lean whole foods: chicken breast, turkey, egg whites, oats, rice, potato, banana, greek yogurt\n"
        f"- AVOID: peanut butter, nuts, oils, cheese, candy, baby food, processed snacks\n"
        f"- Keep grams realistic (max 300g per item)\n- Each meal: 2-3 foods\n\nFormat:\n{example}"
    )
    meal_json = None
    for attempt in range(3):
        r = client.chat.completions.create(
            model=cfg["model"],
            messages=[{"role":"user","content":prompt}],
            temperature=cfg["temp"]
        )
        raw_content = r.choices[0].message.content or ""
        content = re.sub(r"```json|```", "", raw_content).strip()
        m = re.search(r'\{.*\}', content, re.DOTALL)
        if m:
            content = m.group(0)
        try:
            meal_json = json.loads(content)
            break
        except json.JSONDecodeError:
            if attempt == 2:
                raise ValueError("Meal planner returned invalid JSON after 3 attempts.")
    if meal_json is None:
        raise ValueError("Meal planner returned no output.")

    verified = {}
    if db:
        for meal_name, items in meal_json.items():
            verified[meal_name] = db.meal([(i["food"],i["grams"]) for i in items if isinstance(i, dict) and "food" in i and "grams" in i])
        scaled = {}
        for mn, md in verified.items():
            meal_target = cal * MEAL_SPLITS.get(mn, 0.25)
            meal_achieved = md["macros"]["calories"]
            scale = meal_target / meal_achieved if meal_achieved > 0 else 1
            ni = []
            for item in md["items"]:
                rc = db.macros(item["food_name"], round(item["grams"] * scale))
                if rc: ni.append(rc)
            total = {k:round(sum(i[k] for i in ni),1) for k in ["calories","protein","carbs","fat"]}
            scaled[mn] = {"items":ni,"macros":total}
        verified = scaled
    else:
        for meal_name, items in meal_json.items():
            verified[meal_name] = {"items":[{"food_name":i["food"],"grams":i["grams"],
                "calories":0,"protein":0,"carbs":0,"fat":0} for i in items],"macros":{"calories":0,"protein":0,"carbs":0,"fat":0}}

    total = {k:round(sum(m["macros"][k] for m in verified.values()),1) for k in ["calories","protein","carbs","fat"]}
    return {"meals":verified,"total_macros":total}

def parse_dislikes(feedback: str) -> list:
    prompt = f'Extract disliked foods from: "{feedback}"\nReturn ONLY JSON array: ["food1","food2"]'
    r = client.chat.completions.create(model="llama-3.1-8b-instant",
        messages=[{"role":"user","content":prompt}], temperature=0)
    try: return json.loads(re.sub(r"```json|```","",r.choices[0].message.content).strip())
    except: return []

def apply_substitutions(meal_plan: dict, feedback: str) -> tuple:
    dislikes = parse_dislikes(feedback)
    if not dislikes: return meal_plan, []
    changes = []
    for mn, md in meal_plan["meals"].items():
        new_items = []
        for item in md["items"]:
            fn = item["food_name"].lower()
            hated = any(d.lower() in fn or fn in d.lower() for d in dislikes)
            if hated and db:
                prompt = (f'Suggest 5 substitutes for "{item["food_name"]}" NOT in {dislikes}. '
                          f'Use simple foods.\nReturn ONLY JSON array: ["food1","food2","food3","food4","food5"]')
                r = client.chat.completions.create(model="llama-3.1-8b-instant",
                    messages=[{"role":"user","content":prompt}], temperature=0.3)
                try:
                    alts = json.loads(re.sub(r"```json|```","",r.choices[0].message.content).strip())
                    found = False
                    for alt in alts:
                        if any(d.lower() in alt.lower() for d in dislikes): continue
                        rc = db.macros(alt, item["grams"])
                        if rc:
                            target_cal = item["calories"]
                            if rc["calories"] > 0 and target_cal > 0:
                                scale = target_cal / rc["calories"]
                                rc = db.macros(rc["food_name"], round(item["grams"] * scale))
                            if rc:
                                changes.append(f"**{item['food_name']}** → **{rc['food_name']}**")
                                new_items.append(rc)
                                found = True
                                break
                    if not found: new_items.append(item)
                except: new_items.append(item)
            else:
                new_items.append(item)
        total = {k:round(sum(i[k] for i in new_items),1) for k in ["calories","protein","carbs","fat"]}
        meal_plan["meals"][mn] = {"items":new_items,"macros":total}
    meal_plan["total_macros"] = {k:round(sum(m["macros"][k] for m in meal_plan["meals"].values()),1)
                                  for k in ["calories","protein","carbs","fat"]}
    return meal_plan, changes

def extract_inbody(image_path: str) -> dict:
    defaults = {"weight_kg":None,"skeletal_muscle_mass_kg":None,"body_fat_mass_kg":None,
                "body_fat_percent":None,"bmi":None,"bmr_kcal":None,
                "visceral_fat_level":None,"total_body_water_L":None,"height_cm":None}
    try:
        with open(image_path,"rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        ext  = image_path.split(".")[-1].lower()
        mime = "image/png" if ext=="png" else "image/jpeg"
        prompt = ("Extract all body composition data from this InBody scan.\n"
                  "Return ONLY a raw JSON object — no markdown, no explanation, no code fences.\n"
                  "Use exactly these keys (null if not visible):\n"
                  '{"weight_kg":null,"skeletal_muscle_mass_kg":null,"body_fat_mass_kg":null,'
                  '"body_fat_percent":null,"bmi":null,"bmr_kcal":null,"visceral_fat_level":null,'
                  '"total_body_water_L":null,"height_cm":null}')
        r = client.chat.completions.create(model=VISION_MODEL,
            messages=[{"role":"user","content":[
                {"type":"image_url","image_url":{"url":f"data:{mime};base64,{b64}"}},
                {"type":"text","text":prompt}
            ]}], temperature=0)
        raw = re.sub(r"```json|```","", r.choices[0].message.content or "").strip()
        m = re.search(r'\{.*?\}', raw, re.DOTALL)
        if m:
            raw = m.group(0)
        if raw:
            parsed = json.loads(raw)
            return {**defaults, **{k:v for k,v in parsed.items() if k in defaults}}
    except Exception:
        pass
    return defaults

def format_plan(plan: dict, targets: dict, explanation: str) -> str:
    t   = plan["total_macros"]
    tgt = targets["macros"]
    cal_err = round(abs(t["calories"]-targets["daily_calories"])/targets["daily_calories"]*100,1)

    lines = [
        "## Daily Targets",
        f"| Calories | Protein | Fat | Carbs |",
        f"|----------|---------|-----|-------|",
        f"| **{targets['daily_calories']} kcal** | **{tgt['protein_g']}g** | **{tgt['fat_g']}g** | **{tgt['carbs_g']}g** |",
        "",
        explanation,
        "",
        "---",
        "## Meal Plan",
        ""
    ]

    for meal_name, meal_data in plan["meals"].items():
        m = meal_data["macros"]
        lines.append(f"### {meal_name.capitalize()}")
        lines.append(f"*{m['calories']} kcal | P:{m['protein']}g  F:{m['fat']}g  C:{m['carbs']}g*")
        lines.append("")
        for item in meal_data["items"]:
            lines.append(f"- **{item['food_name'].title()}** — {item['grams']}g "
                         f"· {item['calories']} kcal · P:{item['protein']}g")
        lines.append("")

    lines += [
        "---",
        "## Day Total",
        f"| | Calories | Protein | Fat | Carbs |",
        f"|--|----------|---------|-----|-------|",
        f"| **Achieved** | {t['calories']} | {t['protein']}g | {t['fat']}g | {t['carbs']}g |",
        f"| **Target**   | {targets['daily_calories']} | {tgt['protein_g']}g | {tgt['fat_g']}g | {tgt['carbs_g']}g |",
        f"| **Error**    | {cal_err}% | — | — | — |",
    ]
    return "\n".join(lines)

def run_pipeline(raw_input: dict, model_variant: str = "M1 — Consistent"):
    profile, err = validate_and_build_profile(raw_input)
    if err: return None, None, None, f"❌ {err}"
    targets = run_calculator(profile)
    plan    = generate_meal_plan(targets, profile, model_variant)
    return profile, targets, plan, None

TEST_TEXT = "I'm a 25 year old male, 180 lbs, 5'11, I go to the gym 5x a week and want to gain muscle"
TEST_FORM = {"age":25,"sex":"male","height_cm":174,"weight_kg":71,
             "activity_level":"active","goal":"muscle_gain","diet_type":"balanced"}

FA_HEAD = ""

CSS = """
:root {
    --body-background-fill:               #ffffff;
    --block-background-fill:              #ffffff;
    --block-border-color:                 #e5e5e5;
    --block-label-background-fill:        #ffffff;
    --block-title-background-fill:        transparent;
    --block-info-text-color:              #888888;
    --block-label-text-color:             #888888;
    --block-title-text-color:             #111111;
    --background-fill-primary:            #ffffff;
    --background-fill-secondary:          #f5f5f5;
    --border-color-primary:               #e5e5e5;
    --border-color-accent:                #e5e5e5;
    --input-background-fill:              #ffffff;
    --input-border-color:                 #dddddd;
    --input-shadow:                       none;
    --button-primary-background-fill:     #000000;
    --button-primary-background-fill-hover:#333333;
    --button-primary-text-color:          #ffffff;
    --button-secondary-background-fill:   #ffffff;
    --button-secondary-background-fill-hover:#f5f5f5;
    --button-secondary-border-color:      #cccccc;
    --button-secondary-text-color:        #333333;
    --color-accent:                       #000000;
    --body-text-color:                    #111111;
    --body-text-color-subdued:            #888888;
    --panel-background-fill:              #ffffff;
    --section-header-text-color:          #888888;
    --neutral-100:                        #f5f5f5;
    --neutral-200:                        #e5e5e5;
    --neutral-800:                        #ffffff;
    --neutral-900:                        #ffffff;
    --neutral-950:                        #ffffff;
}

* { box-sizing: border-box; }

body, .gradio-container {
    background: #fff !important;
    color: #111 !important;
    max-width: 960px !important;
    margin: 0 auto !important;
    padding: 1.5rem !important;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}
footer, .built-with { display: none !important; }

.tab-nav {
    background: #f0f0f0 !important;
    border-radius: 10px !important;
    padding: 4px !important;
    border: 1px solid #e0e0e0 !important;
    margin-bottom: 1rem !important;
}
.tab-nav button {
    border-radius: 7px !important;
    padding: 8px 18px !important;
    font-weight: 500 !important;
    font-size: 0.875rem !important;
    color: #888 !important;
    border: none !important;
    background: transparent !important;
}
.tab-nav button.selected {
    background: #000 !important;
    color: #fff !important;
    font-weight: 600 !important;
}

#plan-output table { width: 100% !important; border-collapse: collapse !important; margin: 0.6rem 0 !important; }
#plan-output th { background: #f5f5f5 !important; padding: 10px 16px !important; font-size: 0.75rem !important; font-weight: 700 !important; text-transform: uppercase !important; letter-spacing: 0.5px !important; color: #444 !important; border-bottom: 1px solid #e5e5e5 !important; text-align: center !important; }
#plan-output td { padding: 9px 16px !important; text-align: center !important; border-bottom: 1px solid #f0f0f0 !important; color: #222 !important; font-size: 0.88rem !important; }
#plan-output tr:last-child td { border-bottom: none !important; }
#plan-output h2, #plan-output h3 { color: #111 !important; margin-top: 1.4rem !important; }

.block.svelte-12cmxck { background: white !important; }
span.svelte-1gfkn6j { color: black !important; }
label.svelte-1mhtq7j { background: black !important; }
input.svelte-1mhtq7j { background: white !important; }
input.svelte-1mhtq7j.svelte-1mhtq7j.svelte-1mhtq7j:checked { background: grey !important; }
input.svelte-1f354aw.svelte-1f354aw, textarea.svelte-1f354aw.svelte-1f354aw { background: white !important; border: solid black 1px !important; }
.primary.svelte-cmf5ev { background: black !important; }
.secondary.svelte-cmf5ev { background: white !important; color: black !important; border: 1px solid black !important; }

input[type=number].svelte-pjtc3.svelte-pjtc3 { background: white !important; color: black !important; border: solid 1px !important; }
input[type=number].svelte-pjtc3.svelte-pjtc3, .wrap-inner.svelte-1sk0pyu.svelte-1sk0pyu { border-radius: 10px; background: white !important; color: black !important; border: solid 1px !important; }
.wrap.svelte-1sk0pyu.svelte-1sk0pyu { background: white !important; }
.wrap-inner.svelte-1sk0pyu.svelte-1sk0pyu input { color: black !important; }
.wrap.svelte-12ioyct { color: black !important; }
p { color: black !important; }
strong { color: black !important; }
li, ul, em { color: black !important; }
textarea.scroll-hide.svelte-1f354aw {
    color: black;
}
"""

with gr.Blocks(
    theme=gr.themes.Base(),
    title="Nutrition Planner",
    css=CSS
) as app:

    plan_state    = gr.State({})
    targets_state = gr.State({})
    profile_state = gr.State({})
    raw_state     = gr.State({})

    gr.HTML("""
    <div style="background:#fff;border:1px solid #e5e5e5;border-radius:12px;
                padding:1.5rem 2rem;margin-bottom:1rem;text-align:center;">
      <h1 style="font-size:1.6rem;font-weight:700;color:#111;margin:0 0 0.25rem;">Nutrition Planner</h1>
      <p style="color:#888;font-size:0.83rem;margin:0;">Enter your details and get a personalised daily meal plan.</p>
    </div>
    """)

    model_selector = gr.Radio(
        choices=list(MODELS.keys()),
        value="M1 — Consistent",
        label="Model",
        elem_classes="model-radio",
    )

    with gr.Tabs():

        with gr.Tab("Describe Yourself"):
            gr.HTML('<p style="color:#888;font-size:0.85rem;margin:0.5rem 0 0.9rem;">'
                    'Write naturally — your age, weight, height, activity, and goal. Supports lbs and ft/in.</p>')
            text_input = gr.Textbox(
                label="Your details",
                placeholder="e.g. I'm a 25 year old male, 180 lbs, 5'11, I train 5x a week and want to build muscle",
                lines=3)
            with gr.Row():
                text_btn      = gr.Button("Generate Plan", variant="primary")
                text_test_btn = gr.Button("Load Sample", variant="secondary")

        with gr.Tab("InBody Scan"):
            gr.HTML('<p style="color:#888;font-size:0.85rem;margin:0.5rem 0 0.9rem;">'
                    'Upload your InBody report — weight and body composition are read automatically.</p>')
            with gr.Row():
                inbody_img = gr.Image(label="InBody Report (JPG / PNG)", type="filepath")
                with gr.Column():
                    gr.HTML('<p style="font-size:0.72rem;color:#888;'
                            'text-transform:uppercase;letter-spacing:0.5px;margin:0 0 0.5rem;">Additional info</p>')
                    inbody_age      = gr.Number(label="Age", value=25)
                    inbody_sex      = gr.Dropdown(["male","female"], label="Sex", value="male")
                    inbody_activity = gr.Dropdown(["sedentary","light","moderate","active"],
                                                  label="Activity Level", value="moderate")
                    inbody_goal     = gr.Dropdown(["weight_loss","maintenance","muscle_gain"],
                                                  label="Goal", value="muscle_gain")
            with gr.Row():
                inbody_btn      = gr.Button("Extract & Generate Plan", variant="primary")
                inbody_test_btn = gr.Button("Load Sample", variant="secondary")
            inbody_extracted = gr.JSON(label="Extracted Metrics", visible=False)

        with gr.Tab("Manual Entry"):
            gr.HTML('<p style="color:#888;font-size:0.85rem;margin:0.5rem 0 0.9rem;">Fill in your details below.</p>')
            with gr.Row():
                form_age    = gr.Number(label="Age", value=25)
                form_sex    = gr.Dropdown(["male","female"], label="Sex", value="male")
                form_weight = gr.Number(label="Weight (kg)", value=70)
                form_height = gr.Number(label="Height (cm)", value=175)
            with gr.Row():
                form_activity = gr.Dropdown(["sedentary","light","moderate","active"],
                                            label="Activity Level", value="moderate")
                form_goal     = gr.Dropdown(["weight_loss","maintenance","muscle_gain"],
                                            label="Goal", value="maintenance")
                form_diet     = gr.Dropdown(["balanced","vegetarian","vegan","keto","low_carb"],
                                            label="Diet Type", value="balanced")
            with gr.Row():
                form_btn      = gr.Button("Generate Plan", variant="primary")
                form_test_btn = gr.Button("Load Sample", variant="secondary")

    status_box  = gr.Markdown("", elem_id="status-box")
    plan_output = gr.Markdown("", elem_id="plan-output")

    with gr.Row():
        feedback_input = gr.Textbox(
            label="Swap Foods",
            info="Tell us what you want to avoid — we will find an equivalent replacement.",
            placeholder="e.g. no eggs   /   I hate salmon   /   remove broccoli",
            scale=4)
        feedback_btn = gr.Button("Apply", variant="secondary", scale=1)
    changes_output = gr.Markdown("", elem_id="changes-out")

    def _render(profile, targets, plan):
        explanation = generate_explanation(profile, targets)
        md = format_plan(plan, targets, explanation)
        return md, plan, targets, profile

    def from_text(text, model_variant):
        if not text.strip(): return "Please enter a description.", "", {}, {}, {}, {}
        p = extract_profile_from_text(text)
        missing = [k for k in ["age","sex","weight_kg","height_cm","activity_level","goal"] if not p.get(k)]
        if missing: return f"Could not extract: {', '.join(missing)}. Please add more detail.", "", {}, {}, {}, {}
        profile, targets, plan, err = run_pipeline(p, model_variant)
        if err: return err, "", {}, {}, {}, {}
        md, plan, targets, profile = _render(profile, targets, plan)
        return f"Plan ready — {model_variant}.", md, plan, targets, profile.__dict__, p

    def from_form(age, sex, weight, height, activity, goal, diet, model_variant):
        raw = {"age":age,"sex":sex,"weight_kg":weight,"height_cm":height,
               "activity_level":activity,"goal":goal,"diet_type":diet}
        profile, targets, plan, err = run_pipeline(raw, model_variant)
        if err: return err, "", {}, {}, {}, {}
        md, plan, targets, profile = _render(profile, targets, plan)
        return f"Plan ready — {model_variant}.", md, plan, targets, profile.__dict__, raw

    def from_inbody(img_path, age, sex, activity, goal, model_variant):
        if img_path is None:
            return "Please upload an InBody report.", "", gr.update(visible=False), {}, {}, {}, "", {}
        inbody = extract_inbody(img_path)
        note = ""
        if inbody.get("weight_kg") is None and inbody.get("height_cm") is None:
            note = "\n\n*Scan values could not be read — using defaults (70 kg / 170 cm).*"
        raw = {"age":age,"sex":sex,
               "weight_kg": inbody.get("weight_kg") or 70,
               "height_cm": inbody.get("height_cm") or 170,
               "activity_level":activity,"goal":goal}
        profile, targets, plan, err = run_pipeline(raw, model_variant)
        if err: return err, "", gr.update(visible=True, value=inbody), {}, {}, {}, "", {}
        md, plan, targets, profile = _render(profile, targets, plan)
        return (f"Plan ready — {model_variant}." + note, md,
                gr.update(visible=True, value=inbody), plan, targets, profile.__dict__, "", raw)

    def apply_feedback(feedback, plan, targets, profile_dict):
        if not feedback.strip() or not plan: return "", plan, ""
        updated_plan, changes = apply_substitutions(plan, feedback)
        explanation = generate_explanation(
            UserProfile(**{k:v for k,v in profile_dict.items() if k in UserProfile.__dataclass_fields__}),
            targets)
        md = format_plan(updated_plan, targets, explanation)
        change_md = ("**Substitutions applied:**\n" + "\n".join(f"- {c}" for c in changes)
                     if changes else "No matching foods found in the plan.")
        return md, updated_plan, change_md

    text_btn.click(from_text, [text_input, model_selector],
        [status_box, plan_output, plan_state, targets_state, profile_state, raw_state], api_name=False)
    form_btn.click(from_form, [form_age,form_sex,form_weight,form_height,form_activity,form_goal,form_diet,model_selector],
        [status_box, plan_output, plan_state, targets_state, profile_state, raw_state], api_name=False)
    inbody_btn.click(from_inbody, [inbody_img,inbody_age,inbody_sex,inbody_activity,inbody_goal,model_selector],
        [status_box, plan_output, inbody_extracted, plan_state, targets_state, profile_state, changes_output, raw_state], api_name=False)
    feedback_btn.click(apply_feedback, [feedback_input, plan_state, targets_state, profile_state],
        [plan_output, plan_state, changes_output], api_name=False)

    text_test_btn.click(lambda: TEST_TEXT, [], [text_input], api_name=False)
    inbody_test_btn.click(
        lambda: ("inbodyscan.jpg", 25, "male", "active", "muscle_gain"),
        [], [inbody_img, inbody_age, inbody_sex, inbody_activity, inbody_goal], api_name=False)
    form_test_btn.click(
        lambda: (TEST_FORM["age"], TEST_FORM["sex"], TEST_FORM["weight_kg"],
                 TEST_FORM["height_cm"], TEST_FORM["activity_level"],
                 TEST_FORM["goal"], TEST_FORM["diet_type"]),
        [], [form_age, form_sex, form_weight, form_height, form_activity, form_goal, form_diet], api_name=False)

if __name__ == "__main__":
    app.launch(share=True)
