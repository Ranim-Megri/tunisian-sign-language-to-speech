import re
import unicodedata
from typing import List, Dict

COMMON_DERJA_DICT: Dict[str, str] = {
    "3aslema": "عسلامة",
    "labes": "لاباس",
    "oui": "إيه",
    "non": "لا",
    "sbitar": "سبيطار",
    "lyoum": "اليوم",
    "banka": "بنكة",
    "baladya": "بلدية",
    "dar": "دار",
    "taxi": "تاكسي",
    "louage": "لواج",
    "jom3a": "الجمعة",
    "om": "ماما",
    "mama": "ماما",
    "n7eb": "نحب",
    "nemchi": "نمشي",
    "lel": "لـ",
    "bel": "بال",
    "narja3": "نرجع",
    "chnowa": "شنوة",
    "7alek": "حالك",
    "kifech": "كيفاش",
    "nejem": "نجم",
    "n3awnek": "نعاونك",
    "el": "ال",
    "7amdoullah": "الحمد لله",
    "3andi": "عندي",
    "tlatha": "ثلاثة",
    "sghar": "صغار",
    "wa7ed": "واحد",
    "karhba": "كرهبة",
    "a3tini": "أعطيني",
    "5obz": "خبز",
    "7lib": "حليب",
    "yar7am": "يرحم",
    "waldik": "والديك",
    "win": "وين",
    "gar": "غار",
    "mta3": "متاع",
    "tran": "تران",
    "wa9tech": "وقتيش",
    "yo5rej": "يخرج",
    "sfax": "صفاقس",
    "kallamni": "كلمني",
    "3al": "على",
    "telephone": "تليفون",
    "ta9es": "طقس",
    "s5oun": "سخون",
    "barcha": "برشا",
    "sa7ra": "صحراء",
    "chouf": "شوف",
    "toubib": "طبيب",
    "9bal": "قبل",
    "ma": "ما",
    "temchi": "تمشي",
    "9adech": "قداش",
    "soum": "سوم",
    "kilo": "كيلو",
    "toufa7": "تفاح",
    "rendez": "موعد",
    "vous": "",
    "3omda": "عمدة",
    "lezem": "لازم",
    "nbeddel": "نبدل",
    "passeport": "باسبور",
    "ya": "يا",
    "7ajja": "حاجة",
    "tawwa": "توا",
    "ena": "أنا",
    "3amal": "خدمة",
    "bekri": "بكري"
}

ARABIZI_TO_ARABIC: Dict[str, str] = {
    "3": "ع",
    "7": "ح",
    "9": "ق",
    "5": "خ",
    "8": "غ",
    "2": "ء",
    "th": "ث",
    "ch": "ش",
    "kh": "خ",
    "dh": "ذ",
    "ou": "و",
    "ee": "ي",
    "aa": "ا",
    "a": "ا",
    "b": "ب",
    "t": "ت",
    "d": "د",
    "r": "ر",
    "z": "ز",
    "s": "س",
    "f": "ف",
    "k": "ك",
    "l": "ل",
    "m": "م",
    "n": "ن",
    "h": "ه",
    "w": "و",
    "y": "ي"
}

VOCABULARY = [
    "_", " ", "!", "?", ".", ",", "-",
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
    "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
    "2", "3", "5", "7", "8", "9",
    "ء", "آ", "أ", "ؤ", "إ", "ئ", "ا", "ب", "ة", "ت", "ث", "ج", "ح", "خ",
    "د", "ذ", "ر", "ز", "س", "ش", "ص", "ض", "ط", "ظ", "ع", "غ", "ف", "ق",
    "ك", "ل", "م", "ن", "ه", "و", "ى", "ي", "ً", "ٌ", "ٍ", "َ", "ُ", "ِ", "ّ", "ْ"
]

CHAR_TO_ID = {char: idx for idx, char in enumerate(VOCABULARY)}
ID_TO_CHAR = {idx: char for idx, char in enumerate(VOCABULARY)}


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_derja(text: str) -> str:
    cleaned = clean_text(text)
    if not cleaned:
        return ""
    lowered = cleaned.lower()
    lowered = re.sub(r"[^\w\s\d\.\,\!\?\-\u0600-\u06FF]", "", lowered)
    return lowered.strip()


def is_arabic_script(text: str) -> bool:
    arabic_chars = re.findall(r"[\u0600-\u06FF]", text)
    return len(arabic_chars) > (len(text) / 4)


def arabizi_to_derja_arabic(text: str) -> str:
    cleaned = clean_text(text)
    if is_arabic_script(cleaned):
        return cleaned

    words = re.findall(r"[\w\d]+|[^\w\s]", cleaned.lower())
    translated_words = []
    for w in words:
        if w in COMMON_DERJA_DICT:
            translated_words.append(COMMON_DERJA_DICT[w])
        elif re.match(r"[^\w\s]", w):
            translated_words.append(w)
        else:
            w_mapped = w
            for num, ar in [("3", "ع"), ("7", "ح"), ("9", "ق"), ("5", "خ"), ("8", "غ"), ("2", "ء")]:
                w_mapped = w_mapped.replace(num, ar)
            translated_words.append(w_mapped)

    res = " ".join([x for x in translated_words if x])
    res = re.sub(r"\s+([?.!,])", r"\1", res)
    return res


def text_to_token_ids(text: str) -> List[int]:
    normalized = normalize_derja(text)
    token_ids = []
    for char in normalized:
        if char in CHAR_TO_ID:
            token_ids.append(CHAR_TO_ID[char])
        else:
            token_ids.append(CHAR_TO_ID["_"])
    return token_ids


def token_ids_to_text(ids: List[int]) -> str:
    chars = [ID_TO_CHAR.get(idx, "") for idx in ids]
    return "".join(chars)
