import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import joblib, os, re, torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

# =============================================================
#  MODEL DEFINITIONS — must match train_rnn.py / train_fusion.py
# =============================================================
class FashionLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm      = nn.LSTM(embed_dim, hidden_dim,
                                 num_layers=2, batch_first=True, dropout=0.3)
        self.dropout   = nn.Dropout(0.3)
        self.fc        = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        emb            = self.embedding(x)
        _, (hidden, _) = self.lstm(emb)
        features       = self.dropout(hidden[-1])   # (B, 512)
        return self.fc(features), features


class JointProjector(nn.Module):
    def __init__(self, img_dim=2048, txt_dim=512, joint_dim=256):
        super().__init__()
        self.img_proj = nn.Sequential(
            nn.Linear(img_dim, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(512, joint_dim))
        self.txt_proj = nn.Sequential(
            nn.Linear(txt_dim, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(512, joint_dim))

    def forward_img(self, x):
        return F.normalize(self.img_proj(x), dim=-1)

    def forward_txt(self, x):
        return F.normalize(self.txt_proj(x), dim=-1)

    def forward(self, img, txt):
        return F.normalize((self.forward_img(img) + self.forward_txt(txt)) / 2, dim=-1)

# =============================================================
#  PAGE CONFIG
# =============================================================
st.set_page_config(page_title="StyleSense AI — Fashion Intelligence",
                   page_icon="✦", layout="wide",
                   initial_sidebar_state="expanded")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,700;0,900;1,400&family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;500&display=swap');

/* ── CSS VARIABLES (exact from HTML) ── */
:root {
  --bg:      #0a0a08;
  --surface: #111110;
  --surface2:#1a1a18;
  --border:  #2a2a28;
  --accent:  #c8a96e;
  --accent2: #e8c87e;
  --text:    #f0ede8;
  --text2:   #9a9890;
  --text3:   #6a6860;
  --red:     #e05a4a;
  --green:   #6ab87a;
  --blue:    #5a8ae0;
  --purple:  #9a7ae0;
}

/* ── GLOBAL ── */
html, body, [class*="css"] {
  font-family: 'DM Sans', sans-serif !important;
  background: var(--bg) !important;
  color: var(--text) !important;
}

/* Grid noise background — exact from HTML */
[data-testid="stAppViewContainer"]::before {
  content: '';
  position: fixed; inset: 0; z-index: 0; pointer-events: none;
  background-image:
    linear-gradient(rgba(200,169,110,0.03) 1px, transparent 1px),
    linear-gradient(90deg, rgba(200,169,110,0.03) 1px, transparent 1px);
  background-size: 40px 40px;
}

[data-testid="stAppViewContainer"] {
  background: var(--bg) !important;
}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
  background: var(--surface) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * {
  font-family: 'DM Sans', sans-serif !important;
  color: var(--text2) !important;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] .stCheckbox label,
[data-testid="stSidebar"] .stMarkdown p {
  font-family: 'DM Mono', monospace !important;
  font-size: 10px !important;
  letter-spacing: 3px !important;
  text-transform: uppercase !important;
  color: var(--text3) !important;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label,
[data-testid="stSidebar"] .stCheckbox label span {
  font-family: 'DM Sans', sans-serif !important;
  font-size: 13px !important;
  text-transform: none !important;
  letter-spacing: 0 !important;
  color: var(--text2) !important;
}
[data-testid="stSidebar"] .stSelectbox > div > div {
  background: var(--surface2) !important;
  border: 1px solid var(--border) !important;
  border-radius: 3px !important;
  color: var(--text) !important;
}

/* ── PROGRESS BAR ── */
.stProgress > div > div > div {
  background: linear-gradient(90deg, var(--accent), var(--accent2)) !important;
  border-radius: 2px !important;
}
.stProgress > div > div {
  background: var(--surface2) !important;
  border-radius: 2px !important;
}

/* ── TABS — exact DM Mono monospace styling ── */
.stTabs [data-baseweb="tab-list"] {
  background: var(--surface) !important;
  border-radius: 3px !important;
  padding: 4px !important;
  gap: 2px !important;
  border: 1px solid var(--border) !important;
}
.stTabs [data-baseweb="tab"] {
  background: transparent !important;
  color: var(--text3) !important;
  border-radius: 2px !important;
  font-family: 'DM Mono', monospace !important;
  font-size: 10px !important;
  letter-spacing: 1.5px !important;
  text-transform: uppercase !important;
  padding: 7px 14px !important;
  border: none !important;
}
.stTabs [aria-selected="true"] {
  background: var(--surface2) !important;
  color: var(--accent) !important;
}
.stTabs [data-baseweb="tab-panel"] { padding-top: 16px !important; }

/* ── PRIMARY BUTTON — accent gold, dark text ── */
.stButton > button[kind="primary"] {
  background: var(--accent) !important;
  color: var(--bg) !important;
  font-family: 'DM Sans', sans-serif !important;
  font-weight: 500 !important;
  font-size: 14px !important;
  letter-spacing: 0.3px !important;
  border: none !important;
  border-radius: 3px !important;
  padding: 14px 24px !important;
  transition: all 0.2s !important;
}
.stButton > button[kind="primary"]:hover {
  background: var(--accent2) !important;
  transform: translateY(-1px) !important;
}
/* ── SECONDARY BUTTON ── */
.stButton > button:not([kind="primary"]) {
  background: var(--surface) !important;
  color: var(--text2) !important;
  border: 1px solid var(--border) !important;
  border-radius: 3px !important;
  font-family: 'DM Mono', monospace !important;
  font-size: 11px !important;
  letter-spacing: 1px !important;
}

/* ── TEXT INPUT ── */
.stTextInput > div > div {
  background: var(--surface) !important;
  border: 1.5px solid var(--border) !important;
  border-radius: 3px !important;
  color: var(--text) !important;
  font-family: 'DM Sans', sans-serif !important;
  font-size: 14px !important;
}
.stTextInput > div > div:focus-within {
  border-color: var(--accent) !important;
  box-shadow: none !important;
}
.stTextInput input::placeholder { color: var(--text3) !important; }

/* ── FILE UPLOADER ── */
[data-testid="stFileUploader"] {
  background: var(--surface) !important;
  border: 1.5px dashed var(--border) !important;
  border-radius: 3px !important;
}
[data-testid="stFileUploader"]:hover {
  border-color: var(--accent) !important;
}
[data-testid="stFileUploader"] * { color: var(--text2) !important; }

/* ── EXPANDER ── */
.streamlit-expanderHeader {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: 3px !important;
  color: var(--text2) !important;
  font-family: 'DM Mono', monospace !important;
  font-size: 11px !important;
  letter-spacing: 1.5px !important;
  text-transform: uppercase !important;
}
.streamlit-expanderContent {
  background: var(--bg) !important;
  border: 1px solid var(--border) !important;
  border-top: none !important;
  border-radius: 0 0 3px 3px !important;
}

/* ── DIVIDER ── */
hr {
  border-color: var(--border) !important;
  margin: 24px 0 !important;
}

/* ── ALERTS ── */
.stAlert {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: 3px !important;
  font-family: 'DM Sans', sans-serif !important;
}

/* ── CAPTION / SMALL TEXT ── */
.stCaption, .stCaption p {
  font-family: 'DM Mono', monospace !important;
  font-size: 10px !important;
  color: var(--text3) !important;
  letter-spacing: 0.5px !important;
}

/* ── HEADINGS ── */
h1, h2, h3, h4 {
  font-family: 'Playfair Display', serif !important;
  color: var(--text) !important;
}

/* ── SCROLLBAR — slim 4px ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }
</style>""", unsafe_allow_html=True)

# =============================================================
#  LOAD ENGINE
# =============================================================
@st.cache_resource
def load_engine():
    if not os.path.exists("models/fusion_recommender.pkl"):
        st.error("Run train_rnn.py then train_fusion.py first!")
        st.stop()

    data       = joblib.load("models/fusion_recommender.pkl")
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab      = data["vocab"]
    label_map  = data["label_map"]
    hidden_dim = data.get("hidden_dim", 512)
    max_len    = data.get("max_len", 24)
    joint_dim  = data.get("joint_dim", 256)

    # ResNet50
    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    resnet = nn.Sequential(*list(resnet.children())[:-1])
    resnet.to(device).eval()
    transform = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])

    # LSTM
    rnn = FashionLSTM(len(vocab), 300, hidden_dim, len(label_map))
    rnn.load_state_dict(torch.load("models/text_rnn_model.pth",
                                   map_location=device))
    rnn.to(device).eval()

    # Joint Projector
    projector = None
    if os.path.exists("models/joint_projector.pth"):
        projector = JointProjector(2048, hidden_dim, joint_dim)
        projector.load_state_dict(
            torch.load("models/joint_projector.pth", map_location=device))
        projector.to(device).eval()

    return data, resnet, rnn, transform, projector, device, max_len

# =============================================================
#  TEXT UTILITIES (must match train_rnn.py)
# =============================================================
def clean_text(text):
    return re.sub(r'[^a-z0-9\s]', '', str(text).lower()).strip()

# Keyword expansion maps
ARTICLE_KW = {
    'tshirt':'tshirts','tshirts':'tshirts','t-shirt':'tshirts','tee':'tshirts',
    'shirt':'shirts','shirts':'shirts',
    'jeans':'jeans','jean':'jeans','denim':'jeans',
    'dress':'dresses','dresses':'dresses','gown':'dresses',
    'top':'tops','tops':'tops','blouse':'tops',
    'jacket':'jackets','jackets':'jackets','blazer':'jackets',
    'sweatshirt':'sweatshirts','hoodie':'sweatshirts','sweater':'sweatshirts',
    'trouser':'trousers','trousers':'trousers','pants':'trousers','chino':'trousers',
    'watch':'watches','watches':'watches',
    'sneaker':'sneakers','sneakers':'sneakers',
    'shoe':'casual shoes','shoes':'casual shoes',
    'kurta':'kurtas','kurtas':'kurtas','kurti':'kurtas',
    'shorts':'shorts','short':'shorts',
    'handbag':'handbags','bag':'handbags','purse':'handbags',
    'sandal':'sandals','sandals':'sandals',
    'belt':'belts','belts':'belts',
    'wallet':'wallets','wallets':'wallets',
    'socks':'socks','sock':'socks',
    'brief':'briefs','briefs':'briefs','underwear':'briefs',
    'track':'track pants','trackpants':'track pants',
    'saree':'sarees','sari':'sarees',
    'legging':'leggings','leggings':'leggings',
    'skirt':'skirts','skirts':'skirts',
    'backpack':'backpacks','backpacks':'backpacks',
    'sunglass':'sunglasses','sunglasses':'sunglasses','shades':'sunglasses',
    'heel':'heels','heels':'heels','stiletto':'heels',
    'flat':'flats','flats':'flats','ballerina':'flats',
    'boot':'boots','boots':'boots',
}
GENDER_KW = {
    'men':'Men','man':'Men','male':'Men','mens':'Men','gents':'Men',
    'women':'Women','woman':'Women','female':'Women','womens':'Women','ladies':'Women',
    'girls':'Girls','girl':'Girls','boys':'Boys','boy':'Boys',
    'kids':'Boys','children':'Boys','unisex':'Unisex',
}
# Maps query colour word → canonical dataset colour for hard filtering
COLOUR_KW = {
    'black':'Black', 'white':'White', 'blue':'Blue', 'navy':'Navy Blue',
    'red':'Red', 'green':'Green', 'yellow':'Yellow', 'pink':'Pink',
    'purple':'Purple', 'orange':'Orange', 'grey':'Grey', 'gray':'Grey',
    'brown':'Brown', 'maroon':'Maroon', 'beige':'Beige', 'cream':'Cream',
    'teal':'Teal', 'khaki':'Khaki', 'olive':'Olive', 'gold':'Gold',
    'silver':'Silver', 'coral':'Coral', 'magenta':'Magenta',
    'lavender':'Lavender', 'rose':'Rose', 'rust':'Rust', 'mustard':'Mustard',
    'turquoise':'Turquoise Blue', 'peach':'Peach',
    'multi':'Multi', 'multicolor':'Multi',
}

# Colour groups: neighbouring shades accepted when hard-filtering
COLOUR_GROUPS = {
    'Black':      ['Black', 'Charcoal', 'Grey Melange'],
    'White':      ['White', 'Off White', 'Cream'],
    'Blue':       ['Blue', 'Navy Blue', 'Teal', 'Turquoise Blue'],
    'Navy Blue':  ['Navy Blue', 'Blue', 'Teal'],
    'Grey':       ['Grey', 'Grey Melange', 'Silver', 'Steel', 'Charcoal'],
    'Brown':      ['Brown', 'Coffee Brown', 'Mushroom Brown', 'Tan', 'Beige'],
    'Green':      ['Green', 'Olive', 'Sea Green', 'Lime Green', 'Fluorescent Green'],
    'Red':        ['Red', 'Maroon', 'Burgundy', 'Rust'],
    'Pink':       ['Pink', 'Rose', 'Mauve', 'Peach', 'Nude'],
    'Purple':     ['Purple', 'Lavender', 'Magenta'],
    'Yellow':     ['Yellow', 'Mustard', 'Gold'],
    'Orange':     ['Orange', 'Rust', 'Copper'],
    'Maroon':     ['Maroon', 'Red', 'Burgundy'],
}

FABRIC_KW = {
    'cotton':'cotton','denim':'denim','linen':'linen','silk':'silk',
    'polyester':'polyester','fleece':'fleece','wool':'wool','leather':'leather',
}
OCCASION_KW = {
    'casual':'casual','formal':'formal','party':'party','office':'formal',
    'ethnic':'ethnic','sports':'sports','gym':'sports','wedding':'ethnic',
    'travel':'travel',
}

def expand_query(text):
    
    #example: blue men's t-shirt -> blue men tshirt tshirts tshirts blue blue Men
    #Exact dataset pattern values (from styles_enriched.csv pattern column)

    PATTERN_KW_LOCAL = {
        'striped':'Striped','stripes':'Striped','stripe':'Striped',
        'solid':'Solid',
        'plain':'Plain',
        'printed':'Printed','print':'Printed',
        'graphic':'Graphic','graphics':'Graphic',
        'floral':'Floral','flower':'Floral','flowers':'Floral',
        'checked':'Checked','check':'Checked','checkered':'Checked','plaid':'Checked',
        'embroidered':'Embroidered','embroidery':'Embroidered',
        'colourblock':'Colour Block','colorblock':'Colour Block','colour block':'Colour Block',
        'camouflage':'Camouflage','camo':'Camouflage',
        'geometric':'Geometric','abstract':'Abstract',
        'quilted':'Quilted','distressed':'Distressed','washed':'Washed',
        'zari':'Zari','woven':'Woven','gradient':'Gradient',
    }
    # Exact brand column values from dataset (case-sensitive)
    BRAND_KW_LOCAL = {
        'nike':'Nike','puma':'Puma','adidas':'ADIDAS','reebok':'Reebok',
        'levis':'Levis','lee':'Lee','wrangler':'Wrangler',
        'arrow':'Arrow','fastrack':'Fastrack','jockey':'Jockey',
        'woodland':'Woodland','bata':'Bata','fila':'Fila',
        'lotto':'Lotto','wildcraft':'Wildcraft','fabindia':'Fabindia',
        'colorbar':'Colorbar','catwalk':'Catwalk','scullers':'Scullers',
        'proline':'Proline','flying':'Flying','jealous':'Jealous',
        'baggit':'Baggit','prozone':'Prozone','hrx':'HRX',
        'spykar':'Spykar','ucb':'United','benetton':'United',
        'french':'French','lino':'Lino',
    }
    words = clean_text(text).split()
    da, dg, dc_raw, df_kw, do = None, None, None, None, None
    dp, db_words = None, []
    for w in words:
        if not da and w in ARTICLE_KW:        da      = ARTICLE_KW[w]
        if not dg and w in GENDER_KW:         dg      = GENDER_KW[w]
        if not dc_raw and w in COLOUR_KW:     dc_raw  = w
        if not df_kw and w in FABRIC_KW:      df_kw   = FABRIC_KW[w]
        if not do and w in OCCASION_KW:       do      = OCCASION_KW[w]
        if not dp and w in PATTERN_KW_LOCAL:  dp      = PATTERN_KW_LOCAL[w]
        if w in BRAND_KW_LOCAL:               db_words.append(w)

    # Canonical colour for hard filtering (matches dataset values)
    dc_canonical = COLOUR_KW.get(dc_raw) if dc_raw else None

    parts = list(words)
    if da:       parts += [da, da]
    if dc_raw:   parts += [dc_raw, dc_raw]
    if dg:       parts += [dg]
    if df_kw:    parts += [df_kw]
    if do:       parts += [do]
    if dp:       parts += [dp.lower(), dp.lower()]
    if db_words: parts += db_words
    return " ".join(parts), da, dc_canonical, dg

# =============================================================
#  IMAGE COLOUR DETECTION (pixel-based, background-masked)
# =============================================================
def detect_image_colour(pil_image):
    """
    Detect dominant foreground colour of a fashion item from its pixels.
    Uses MEDIAN (not mean) — robust to logo/print text outliers.
    Dark+unsaturated shortcut catches black/charcoal garments reliably.
    Returns a canonical dataset colour string e.g. 'Black', 'Navy Blue', or None.
    """
    try:
        img = pil_image.convert("RGB").resize((100, 100))
        pixels = np.array(img).reshape(-1, 3).astype(float)
        # Remove white AND neutral-grey studio backgrounds
        px  = pixels
        r0, g0, b0 = px[:,0], px[:,1], px[:,2]
        near_white  = (r0>200) & (g0>200) & (b0>200)
        ch_range    = (np.abs(r0-g0)<20) & (np.abs(g0-b0)<20) & (np.abs(r0-b0)<20)
        near_grey   = ch_range & (np.maximum(np.maximum(r0,g0),b0) > 155)
        fg = px[~(near_white | near_grey)]
        if len(fg) < 50:
            return None
        # MEDIAN is robust to coloured logo/text that skews the mean
        r = float(np.median(fg[:,0]))
        g = float(np.median(fg[:,1]))
        b = float(np.median(fg[:,2]))
        # Dark + unsaturated → Black/Charcoal garment
        mx = max(r, g, b)
        saturation = (mx - min(r, g, b)) / (mx + 1e-5)
        if mx < 90 and saturation < 0.25:
            return 'Black'
        COLOUR_RGB = {
            'Black':     (55,  55,  55),
            'White':     (235, 235, 235),
            'Grey':      (140, 140, 140),
            'Navy Blue': (10,  20,  55),   # tuned for very dark navy garments
            'Blue':      (40,  90,  200),
            'Red':       (200, 30,  30),
            'Green':     (30,  140, 40),
            'Yellow':    (220, 200, 20),
            'Orange':    (220, 110, 30),
            'Pink':      (220, 110, 160),
            'Purple':    (110, 30,  180),
            'Brown':     (130, 75,  35),
            'Maroon':    (130, 20,  40),
            'Beige':     (210, 190, 155),
            'Khaki':     (185, 170, 115),
            'Olive':     (110, 115, 45),
            'Teal':      (20,  135, 130),
        }
        best, best_d = None, float("inf")
        for name, (cr, cg, cb) in COLOUR_RGB.items():
            d = ((r-cr)**2 + (g-cg)**2 + (b-cb)**2) ** 0.5
            if d < best_d:
                best_d, best = d, name
        return best
    except Exception:
        return None

# =============================================================
#  FEATURE EXTRACTION
# =============================================================
def get_text_vector_raw(text, rnn, vocab, device, max_len):
    # """Returns RAW 512-D LSTM vector (before projection)."""
    expanded, _, _, _ = expand_query(text)
    tokens = [vocab.get(t, vocab.get("<UNK>", 1))
              for t in expanded.split()]
    tokens = tokens[:max_len] + [0] * max(0, max_len - len(tokens))
    t      = torch.tensor([tokens], dtype=torch.long).to(device)
    with torch.no_grad():
        _, feats = rnn(t)
    return feats.cpu().numpy().flatten()          # 512-D

def get_image_vector_raw(image, resnet, transform, device):
    # """Returns RAW 2048-D ResNet vector (before projection)."""
    if image.mode != "RGB": image = image.convert("RGB")
    t = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        feats = resnet(t).flatten(1)
    return feats.cpu().numpy().flatten()          # 2048-D

def project_txt(txt_raw, projector, device):
    # """Projects raw 512-D RNN vector → 256-D joint space."""
    t = torch.tensor(txt_raw, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        out = projector.forward_txt(t)
    return out.cpu().numpy().flatten()            # 256-D

def project_img(img_raw, projector, device):
    # """Projects raw 2048-D CNN vector → 256-D joint space."""
    i = torch.tensor(img_raw, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        out = projector.forward_img(i)
    return out.cpu().numpy().flatten()            # 256-D

def get_joint_vector(img_raw, txt_raw, projector, device):
    """
    Projects raw 2048-D image + raw 512-D text → 256-D joint embedding.
    MUST receive raw vectors, NOT already-projected vectors.
    """
    i = torch.tensor(img_raw, dtype=torch.float32).unsqueeze(0).to(device)
    t = torch.tensor(txt_raw, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        joint = projector(i, t)                   # forward() handles both projections
    return joint.cpu().numpy().flatten()          # 256-D

def cosine_sim(query, matrix):
    q_norm = query / (np.linalg.norm(query) + 1e-10)
    m_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10)
    return m_norm @ q_norm

# =============================================================
#  PERSONALIZATION
# =============================================================
def personalize(sim_scores, pref_scores, weight=0.30):
    
    #final_score = 0.7 * similarity + 0.3 * preference

    return (1 - weight) * sim_scores + weight * pref_scores

# =============================================================
#  PRIC
# =============================================================
PRICE_BUDGETS = {
    "Budget  (under ₹500)":           (0,     500),
    "Value   (₹500 – ₹1,500)":        (500,   1500),
    "Mid-Range (₹1,500 – ₹3,000)":    (1500,  3000),
    "Premium (₹3,000 – ₹6,000)":      (3000,  6000),
    "Luxury  (above ₹6,000)":         (6000,  999999),
}

def parse_price(text):
    t = text.lower()
    for pattern, fn in [
        (r'(under|below|less than|within|upto|up to)\s*(?:rs\.?|₹\s*)?(\d+)', lambda m: (0, int(m.group(2)))),
        (r'(above|over|more than)\s*(?:rs\.?|₹\s*)?(\d+)',                    lambda m: (int(m.group(2)), 999999)),
        (r'between\s*(\d+)\s*(?:and|to|-)\s*(\d+)',                            lambda m: (int(m.group(1)), int(m.group(2)))),
    ]:
        m = re.search(pattern, t)
        if m: return fn(m)
    return None

# =============================================================
#  ATTRIBUTE ANALYSIS
# =============================================================
SEMANTIC_TAGS = {
    "Tshirts":     ["Cotton", "Casual", "Solid", "Regular Fit", "Lightweight", "Summer"],
    "Shirts":      ["Formal", "Cotton", "Full Sleeve", "Collar", "Slim Fit", "Office"],
    "Jeans":       ["Denim", "Slim Fit", "5-Pocket", "Stretch", "Everyday", "Casual"],
    "Dresses":     ["Flowy", "Summer", "Printed", "A-Line", "Feminine"],
    "Tops":        ["Casual", "Sleeveless", "Lightweight", "Summer", "Relaxed"],
    "Jackets":     ["Outerwear", "Zip-Up", "Warm", "Layer", "Sporty"],
    "Sweatshirts": ["Fleece", "Warm", "Winter", "Comfortable", "Oversized"],
    "Kurtas":      ["Ethnic", "Cotton", "Festive", "Traditional", "Printed"],
    "Watches":     ["Accessory", "Stainless Steel", "Water Resistant", "Classic"],
    "Sneakers":    ["Rubber Sole", "Lace-Up", "Breathable", "Everyday", "Casual"],
}
DEFAULT_TAGS = ["Fashion", "Style", "Trendy", "Popular"]

# =============================================================
#  OUTFIT INTELLIGENCE — full complement map with role labels
#  Structure: { articleType: [(pairedType, role_label), ...] }
#  Roles: "Bottom", "Top", "Footwear", "Accessory", "Outerwear", "Bag", "Layer"
# =============================================================
OUTFIT_MAP = {
    # ── TOPS ──────────────────────────────────────────────────
    "Tshirts": [
        ("Jeans",           "Bottom"),
        ("Track Pants",     "Bottom"),
        ("Shorts",          "Bottom"),
        ("Casual Trousers", "Bottom"),
        ("Sneakers",        "Footwear"),
        ("Casual Shoes",    "Footwear"),
        ("Sandals",         "Footwear"),
        ("Watches",         "Accessory"),
        ("Sunglasses",      "Accessory"),
        ("Caps",            "Accessory"),
        ("Backpacks",       "Bag"),
        ("Handbags",        "Bag"),
        ("Wallets",         "Bag"),
        ("Laptop Bags",     "Bag"),
        ("Jackets",         "Layer"),
        ("Sweatshirts",     "Layer"),
        ("Blazers",         "Layer"),
        ("Hoodies",         "Layer"),
    ],
    "Shirts": [
        ("Trousers",        "Bottom"),
        ("Jeans",           "Bottom"),
        ("Chinos",          "Bottom"),
        ("Formal Shoes",    "Footwear"),
        ("Loafers",         "Footwear"),
        ("Casual Shoes",    "Footwear"),
        ("Belts",           "Accessory"),
        ("Watches",         "Accessory"),
        ("Ties",            "Accessory"),
        ("Sunglasses",      "Accessory"),
        ("Laptop Bag",      "Bag"),
        ("Backpacks",       "Bag"),
        ("Wallets",         "Bag"),
        ("Briefcases",      "Bag"),
        ("Blazers",         "Layer"),
        ("Jackets",         "Layer"),
        ("Sweaters",        "Layer"),
        ("Vests",           "Layer"),
    ],
    "Tops": [
        ("Jeans",           "Bottom"),
        ("Trousers",        "Bottom"),
        ("Skirts",          "Bottom"),
        ("Leggings",        "Bottom"),
        ("Heels",           "Footwear"),
        ("Sandals",         "Footwear"),
        ("Flats",           "Footwear"),
        ("Earrings",        "Accessory"),
        ("Necklace",        "Accessory"),
        ("Watches",         "Accessory"),
        ("Handbags",        "Bag"),
        ("Clutches",        "Bag"),
        ("Tote Bags",       "Bag"),
        ("Backpacks",       "Bag"),
        ("Jackets",         "Layer"),
        ("Blazers",         "Layer"),
        ("Cardigans",       "Layer"),
        ("Shrugs",          "Layer"),
    ],
    "Sweatshirts": [
        ("Track Pants",     "Bottom"),
        ("Jeans",           "Bottom"),
        ("Shorts",          "Bottom"),
        ("Sneakers",        "Footwear"),
        ("Sports Shoes",    "Footwear"),
        ("Watches",         "Accessory"),
        ("Caps",            "Accessory"),
        ("Backpacks",       "Bag"),
        ("Gym Bags",        "Bag"),
        ("Duffel Bags",     "Bag"),
        ("Wallets",         "Bag"),
        ("Jackets",         "Layer"),
        ("Hoodies",         "Layer"),
        ("Vests",           "Layer"),
        ("Windcheaters",    "Layer"),
    ],
    "Kurtas": [
        ("Leggings",        "Bottom"),
        ("Salwar",          "Bottom"),
        ("Churidar",        "Bottom"),
        ("Sandals",         "Footwear"),
        ("Flats",           "Footwear"),
        ("Jutis",           "Footwear"),
        ("Earrings",        "Accessory"),
        ("Necklace",        "Accessory"),
        ("Bangles",         "Accessory"),
        ("Watches",         "Accessory"),
        ("Dupatta",         "Accessory"),
        ("Clutches",        "Bag"),
        ("Handbags",        "Bag"),
        ("Potli Bags",      "Bag"),
        ("Tote Bags",       "Bag"),
        ("Dupattas",        "Layer"),
        ("Shrugs",          "Layer"),
        ("Jackets",         "Layer"),
        ("Kurtis",          "Layer"),
    ],
    "Jackets": [
        ("Jeans",           "Bottom"),
        ("Trousers",        "Bottom"),
        ("Tshirts",         "Top"),
        ("Shirts",          "Top"),
        ("Sneakers",        "Footwear"),
        ("Boots",           "Footwear"),
        ("Watches",         "Accessory"),
        ("Sunglasses",      "Accessory"),
        ("Scarves",         "Accessory"),
        ("Backpacks",       "Bag"),
        ("Duffel Bags",     "Bag"),
        ("Wallets",         "Bag"),
        ("Messenger Bags",  "Bag"),
    ],
    "Blazers": [
        ("Trousers",        "Bottom"),
        ("Formal Trousers", "Bottom"),
        ("Jeans",           "Bottom"),
        ("Shirts",          "Top"),
        ("Tshirts",         "Top"),
        ("Formal Shoes",    "Footwear"),
        ("Loafers",         "Footwear"),
        ("Watches",         "Accessory"),
        ("Ties",            "Accessory"),
        ("Belts",           "Accessory"),
        ("Laptop Bag",      "Bag"),
        ("Briefcases",      "Bag"),
        ("Wallets",         "Bag"),
        ("Backpacks",       "Bag"),
    ],
    "Dresses": [
        ("Heels",           "Footwear"),
        ("Sandals",         "Footwear"),
        ("Flats",           "Footwear"),
        ("Wedges",          "Footwear"),
        ("Handbags",        "Bag"),
        ("Clutches",        "Bag"),
        ("Tote Bags",       "Bag"),
        ("Sling Bags",      "Bag"),
        ("Earrings",        "Accessory"),
        ("Necklace",        "Accessory"),
        ("Watches",         "Accessory"),
        ("Sunglasses",      "Accessory"),
        ("Scarves",         "Accessory"),
        ("Jackets",         "Layer"),
        ("Shrugs",          "Layer"),
        ("Cardigans",       "Layer"),
        ("Blazers",         "Layer"),
    ],
    "Sarees": [
        ("Heels",           "Footwear"),
        ("Sandals",         "Footwear"),
        ("Jutis",           "Footwear"),
        ("Earrings",        "Accessory"),
        ("Necklace",        "Accessory"),
        ("Bangles",         "Accessory"),
        ("Watches",         "Accessory"),
        ("Clutches",        "Bag"),
        ("Handbags",        "Bag"),
        ("Potli Bags",      "Bag"),
        ("Minaudiere",      "Bag"),
    ],
    "Leggings": [
        ("Kurtas",          "Top"),
        ("Tops",            "Top"),
        ("Tshirts",         "Top"),
        ("Sneakers",        "Footwear"),
        ("Flats",           "Footwear"),
        ("Sandals",         "Footwear"),
        ("Watches",         "Accessory"),
        ("Handbags",        "Bag"),
        ("Backpacks",       "Bag"),
        ("Tote Bags",       "Bag"),
        ("Clutches",        "Bag"),
    ],
    "Skirts": [
        ("Tops",            "Top"),
        ("Tshirts",         "Top"),
        ("Shirts",          "Top"),
        ("Heels",           "Footwear"),
        ("Flats",           "Footwear"),
        ("Sandals",         "Footwear"),
        ("Earrings",        "Accessory"),
        ("Watches",         "Accessory"),
        ("Handbags",        "Bag"),
        ("Clutches",        "Bag"),
        ("Tote Bags",       "Bag"),
        ("Sling Bags",      "Bag"),
        ("Jackets",         "Layer"),
        ("Blazers",         "Layer"),
        ("Cardigans",       "Layer"),
        ("Shrugs",          "Layer"),
    ],
    # ── BOTTOMS ───────────────────────────────────────────────
    "Jeans": [
        ("Tshirts",         "Top"),
        ("Shirts",          "Top"),
        ("Tops",            "Top"),
        ("Sweatshirts",     "Top"),
        ("Sneakers",        "Footwear"),
        ("Casual Shoes",    "Footwear"),
        ("Boots",           "Footwear"),
        ("Belts",           "Accessory"),
        ("Watches",         "Accessory"),
        ("Sunglasses",      "Accessory"),
        ("Backpacks",       "Bag"),
        ("Handbags",        "Bag"),
        ("Wallets",         "Bag"),
        ("Messenger Bags",  "Bag"),
        ("Jackets",         "Layer"),
        ("Blazers",         "Layer"),
        ("Hoodies",         "Layer"),
        ("Sweatshirts",     "Layer"),
    ],
    "Trousers": [
        ("Shirts",          "Top"),
        ("Tshirts",         "Top"),
        ("Blazers",         "Layer"),
        ("Formal Shoes",    "Footwear"),
        ("Loafers",         "Footwear"),
        ("Casual Shoes",    "Footwear"),
        ("Belts",           "Accessory"),
        ("Watches",         "Accessory"),
        ("Laptop Bag",      "Bag"),
        ("Briefcases",      "Bag"),
        ("Backpacks",       "Bag"),
        ("Wallets",         "Bag"),
        ("Jackets",         "Layer"),
        ("Vests",           "Layer"),
        ("Sweaters",        "Layer"),
        ("Cardigans",       "Layer"),
    ],
    "Track Pants": [
        ("Tshirts",         "Top"),
        ("Sweatshirts",     "Top"),
        ("Sports Shoes",    "Footwear"),
        ("Sneakers",        "Footwear"),
        ("Watches",         "Accessory"),
        ("Caps",            "Accessory"),
        ("Backpacks",       "Bag"),
        ("Gym Bags",        "Bag"),
        ("Duffel Bags",     "Bag"),
        ("Sports Bags",     "Bag"),
        ("Jackets",         "Layer"),
        ("Hoodies",         "Layer"),
        ("Windcheaters",    "Layer"),
        ("Vests",           "Layer"),
    ],
    "Shorts": [
        ("Tshirts",         "Top"),
        ("Shirts",          "Top"),
        ("Sneakers",        "Footwear"),
        ("Sandals",         "Footwear"),
        ("Flip Flops",      "Footwear"),
        ("Watches",         "Accessory"),
        ("Sunglasses",      "Accessory"),
        ("Caps",            "Accessory"),
        ("Backpacks",       "Bag"),
        ("Gym Bags",        "Bag"),
        ("Wallets",         "Bag"),
        ("Sling Bags",      "Bag"),
        ("Jackets",         "Layer"),
        ("Hoodies",         "Layer"),
        ("Sweatshirts",     "Layer"),
        ("Vests",           "Layer"),
    ],
    # ── FOOTWEAR ──────────────────────────────────────────────
    "Sneakers": [
        ("Jeans",           "Bottom"),
        ("Track Pants",     "Bottom"),
        ("Shorts",          "Bottom"),
        ("Tshirts",         "Top"),
        ("Sweatshirts",     "Top"),
        ("Watches",         "Accessory"),
        ("Caps",            "Accessory"),
        ("Backpacks",       "Bag"),
        ("Gym Bags",        "Bag"),
        ("Wallets",         "Bag"),
        ("Sling Bags",      "Bag"),
        ("Socks",           "Accessory"),
    ],
    "Formal Shoes": [
        ("Trousers",        "Bottom"),
        ("Jeans",           "Bottom"),
        ("Shirts",          "Top"),
        ("Blazers",         "Layer"),
        ("Belts",           "Accessory"),
        ("Watches",         "Accessory"),
        ("Ties",            "Accessory"),
        ("Laptop Bag",      "Bag"),
        ("Briefcases",      "Bag"),
        ("Wallets",         "Bag"),
        ("Backpacks",       "Bag"),
        ("Socks",           "Accessory"),
    ],
    "Heels": [
        ("Dresses",         "Top"),
        ("Skirts",          "Bottom"),
        ("Trousers",        "Bottom"),
        ("Jeans",           "Bottom"),
        ("Handbags",        "Bag"),
        ("Clutches",        "Bag"),
        ("Tote Bags",       "Bag"),
        ("Sling Bags",      "Bag"),
        ("Earrings",        "Accessory"),
        ("Watches",         "Accessory"),
        ("Sunglasses",      "Accessory"),
    ],
    "Sandals": [
        ("Dresses",         "Top"),
        ("Kurtas",          "Top"),
        ("Jeans",           "Bottom"),
        ("Shorts",          "Bottom"),
        ("Tops",            "Top"),
        ("Handbags",        "Bag"),
        ("Tote Bags",       "Bag"),
        ("Clutches",        "Bag"),
        ("Sling Bags",      "Bag"),
        ("Watches",         "Accessory"),
        ("Sunglasses",      "Accessory"),
    ],
    "Sports Shoes": [
        ("Track Pants",     "Bottom"),
        ("Shorts",          "Bottom"),
        ("Tshirts",         "Top"),
        ("Sweatshirts",     "Top"),
        ("Watches",         "Accessory"),
        ("Caps",            "Accessory"),
        ("Backpacks",       "Bag"),
        ("Gym Bags",        "Bag"),
        ("Duffel Bags",     "Bag"),
        ("Sports Bags",     "Bag"),
        ("Socks",           "Accessory"),
    ],
    "Casual Shoes": [
        ("Jeans",           "Bottom"),
        ("Chinos",          "Bottom"),
        ("Tshirts",         "Top"),
        ("Shirts",          "Top"),
        ("Watches",         "Accessory"),
        ("Belts",           "Accessory"),
        ("Backpacks",       "Bag"),
        ("Messenger Bags",  "Bag"),
        ("Wallets",         "Bag"),
        ("Sling Bags",      "Bag"),
        ("Socks",           "Accessory"),
    ],
    # ── ACCESSORIES ───────────────────────────────────────────
    "Watches": [
        ("Shirts",          "Top"),
        ("Tshirts",         "Top"),
        ("Jeans",           "Bottom"),
        ("Trousers",        "Bottom"),
        ("Formal Shoes",    "Footwear"),
        ("Sneakers",        "Footwear"),
        ("Belts",           "Accessory"),
        ("Sunglasses",      "Accessory"),
    ],
    "Sunglasses": [
        ("Tshirts",         "Top"),
        ("Shirts",          "Top"),
        ("Dresses",         "Top"),
        ("Jeans",           "Bottom"),
        ("Shorts",          "Bottom"),
        ("Sneakers",        "Footwear"),
        ("Watches",         "Accessory"),
        ("Caps",            "Accessory"),
    ],
    "Handbags": [
        ("Dresses",         "Top"),
        ("Tops",            "Top"),
        ("Jeans",           "Bottom"),
        ("Trousers",        "Bottom"),
        ("Heels",           "Footwear"),
        ("Sandals",         "Footwear"),
        ("Watches",         "Accessory"),
        ("Sunglasses",      "Accessory"),
    ],
    "Backpacks": [
        ("Tshirts",         "Top"),
        ("Sweatshirts",     "Top"),
        ("Jeans",           "Bottom"),
        ("Track Pants",     "Bottom"),
        ("Sneakers",        "Footwear"),
        ("Sports Shoes",    "Footwear"),
        ("Caps",            "Accessory"),
        ("Watches",         "Accessory"),
    ],
    "Belts": [
        ("Jeans",           "Bottom"),
        ("Trousers",        "Bottom"),
        ("Shirts",          "Top"),
        ("Formal Shoes",    "Footwear"),
        ("Watches",         "Accessory"),
    ],
}

# Role → display emoji + section heading
OUTFIT_ROLE_META = {
    "Top":       ("👕", "Tops to pair"),
    "Bottom":    ("👖", "Bottoms to pair"),
    "Footwear":  ("👟", "Footwear"),
    "Accessory": ("⌚", "Accessories"),
    "Bag":       ("👜", "Bags"),
    "Layer":     ("🧥", "Layering pieces"),
}

# Colour harmony: for each base colour, acceptable pairing colours
COLOUR_HARMONY = {
    "Black":     ["White", "Grey", "Red", "Yellow", "Blue", "Pink", "Beige", "Gold", "Silver"],
    "White":     ["Black", "Navy Blue", "Blue", "Grey", "Beige", "Brown", "Red", "Pink"],
    "Navy Blue": ["White", "Beige", "Grey", "Brown", "Khaki", "Cream", "Light Blue"],
    "Blue":      ["White", "Grey", "Brown", "Beige", "Black", "Orange", "Yellow"],
    "Grey":      ["White", "Black", "Blue", "Navy Blue", "Pink", "Yellow", "Red"],
    "Brown":     ["Beige", "Cream", "White", "Olive", "Khaki", "Navy Blue", "Orange"],
    "Beige":     ["Brown", "White", "Navy Blue", "Olive", "Khaki", "Black", "Maroon"],
    "Green":     ["White", "Beige", "Khaki", "Brown", "Black", "Navy Blue"],
    "Olive":     ["White", "Beige", "Brown", "Khaki", "Cream", "Black"],
    "Maroon":    ["White", "Beige", "Grey", "Black", "Cream", "Navy Blue"],
    "Red":       ["White", "Black", "Grey", "Navy Blue", "Beige", "Cream"],
    "Pink":      ["White", "Grey", "Black", "Beige", "Lavender", "Cream"],
    "Purple":    ["White", "Grey", "Black", "Lavender", "Beige", "Cream"],
    "Yellow":    ["White", "Black", "Navy Blue", "Grey", "Brown"],
    "Orange":    ["White", "Black", "Navy Blue", "Brown", "Beige"],
    "Khaki":     ["White", "Beige", "Brown", "Olive", "Navy Blue", "Black"],
    "Cream":     ["Brown", "Beige", "Navy Blue", "Maroon", "Black", "Olive"],
    "Teal":      ["White", "Beige", "Brown", "Navy Blue", "Black", "Cream"],
    "Rust":      ["White", "Beige", "Brown", "Cream", "Olive", "Black"],
    "Mustard":   ["White", "Brown", "Navy Blue", "Black", "Olive"],
}

# Keep old alias for any legacy reference
COMPLEMENT_MAP = {k: [p for p, _ in v] for k, v in OUTFIT_MAP.items()}


def get_outfit_sections(article_type):

    pairs = OUTFIT_MAP.get(article_type, [])            #article type ex: tshirt, []-pair
    sections = {}
    for atype, role in pairs:
        sections.setdefault(role, []).append(atype)     #group by role
    return sections


def pick_outfit_item(meta, article_type, base_colour, has_price, gender_filter=None, n=1):

    pool = meta[meta["articleType"] == article_type].copy() if "articleType" in meta.columns else pd.DataFrame()
    if len(pool) == 0:
        return pd.DataFrame()

    # Gender filter — allow Unisex always
    if gender_filter and "gender" in pool.columns:
        gm = (pool["gender"] == gender_filter) | (pool["gender"] == "Unisex")
        if gm.sum() >= 1:
            pool = pool[gm]

    # Colour harmony filter
    harmony_colours = COLOUR_HARMONY.get(base_colour, [])
    if harmony_colours and "baseColour" in pool.columns:
        colour_pool = pool[pool["baseColour"].isin(harmony_colours)]
        if len(colour_pool) >= 1:
            pool = colour_pool

    # Prefer higher rated items
    if "rating" in pool.columns:
        pool = pool.nlargest(min(20, len(pool)), "rating")

    return pool.sample(min(n, len(pool)), random_state=42) if len(pool) > 0 else pd.DataFrame()

def predict_attributes(meta, top_idx, scores):
    top = meta.iloc[top_idx]                                        #take top 20 items
    sw  = np.array([max(scores[i], 0) for i in top_idx])
    sw  = sw / (sw.sum() + 1e-10)                                   #sum sim scores and normalize

    def wmode(col):                                                 #calc attribute confidence
        if col not in top.columns: return "Unknown", 0.5
        c = {}
        for v, w in zip(top[col].fillna("Unknown"), sw):
            c[str(v)] = c.get(str(v), 0) + w
        best = max(c, key=c.get)
        return best, min(c[best] / (sum(c.values()) + 1e-10), 0.99) #best_weight / total_weight
                                                                    #max is .99
    def dist(col, n=5):
        if col not in top.columns: return {}
        c = {}
        for v, w in zip(top[col].fillna("Unknown"), sw):
            c[str(v)] = c.get(str(v), 0) + w
        t = sum(c.values()) + 1e-10
        return {k: round(v/t, 3)
                for k, v in sorted(c.items(), key=lambda x: -x[1])[:n]}

    mc, mc_c = wmode("masterCategory")
    sc, sc_c = wmode("subCategory")
    at, at_c = wmode("articleType")
    co, co_c = wmode("baseColour")
    ge, ge_c = wmode("gender")
    se, se_c = wmode("season")
    us, us_c = wmode("usage")

    return {
        "overall":        float(np.mean([mc_c, sc_c, at_c, co_c, ge_c, se_c, us_c])),
        "masterCategory": (mc, mc_c),
        "subCategory":    (sc, sc_c),
        "articleType":    (at, at_c),
        "baseColour":     (co, co_c),
        "gender":         (ge, ge_c),
        "season":         (se, se_c),
        "usage":          (us, us_c),
        "article_dist":   dist("articleType"),
        "tags":           SEMANTIC_TAGS.get(at, DEFAULT_TAGS),
    }

# =============================================================
#  UI HELPERS
# =============================================================
def attr_card(label, value, conf, color):
    bar = int(conf * 100)
    st.markdown(f"""
    <div style="background:#111110;border:1px solid #2a2a28;border-radius:3px;
                padding:16px;margin-bottom:10px;position:relative;overflow:hidden;">
      <div style="position:absolute;top:0;left:0;right:0;height:2px;
                  background:{color};
                  transform-origin:left;"></div>
      <div style="font-family:'DM Mono',monospace;font-size:9px;letter-spacing:2px;
                  text-transform:uppercase;color:#6a6860;margin-bottom:8px;">{label}</div>
      <div style="font-family:'DM Sans',sans-serif;font-size:18px;font-weight:500;
                  color:#f0ede8;margin-bottom:10px;line-height:1.2;">{value}</div>
      <div style="background:#1a1a18;border-radius:2px;height:3px;overflow:hidden;">
        <div style="background:{color};width:{bar}%;height:3px;border-radius:2px;
                    transition:width 0.8s cubic-bezier(0.34,1.56,0.64,1);"></div>
      </div>
      <div style="font-family:'DM Mono',monospace;font-size:10px;color:#6a6860;
                  margin-top:6px;">{bar}% confidence</div>
    </div>""", unsafe_allow_html=True)

def render_analysis(attrs, mode, valid_text, valid_img, has_joint):
    pct = int(attrs["overall"] * 100)
    if mode == "Fusion (Text + Image)" and valid_text and valid_img:
        cnn_w, rnn_w = "50%", "50%"
        space_label  = "256-D Joint Embedding Space" if has_joint else "Late Fusion"
    elif valid_img: cnn_w, rnn_w, space_label = "100%", "0%", "256-D CNN Space"
    else:           cnn_w, rnn_w, space_label = "0%", "100%", "256-D RNN Space"

    # Confidence header — exact StyleSense confidence-header layout
    st.markdown(f"""
    <div style="display:flex;align-items:center;justify-content:space-between;
                margin-bottom:20px;flex-wrap:wrap;gap:16px;">
      <div style="display:flex;align-items:baseline;gap:10px;">
        <span style="font-family:'Playfair Display',serif;font-size:56px;font-weight:900;
                     color:#c8a96e;line-height:1;">{pct}%</span>
        <div>
          <div style="font-family:'DM Sans',sans-serif;font-size:13px;
                      color:#9a9890;font-weight:500;">Overall Confidence</div>
          <div style="font-family:'DM Mono',monospace;font-size:10px;
                      color:#6a6860;letter-spacing:1px;text-transform:uppercase;">
            {space_label}
          </div>
        </div>
      </div>
      <div style="display:flex;flex-direction:column;gap:6px;align-items:flex-end;">
        <div style="display:flex;align-items:center;gap:8px;font-family:'DM Mono',monospace;
                    font-size:10px;color:#9a9890;letter-spacing:0.5px;">
          <div style="width:6px;height:6px;border-radius:50%;background:#5a8ae0;"></div>
          CNN Visual: <span style="color:#5a8ae0;margin-left:4px;">{cnn_w}</span>
        </div>
        <div style="display:flex;align-items:center;gap:8px;font-family:'DM Mono',monospace;
                    font-size:10px;color:#9a9890;letter-spacing:0.5px;">
          <div style="width:6px;height:6px;border-radius:50%;background:#9a7ae0;"></div>
          RNN Text: <span style="color:#9a7ae0;margin-left:4px;">{rnn_w}</span>
        </div>
      </div>
    </div>
    <div style="height:1px;background:#2a2a28;margin-bottom:20px;"></div>
    """, unsafe_allow_html=True)

    # Attribute Predictions section title
    st.markdown("""
    <div style="font-family:'Playfair Display',serif;font-size:18px;font-weight:700;
                color:#f0ede8;display:flex;align-items:center;gap:10px;margin-bottom:16px;">
      <span style="font-size:14px;opacity:0.6;">🎯</span> Attribute Predictions
    </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        attr_card("Master Category", attrs["masterCategory"][0], attrs["masterCategory"][1], "#c8a96e")
        attr_card("Article Type",    attrs["articleType"][0],    attrs["articleType"][1],    "#9a7ae0")
        attr_card("Gender",          attrs["gender"][0],         attrs["gender"][1],         "#6ab87a")
        attr_card("Usage",           attrs["usage"][0],          attrs["usage"][1],          "#e05a4a")
    with c2:
        attr_card("Sub Category",    attrs["subCategory"][0],    attrs["subCategory"][1],    "#5a8ae0")
        attr_card("Base Color",      attrs["baseColour"][0],     attrs["baseColour"][1],     "#5a8ae0")
        attr_card("Season",          attrs["season"][0],         attrs["season"][1],         "#e08040")

    # Category Distribution
    st.markdown("""
    <div style="height:1px;background:#2a2a28;margin:16px 0 16px;"></div>
    <div style="font-family:'Playfair Display',serif;font-size:18px;font-weight:700;
                color:#f0ede8;display:flex;align-items:center;gap:10px;margin-bottom:16px;">
      <span style="font-size:14px;opacity:0.6;">📊</span> Category Distribution
    </div>""", unsafe_allow_html=True)

    for i, (cat, prob) in enumerate(attrs["article_dist"].items(), 1):
        w = int(prob * 100)
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:12px;padding:10px 14px;
                    background:#111110;border:1px solid #2a2a28;border-radius:3px;
                    margin-bottom:6px;font-size:13px;">
          <span style="font-family:'DM Mono',monospace;font-size:10px;
                       color:#6a6860;width:24px;flex-shrink:0;">#{i}</span>
          <span style="color:#f0ede8;flex:1;">{cat}</span>
          <div style="width:120px;height:4px;background:#1a1a18;
                      border-radius:2px;overflow:hidden;">
            <div style="background:linear-gradient(90deg,#c8a96e,#e8c87e);
                        width:{w}%;height:4px;border-radius:2px;"></div>
          </div>
          <span style="font-family:'DM Mono',monospace;font-size:11px;
                       color:#c8a96e;width:36px;text-align:right;">{w}%</span>
        </div>""", unsafe_allow_html=True)

    # Semantic Tags
    st.markdown("""
    <div style="height:1px;background:#2a2a28;margin:16px 0 16px;"></div>
    <div style="font-family:'Playfair Display',serif;font-size:18px;font-weight:700;
                color:#f0ede8;display:flex;align-items:center;gap:10px;margin-bottom:16px;">
      <span style="font-size:14px;opacity:0.6;">🏷️</span> Semantic Feature Tags
    </div>""", unsafe_allow_html=True)

    html = ""
    for i, tag in enumerate(attrs["tags"]):
        if i < 2:
            s = ("border:1px solid #c8a96e;color:#c8a96e;"
                 "background:rgba(200,169,110,0.08);")
        else:
            s = "border:1px solid #2a2a28;color:#6a6860;background:#111110;"
        html += (f'<span style="font-family:DM Mono,monospace;display:inline-block;'
                 f'padding:5px 12px;border-radius:2px;{s}'
                 f'margin:4px;font-size:11px;letter-spacing:0.5px;">{tag}</span>')
    st.markdown(f'<div style="margin-top:4px;">{html}</div>', unsafe_allow_html=True)

# CHANGE 1: Match score shown as decimal (e.g. 0.98) — no percentage bar
def product_card(col, item, score, has_price=True, fixed_img_height=None, exact_match=False):
    with col:
        path   = str(item.get("image_path", ""))
        name   = str(item.get("productDisplayName", "Product"))
        col_v  = str(item.get("baseColour", ""))
        atype  = str(item.get("articleType", ""))
        brand  = str(item.get("brand", ""))
        price  = item.get("price", None)
        orig   = item.get("original_price", None)
        disc   = item.get("discount_pct", None)
        rating = item.get("rating", None)
        sc     = min(float(max(score, 0)), 1.0)
        # Format score as decimal with 2 decimal places
        sc_dec = f"{sc:.2f}"

        # ── image ──
        # CHANGE 2: Use object-fit:contain so portrait images are never cropped
        if fixed_img_height:
            if os.path.exists(path):
                st.markdown(
                    f'<div style="width:100%;height:{fixed_img_height}px;overflow:hidden;'
                    f'border-radius:2px;display:flex;align-items:center;justify-content:center;'
                    f'background:#1a1a18;">'
                    f'<img src="data:image/jpeg;base64,{_img_to_b64(path)}" '
                    f'style="width:100%;height:{fixed_img_height}px;object-fit:contain;" /></div>',
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<div style="height:{fixed_img_height}px;background:#1a1a18;border-radius:2px;'
                    f'display:flex;align-items:center;justify-content:center;'
                    f'color:#2a2a28;font-size:32px;border:1px solid #2a2a28;">✦</div>',
                    unsafe_allow_html=True)
        else:
            if os.path.exists(path):
                st.image(path, use_container_width=True)
            else:
                st.markdown(
                    '<div style="height:180px;background:#1a1a18;border-radius:2px;'
                    'display:flex;align-items:center;justify-content:center;'
                    'color:#2a2a28;font-size:32px;border:1px solid #2a2a28;">✦</div>',
                    unsafe_allow_html=True)

        # ── build pieces as plain strings (no nested f-strings) ──
        disc_badge = ""
        if orig and disc and int(disc) > 0:
            disc_badge = (
                '<span style="font-family:DM Mono,monospace;font-size:9px;'
                'letter-spacing:0.5px;background:rgba(106,184,122,0.1);'
                'color:#6ab87a;padding:2px 8px;border-radius:2px;'
                'border:1px solid rgba(106,184,122,0.3);">'
                + str(int(disc)) + '% off</span>'
            )

        # ── exact match badge (dataset image detected) ──
        exact_badge = ""
        if exact_match:
            exact_badge = (
                '<span style="font-family:DM Mono,monospace;font-size:9px;'
                'letter-spacing:0.5px;background:rgba(200,169,110,0.15);'
                'color:#c8a96e;padding:2px 8px;border-radius:2px;'
                'border:1px solid rgba(200,169,110,0.5);">✦ Exact Match</span>'
            )

        price_html = ""
        if has_price and price is not None:
            price_html = (
                '<span style="font-family:DM Mono,monospace;color:#c8a96e;font-size:15px;">₹'
                + "{:,}".format(int(price)) + '</span>'
            )
            if orig and disc and int(disc) > 0:
                price_html += (
                    ' <span style="font-family:DM Mono,monospace;color:#6a6860;'
                    'text-decoration:line-through;font-size:11px;">₹'
                    + "{:,}".format(int(orig)) + '</span>'
                )

        stars_html = ""
        if rating is not None:
            r      = float(rating)
            filled = int(round(r))
            stars_html = (
                '<span style="color:#c8a96e;font-size:10px;">'
                + "★" * filled + "☆" * (5 - filled) + '</span>'
                + '<span style="font-family:DM Mono,monospace;color:#6a6860;'
                  'font-size:10px;margin-left:4px;">' + "{:.1f}".format(r) + '</span>'
            )

        # CHANGE 1: Match shown as decimal score, no progress bar
        match_score_html = ""
        if sc > 0:
            score_color = "#c8a96e" if not exact_match else "#e8c87e"
            match_score_html = (
                '<div style="margin-top:10px;display:flex;align-items:center;'
                'justify-content:space-between;">'
                '<span style="font-family:DM Mono,monospace;font-size:9px;'
                'letter-spacing:1.5px;color:#6a6860;text-transform:uppercase;">Match</span>'
                '<span style="font-family:DM Mono,monospace;font-size:13px;'
                'color:' + score_color + ';font-weight:500;">' + sc_dec + '</span>'
                '</div>'
            )

        name_safe = name[:46] + ("…" if len(name) > 46 else "")

        # Border highlight for exact match card
        card_border = "#c8a96e" if exact_match else "#2a2a28"
        card_shadow = "box-shadow:0 0 0 1px rgba(200,169,110,0.3);" if exact_match else ""

        card_html = (
            '<div style="background:#111110;border:1px solid ' + card_border + ';border-radius:3px;'
            'padding:14px;margin-top:6px;' + card_shadow + '">'

            # header row: article type + badges
            '<div style="display:flex;justify-content:space-between;align-items:center;'
            'margin-bottom:8px;gap:6px;flex-wrap:wrap;">'
            '<span style="font-family:DM Mono,monospace;font-size:9px;letter-spacing:2px;'
            'color:#6a6860;text-transform:uppercase;">' + atype + '</span>'
            '<div style="display:flex;gap:4px;flex-wrap:wrap;">'
            + exact_badge + disc_badge +
            '</div>'
            '</div>'

            # product name
            '<div style="font-family:DM Sans,sans-serif;font-size:13px;font-weight:500;'
            'color:#f0ede8;line-height:1.4;margin-bottom:5px;">' + name_safe + '</div>'

            # brand · colour
            '<div style="font-family:DM Mono,monospace;font-size:9px;letter-spacing:0.5px;'
            'color:#6a6860;margin-bottom:8px;">' + brand + ' · ' + col_v + '</div>'

            # price
            + price_html +

            # stars
            '<div style="margin-top:4px;">' + stars_html + '</div>'

            # match score (decimal, no bar)
            + match_score_html +

            '</div>'
        )

        st.markdown(card_html, unsafe_allow_html=True)


def _img_to_b64(path):
    """Convert image file to base64 string for inline HTML embedding."""
    import base64
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        return ""

# =============================================================
#  DATASET IMAGE MATCH DETECTION
#  Computes a difference-hash (dHash) of the uploaded image and
#  compares it against every image file listed in metadata.
#  Returns the metadata row-index of the exact match, or None.
#  No recommendation logic is altered by this function.
# =============================================================
@st.cache_data(show_spinner=False)
def _build_dataset_hash_index(_meta, hash_size=16):

    def dhash(img_path, size=hash_size):
        gray = Image.open(img_path).convert("L").resize((size + 1, size), Image.LANCZOS)
        arr  = np.array(gray)
        diff = arr[:, 1:] > arr[:, :-1]
        return diff.flatten().tobytes()

    index = {}
    if "image_path" not in _meta.columns:
        return index
    for idx, path in enumerate(_meta["image_path"].values):
        try:
            if os.path.exists(str(path)):
                h = dhash(str(path))
                index[h] = idx          # last writer wins for hash collisions
        except Exception:
            continue
    return index


def find_dataset_match(pil_image, meta, hash_size=16):
    
    # Returns the integer row-index of the matching item in `meta`,
    # or None if the uploaded image is not from the dataset.

    # Strategy:
    #   1. Build (or retrieve from cache) a dHash index of all dataset images.
    #   2. Hash the uploaded image.
    #   3. Return the matching index if found; None otherwise.
    
    def dhash(img, size=hash_size):
        gray = img.convert("L").resize((size + 1, size), Image.LANCZOS)
        arr  = np.array(gray)
        diff = arr[:, 1:] > arr[:, :-1]
        return diff.flatten().tobytes()

    try:
        query_hash = dhash(pil_image)
    except Exception:
        return None

    hash_index = _build_dataset_hash_index(meta)
    return hash_index.get(query_hash, None)     # O(1) lookup

# =============================================================
#  SESSION STATE
# =============================================================
if "recently_viewed" not in st.session_state:
    st.session_state.recently_viewed = []

# CHANGE 4: Store similarity score in history alongside price
def add_to_history(items_df, scores=None):
    for i, (_, row) in enumerate(items_df.iterrows()):
        sc_val = float(scores[i]) if scores is not None and i < len(scores) else 0.0
        e = {
            "name":  str(row.get("productDisplayName", ""))[:40],
            "path":  str(row.get("image_path", "")),
            "price": row.get("price", None),
            "brand": str(row.get("brand", "")),
            "score": round(sc_val, 2),
        }
        if e not in st.session_state.recently_viewed:
            st.session_state.recently_viewed.insert(0, e)
    st.session_state.recently_viewed = st.session_state.recently_viewed[:12]

# =============================================================
#  MAIN
# =============================================================
def main():
    data, resnet, rnn, transform, projector, device, max_len = load_engine()
    meta      = data["metadata"]
    has_price = "price" in meta.columns
    has_joint = (projector is not None) and ("joint_features" in data)
    has_pref  = "user_preference_score" in meta.columns

    # Preference scores for personalization
    pref_scores = (meta["user_preference_score"].values.astype(np.float32)
                   if has_pref else np.zeros(len(meta)))

    # ── SIDEBAR ──────────────────────────────────────────────
    st.sidebar.markdown("""
    <div style="padding:24px 4px 16px;">
      <div style="font-family:'DM Mono',monospace;font-size:10px;letter-spacing:3px;
                  color:#6a6860;margin-bottom:10px;text-transform:uppercase;">
        Multimodal AI · CNN + RNN Fusion
      </div>
      <div style="font-family:'Playfair Display',serif;font-size:26px;font-weight:900;
                  color:#f0ede8;letter-spacing:-0.5px;line-height:1;">
        Style<span style="color:#c8a96e;">Sense</span>
      </div>
    </div>
    <div style="height:1px;background:#2a2a28;margin-bottom:20px;"></div>
    <div style="font-family:'DM Mono',monospace;font-size:10px;letter-spacing:3px;
                color:#6a6860;text-transform:uppercase;margin-bottom:10px;">
      Search Mode
    </div>""", unsafe_allow_html=True)
    mode = st.sidebar.radio("", ["Text Search", "Image Search", "Fusion (Text + Image)"],
                            label_visibility="collapsed")
    st.sidebar.markdown("""
    <div style="height:1px;background:#2a2a28;margin:16px 0 16px;"></div>
    <div style="font-family:'DM Mono',monospace;font-size:10px;letter-spacing:3px;
                color:#6a6860;text-transform:uppercase;margin-bottom:10px;">
      Refine Results
    </div>""", unsafe_allow_html=True)

    def opts(col):
        return (["All"] + sorted(meta[col].dropna().unique().tolist())
                if col in meta.columns else ["All"])

    sel_gender   = st.sidebar.selectbox("Gender",      opts("gender"))
    sel_category = st.sidebar.selectbox("Category",    opts("articleType"))
    sel_color    = st.sidebar.selectbox("Color",       opts("baseColour"))
    sel_brand    = st.sidebar.selectbox("Brand Tier",  opts("brand_tier") if "brand_tier" in meta.columns else ["All"])
    sel_fabric   = st.sidebar.selectbox("Fabric",      opts("fabric")     if "fabric"     in meta.columns else ["All"])
    sel_occasion = st.sidebar.selectbox("Occasion",    opts("occasion")   if "occasion"   in meta.columns else ["All"])

    # CHANGE 5: Max rating filter instead of min rating
    sel_rating_max = st.sidebar.slider("Max Rating", 1.0, 5.0, 5.0, 0.5) if "rating" in meta.columns else 5.0

    st.sidebar.markdown("""
    <div style="height:1px;background:#2a2a28;margin:16px 0 16px;"></div>
    <div style="font-family:'DM Mono',monospace;font-size:10px;letter-spacing:3px;
                color:#6a6860;text-transform:uppercase;margin-bottom:10px;">
      Price Range
    </div>""", unsafe_allow_html=True)
    price_mode = st.sidebar.radio("", ["No Filter", "Budget Tier", "Custom Range"],
                                  label_visibility="collapsed")
    price_lo, price_hi = 0, 999999
    if price_mode == "Budget Tier" and has_price:
        tier = st.sidebar.selectbox("Tier", list(PRICE_BUDGETS.keys()))
        price_lo, price_hi = PRICE_BUDGETS[tier]
    elif price_mode == "Custom Range" and has_price:
        mn, mx = int(meta["price"].min()), int(meta["price"].max())
        price_lo, price_hi = st.sidebar.slider("Range (₹)", mn, mx, (mn, mx))

    # Personalization weight
    if has_pref:
        st.sidebar.markdown("""
        <div style="height:1px;background:#2a2a28;margin:16px 0 16px;"></div>
        <div style="font-family:'DM Mono',monospace;font-size:10px;letter-spacing:3px;
                    color:#6a6860;text-transform:uppercase;margin-bottom:10px;">
          Personalization
        </div>""", unsafe_allow_html=True)
        pref_weight = st.sidebar.slider(
            "Preference Weight", 0.0, 0.5, 0.10, 0.05,
            help="0 = pure similarity | 0.5 = heavily personalized")
    else:
        pref_weight = 0.0

    st.sidebar.markdown("""
    <div style="height:1px;background:#2a2a28;margin:16px 0 16px;"></div>
    <div style="font-family:'DM Mono',monospace;font-size:10px;letter-spacing:3px;
                color:#6a6860;text-transform:uppercase;margin-bottom:10px;">
      Display
    </div>""", unsafe_allow_html=True)
    top_k         = st.sidebar.slider("Results count", 3, 12, 6)
    show_analysis = st.sidebar.checkbox("Show AI Analysis",        value=True)
    show_outfits  = st.sidebar.checkbox("Show Outfit Suggestions", value=True)
    show_history  = st.sidebar.checkbox("Show Recently Viewed",    value=True)

    st.sidebar.markdown("""
    <div style="margin-top:28px;padding:14px 16px;background:#111110;
                border:1px solid #2a2a28;border-radius:3px;">
      <div style="font-family:'DM Mono',monospace;font-size:9px;letter-spacing:2px;
                  color:#6a6860;text-transform:uppercase;margin-bottom:8px;">Powered by</div>
      <div style="font-family:'DM Mono',monospace;font-size:10px;color:#6a6860;line-height:2;">
        ResNet50 · LSTM<br>Triplet Loss · 256-D Joint Space
      </div>
    </div>""", unsafe_allow_html=True)

    # ── HEADER ───────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center;padding:64px 48px 48px;
                border-bottom:1px solid #2a2a28;margin-bottom:40px;position:relative;z-index:1;">
      <div style="font-family:'DM Mono',monospace;font-size:11px;letter-spacing:4px;
                  color:#c8a96e;text-transform:uppercase;margin-bottom:20px;">
        Fashion Product Intelligence
      </div>
      <h1 style="font-family:'Playfair Display',serif;font-size:clamp(38px,5vw,68px);
                 font-weight:900;line-height:1.05;letter-spacing:-2px;
                 color:#f0ede8;margin-bottom:18px;">
        Find your perfect <em style="font-style:italic;color:#c8a96e;">style</em>
      </h1>
      <p style="font-family:'DM Sans',sans-serif;max-width:520px;margin:0 auto 32px;
                font-size:15px;color:#9a9890;line-height:1.7;">
        Multimodal CNN + RNN fusion model analyzes visual features and text semantics
        to surface the most relevant fashion products — trained on the Myntra Fashion Dataset.
      </p>
      <div style="display:flex;gap:10px;justify-content:center;flex-wrap:wrap;">
        <span style="font-family:'DM Mono',monospace;font-size:10px;letter-spacing:1.5px;
                     text-transform:uppercase;padding:5px 12px;border-radius:2px;
                     border:1px solid #5a8ae0;color:#5a8ae0;
                     background:rgba(90,138,224,0.08);">Visual Feature Extraction (CNN)</span>
        <span style="font-family:'DM Mono',monospace;font-size:10px;letter-spacing:1.5px;
                     text-transform:uppercase;padding:5px 12px;border-radius:2px;
                     border:1px solid #9a7ae0;color:#9a7ae0;
                     background:rgba(154,122,224,0.08);">Text Understanding (RNN)</span>
        <span style="font-family:'DM Mono',monospace;font-size:10px;letter-spacing:1.5px;
                     text-transform:uppercase;padding:5px 12px;border-radius:2px;
                     border:1px solid #c8a96e;color:#c8a96e;
                     background:rgba(200,169,110,0.08);">Multimodal Fusion</span>
        <span style="font-family:'DM Mono',monospace;font-size:10px;letter-spacing:1.5px;
                     text-transform:uppercase;padding:5px 12px;border-radius:2px;
                     border:1px solid #6ab87a;color:#6ab87a;
                     background:rgba(106,184,122,0.08);">Real-Time Recommendation</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── INPUT ────────────────────────────────────────────────
    user_text, user_image = "", None
    ci, cp = st.columns([3, 1])
    with ci:
        if mode in ["Text Search", "Fusion (Text + Image)"]:
            user_text = st.text_input(
                "Describe the fashion item",
                placeholder="e.g. 'Black men tshirt under 1000', "
                            "'Blue slim fit jeans', 'Red women dress cotton party'")
        if mode in ["Image Search", "Fusion (Text + Image)"]:
            up = st.file_uploader("Upload product image",
                                  type=["jpg", "png", "jpeg"])
            if up: user_image = Image.open(up)
    with cp:
        if user_image is not None:
            st.image(user_image, width=160, caption="Query image")

    search = st.button("⚡  Analyze with AI", type="primary",
                       use_container_width=True)

    if search:
        txt_vec, img_vec   = None, None
        valid_text, valid_img = False, False
        detected_price   = None
        detected_colour  = None   # set by text query or image analysis
        detected_article = None
        detected_gender  = None

        # Detect price from text
        if user_text:
            detected_price = parse_price(user_text)
            if detected_price and price_mode == "No Filter":
                price_lo, price_hi = detected_price
                st.info(f"💰 Price filter detected: ₹{price_lo:,} – ₹{price_hi:,}")

        # Text → raw 512-D (projection happens later per mode)
        if user_text:
            with st.spinner("Analysing text query..."):
                txt_vec    = get_text_vector_raw(user_text, rnn,
                                                 data["vocab"], device, max_len)
                valid_text = True
            _, detected_article, detected_colour, detected_gender = expand_query(user_text)
            # Build detection caption including pattern if found
            _PAT_SHOW = {
                'striped':'Striped','stripes':'Striped','solid':'Solid','plain':'Solid',
                'printed':'Printed','graphic':'Printed','floral':'Floral',
                'checked':'Checked','plaid':'Checked','embroidered':'Embroidered',
                'camo':'Camouflage','camouflage':'Camouflage','colorblock':'Colourblocked',
            }
            _PAT_CAP = {
                'striped':'Striped','stripes':'Striped','solid':'Solid','plain':'Plain',
                'printed':'Printed','graphic':'Graphic','floral':'Floral',
                'checked':'Checked','plaid':'Checked','embroidered':'Embroidered',
                'camo':'Camouflage','camouflage':'Camouflage',
                'colourblock':'Colour Block','colorblock':'Colour Block',
            }
            _BRAND_CAP = {
                'nike':'Nike','puma':'Puma','adidas':'Adidas','reebok':'Reebok',
                'levis':'Levis','lee':'Lee','wrangler':'Wrangler','arrow':'Arrow',
                'fastrack':'Fastrack','jockey':'Jockey','fila':'Fila','hrx':'HRX',
                'ucb':'UCB','benetton':'UCB','woodland':'Woodland','bata':'Bata',
                'spykar':'Spykar','flying':'Flying Machine','fabindia':'Fabindia',
            }
            _qwords = clean_text(user_text).split()
            _det_pat   = next((_PAT_CAP[w]   for w in _qwords if w in _PAT_CAP),   None)
            _det_brand = next((_BRAND_CAP[w] for w in _qwords if w in _BRAND_CAP), None)
            tags = " | ".join(filter(None, [
                f"Type: {detected_article}"    if detected_article else None,
                f"Colour: {detected_colour}"   if detected_colour  else None,
                f"Gender: {detected_gender}"   if detected_gender  else None,
                f"Pattern: {_det_pat}"         if _det_pat         else None,
                f"Brand: {_det_brand}"         if _det_brand       else None]))
            if tags: st.caption(f"🔍 Detected → {tags}")

        # Image → raw 2048-D (projection happens later per mode)
        if user_image:
            with st.spinner("Analysing image..."):
                img_vec   = get_image_vector_raw(user_image, resnet,
                                                 transform, device)
                valid_img = True

        # ── SELECT FEATURE MATRIX ────────────────────────────
        if mode == "Fusion (Text + Image)" and valid_text and valid_img:
            if has_joint and projector:
                # Raw 2048-D img + raw 512-D txt → projector → 256-D joint
                query_vec   = get_joint_vector(img_vec, txt_vec, projector, device)
                feat_matrix = data["joint_features"]
                mode_label  = "🧠 Joint Embedding Space (CNN + RNN → 256-D)"
            else:
                # Fallback late fusion with raw normalised vectors
                i_n = img_vec / (np.linalg.norm(img_vec) + 1e-10)
                t_n = txt_vec / (np.linalg.norm(txt_vec) + 1e-10)
                query_vec   = (i_n + t_n) / 2
                feat_matrix = data.get("joint_features",
                              data.get("image_features"))
                mode_label  = "🧠 Late Fusion"
            st.success(mode_label)
            valid = True
        elif valid_text:
            if projector and "txt_proj_features" in data:
                # Project raw 512-D → 256-D
                query_vec   = project_txt(txt_vec, projector, device)
                feat_matrix = data["txt_proj_features"]
            else:
                query_vec   = txt_vec
                feat_matrix = data["text_features"]
            valid = True
        elif valid_img:
            if projector and "img_proj_features" in data:
                # Project raw 2048-D → 256-D
                query_vec   = project_img(img_vec, projector, device)
                feat_matrix = data["img_proj_features"]
            else:
                query_vec   = img_vec
                feat_matrix = data["image_features"]
            valid = True
        else:
            st.warning("Please enter a text query or upload an image.")
            valid = False

        # ── IMAGE PIXEL COLOUR DETECTION (all modes with image) ──────────
        if valid_img and user_image is not None:
            _pixel_col = detect_image_colour(user_image)
            if _pixel_col:
                if not detected_colour:
                    detected_colour = _pixel_col
                    st.caption(f"🎨 Image colour detected → **{_pixel_col}**")
                elif detected_colour != _pixel_col:
                    st.caption(f"🎨 Image colour: **{_pixel_col}** · using text colour: **{detected_colour}**")

        # ── DATASET IMAGE MATCH DETECTION ────────────────────────────────
        # Check if the uploaded image is already present in the dataset.
        # If found, we will pin it as the first result with a score of 1.0.
        # This does NOT alter any cosine similarity or recommendation logic.
        dataset_match_idx = None
        if valid_img and user_image is not None:
            with st.spinner("Checking dataset for exact match..."):
                dataset_match_idx = find_dataset_match(user_image, meta)
            if dataset_match_idx is not None:
                matched_name = str(meta.iloc[dataset_match_idx].get(
                    "productDisplayName", "this item"))[:50]
                st.success(f"✦ Dataset image recognised — pinning **{matched_name}** as top result (score: 1.00)")

        if valid:
            # Raw cosine similarity
            sim_scores = cosine_sim(query_vec, feat_matrix.astype(np.float32))

            # Personalized re-ranking (GAP3 fix)
            if has_pref and pref_weight > 0:
                final_scores = personalize(sim_scores, pref_scores, pref_weight)
            else:
                final_scores = sim_scores

            # ── FILTERS ──────────────────────────────────────
            mask = np.ones(len(meta), dtype=bool)
            if sel_gender   != "All" and "gender"      in meta.columns:
                mask &= meta["gender"].values      == sel_gender
            if sel_category != "All" and "articleType" in meta.columns:
                mask &= meta["articleType"].values == sel_category
            if sel_color    != "All" and "baseColour"  in meta.columns:
                mask &= meta["baseColour"].values  == sel_color
            if sel_brand    != "All" and "brand_tier"  in meta.columns:
                mask &= meta["brand_tier"].values  == sel_brand
            if sel_fabric   != "All" and "fabric"      in meta.columns:
                mask &= meta["fabric"].values      == sel_fabric
            if sel_occasion != "All" and "occasion"    in meta.columns:
                mask &= meta["occasion"].values    == sel_occasion

            # CHANGE 5: Apply max rating filter (<=) instead of min (>=)
            if "rating" in meta.columns:
                mask &= meta["rating"].values <= sel_rating_max

            # ── AUTO COLOUR FILTER ───────────────────────────────────────────
            _apply_col = detected_colour
            if not _apply_col and user_text and "baseColour" in meta.columns:
                _, _art, _apply_col, _gen = expand_query(user_text)
            if (_apply_col and "baseColour" in meta.columns and sel_color == "All"):
                allowed = COLOUR_GROUPS.get(_apply_col, [_apply_col])
                c_mask = np.zeros(len(meta), dtype=bool)
                for c in allowed:
                    c_mask |= (meta["baseColour"].values == c)
                if c_mask.sum() >= 3:
                    mask &= c_mask

            # ── AUTO CATEGORY FILTER (from query text) ──────────────────────
            ART_CANONICAL = {
                'tshirts':'Tshirts','shirts':'Shirts','jeans':'Jeans',
                'trousers':'Trousers','dresses':'Dresses','tops':'Tops',
                'kurtas':'Kurtas','jackets':'Jackets','watches':'Watches',
                'sneakers':'Sneakers','casual shoes':'Casual Shoes',
                'sports shoes':'Sports Shoes','formal shoes':'Formal Shoes',
                'shorts':'Shorts','handbags':'Handbags','sandals':'Sandals',
                'belts':'Belts','wallets':'Wallets','socks':'Socks',
                'track pants':'Track Pants','sweatshirts':'Sweatshirts',
                'sarees':'Sarees','leggings':'Leggings','skirts':'Skirts',
                'backpacks':'Backpacks','sunglasses':'Sunglasses',
                'heels':'Heels','flats':'Flats','boots':'Boots',
            }
            if (user_text and "articleType" in meta.columns and sel_category == "All"):
                _, _art, _col, _gen = expand_query(user_text)
                if _art:
                    canonical = ART_CANONICAL.get(_art.lower())
                    if canonical:
                        a_mask = (meta["articleType"].values == canonical)
                        if a_mask.sum() >= 3:
                            mask &= a_mask

            # ── AUTO GENDER FILTER (from query text) ────────────────────────
            GENDER_DEFAULT_CATS = {
                'Tshirts','Shirts','Jeans','Trousers','Shorts','Sweatshirts',
                'Track Pants','Jackets','Kurtas','Tops','Dresses','Leggings',
                'Skirts','Sarees','Briefs','Socks',
            }
            if "gender" in meta.columns and sel_gender == "All":
                _gen_filter = None
                if detected_gender:
                    _gen_filter = detected_gender
                elif user_text:
                    _, _art2, _col2, _gen2 = expand_query(user_text)
                    if _gen2:
                        _gen_filter = _gen2
                    elif _art2:
                        WOMENS_DEFAULT = {'dresses','Dresses','sarees','Sarees',
                                         'leggings','Leggings','skirts','Skirts',
                                         'heels','Heels','flats','Flats'}
                        if _art2 not in WOMENS_DEFAULT:
                            _gen_filter = "Men"
                if _gen_filter:
                    g_mask = (meta["gender"].values == _gen_filter)
                    g_mask |= (meta["gender"].values == "Unisex")
                    if g_mask.sum() >= 3:
                        mask &= g_mask

            # ── AUTO PATTERN FILTER ─────────────────────────────────────────
            if user_text and "pattern" in meta.columns:
                _PCAN = {
                    'striped':'Striped','stripes':'Striped','stripe':'Striped',
                    'solid':'Solid','plain':'Plain',
                    'printed':'Printed','print':'Printed',
                    'graphic':'Graphic','graphics':'Graphic',
                    'floral':'Floral','flower':'Floral','flowers':'Floral',
                    'checked':'Checked','check':'Checked','checkered':'Checked','plaid':'Checked',
                    'embroidered':'Embroidered','embroidery':'Embroidered',
                    'colourblock':'Colour Block','colorblock':'Colour Block',
                    'camouflage':'Camouflage','camo':'Camouflage',
                    'geometric':'Geometric','abstract':'Abstract',
                    'quilted':'Quilted','distressed':'Distressed',
                }
                _words_lower = clean_text(user_text).split()
                _pat_detected = next((_PCAN[w] for w in _words_lower if w in _PCAN), None)
                if _pat_detected:
                    p_mask = (meta["pattern"].values == _pat_detected)
                    if p_mask.sum() >= 3:
                        mask &= p_mask

            # ── AUTO BRAND FILTER ────────────────────────────────────────────
            if user_text and "brand" in meta.columns:
                _BCAN = {
                    'nike':'Nike','puma':'Puma','adidas':'ADIDAS','reebok':'Reebok',
                    'levis':'Levis','lee':'Lee','wrangler':'Wrangler',
                    'arrow':'Arrow','fastrack':'Fastrack','jockey':'Jockey',
                    'woodland':'Woodland','bata':'Bata','fila':'Fila',
                    'lotto':'Lotto','wildcraft':'Wildcraft','fabindia':'Fabindia',
                    'colorbar':'Colorbar','catwalk':'Catwalk','scullers':'Scullers',
                    'proline':'Proline','hrx':'HRX','spykar':'Spykar',
                    'ucb':'United','benetton':'United','french':'French',
                    'flying':'Flying','jealous':'Jealous','baggit':'Baggit',
                }
                _brand_detected = next((_BCAN[w] for w in clean_text(user_text).split()
                                        if w in _BCAN), None)
                if _brand_detected:
                    b_mask = (meta["brand"].values == _brand_detected)
                    if b_mask.sum() >= 3:
                        mask &= b_mask

            price_active = (price_mode != "No Filter") or (detected_price is not None)
            if has_price and price_active:
                mask &= ((meta["price"].values >= price_lo) &
                         (meta["price"].values <= price_hi))

            masked        = final_scores.copy()
            masked[~mask] = -1.0
            top_idx       = np.argsort(masked)[::-1][:top_k]

            # ── PIN DATASET-MATCHED IMAGE AS FIRST RESULT ────────────────────
            # If the uploaded image was found in the dataset, force it to
            # position 0 with a score of 1.0.  All other results are
            # unchanged — this is purely a display-level reorder.
            if dataset_match_idx is not None:
                # Force the matched item's score to 1.0 in the masked array
                masked[dataset_match_idx] = 1.0
                # Rebuild top_idx: matched item first, then the rest (deduped)
                rest = [i for i in top_idx if i != dataset_match_idx]
                top_idx = np.array([dataset_match_idx] + rest[:top_k - 1])

            # ── AI ANALYSIS ──────────────────────────────────
            if show_analysis:
                attrs = predict_attributes(meta, top_idx, final_scores)
                with st.expander("AI Attribute Analysis", expanded=True):
                    render_analysis(attrs, mode, valid_text, valid_img, has_joint)

            # ── RESULTS ──────────────────────────────────────
            st.markdown('<div style="height:1px;background:#2a2a28;margin:28px 0 24px;"></div>', unsafe_allow_html=True)
            pref_note = (f"<span style='font-family:DM Mono,monospace;font-size:11px;"
                         f"color:#6a6860;'> · personalized · {pref_weight:.0%}</span>"
                         if has_pref and pref_weight > 0 else "")
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:12px;margin-bottom:20px;">
              <span style="font-family:'DM Mono',monospace;font-size:10px;letter-spacing:3px;
                           color:#6a6860;text-transform:uppercase;white-space:nowrap;">
                Top {top_k} Results
              </span>
              <div style="flex:1;height:1px;background:#2a2a28;"></div>
              {pref_note}
            </div>""", unsafe_allow_html=True)

            if not np.any(mask):
                st.warning("No items match these filters. Try relaxing them.")
            else:
                cols = st.columns(3)
                for i, idx in enumerate(top_idx):
                    is_exact = (dataset_match_idx is not None and idx == dataset_match_idx)
                    product_card(cols[i % 3], meta.iloc[idx],
                                 masked[idx], has_price=has_price,
                                 exact_match=is_exact)
                # CHANGE 4: Pass scores list to add_to_history
                top_scores = [float(masked[idx]) for idx in top_idx]
                add_to_history(meta.iloc[top_idx], scores=top_scores)

            # Price tier tabs (only when no explicit price filter)
            if has_price and not price_active:
                st.markdown('<div style="height:1px;background:#2a2a28;margin:28px 0 24px;"></div>', unsafe_allow_html=True)
                st.markdown("""
                <div style="display:flex;align-items:center;gap:12px;margin-bottom:20px;">
                  <span style="font-family:'DM Mono',monospace;font-size:10px;letter-spacing:3px;
                               color:#6a6860;text-transform:uppercase;white-space:nowrap;">Shop by Price</span>
                  <div style="flex:1;height:1px;background:#2a2a28;"></div>
                </div>""", unsafe_allow_html=True)
                for tab, (_, (lo, hi)) in zip(
                        st.tabs(list(PRICE_BUDGETS.keys())),
                        PRICE_BUDGETS.items()):
                    with tab:
                        tm = mask & (meta["price"].values >= lo) & (meta["price"].values <= hi)
                        ts = final_scores.copy(); ts[~tm] = -1.0
                        ti = np.argsort(ts)[::-1][:3]
                        if not np.any(tm) or ts[ti[0]] < 0:
                            cat_hint = detected_article.capitalize() if (user_text and detected_article) else "items"
                            st.info(f"No {cat_hint} available in this price range.")
                        else:
                            tc = st.columns(3)
                            for j, tidx in enumerate(ti):
                                if ts[tidx] >= 0:
                                    product_card(tc[j], meta.iloc[tidx], ts[tidx])

            # ── OUTFIT INTELLIGENCE ──────────────────────────────
            if show_outfits and len(top_idx) > 0:
                top_item    = meta.iloc[top_idx[0]]
                top_art     = str(top_item.get("articleType", ""))
                top_colour  = str(top_item.get("baseColour", ""))
                top_gender  = str(top_item.get("gender", ""))
                sections    = get_outfit_sections(top_art)

                if sections:
                    st.markdown('<div style="height:1px;background:#2a2a28;margin:28px 0 24px;"></div>', unsafe_allow_html=True)
                    st.markdown("""
                    <div style="display:flex;align-items:center;gap:12px;margin-bottom:16px;">
                      <span style="font-family:'DM Mono',monospace;font-size:10px;letter-spacing:3px;
                                   color:#6a6860;text-transform:uppercase;white-space:nowrap;">Style It</span>
                      <div style="flex:1;height:1px;background:#2a2a28;"></div>
                      <span style="font-family:'Playfair Display',serif;font-size:20px;
                                   font-weight:700;color:#f0ede8;">Complete the Outfit</span>
                    </div>""", unsafe_allow_html=True)

                    # Colour harmony hint
                    harmony_cols = COLOUR_HARMONY.get(top_colour, [])
                    if harmony_cols and top_colour != "Unknown":
                        chips = "".join([
                            f'<span style="font-family:DM Mono,monospace;font-size:10px;'
                            f'letter-spacing:0.5px;padding:4px 12px;'
                            f'background:#111110;border:1px solid #2a2a28;border-radius:2px;'
                            f'color:#6a6860;margin:3px 4px 3px 0;display:inline-block;">{c}</span>'
                            for c in harmony_cols[:7]])
                        st.markdown(
                            f'<div style="margin-bottom:16px;">'
                            f'<span style="font-family:DM Mono,monospace;font-size:9px;'
                            f'letter-spacing:2px;color:#6a6860;text-transform:uppercase;'
                            f'margin-right:10px;">Colour Harmony</span>{chips}</div>',
                            unsafe_allow_html=True)

                    # Role priority order for tabs
                    role_order = ["Bottom", "Top", "Footwear",
                                  "Accessory", "Bag", "Layer", "Outerwear"]
                    active_roles = [r for r in role_order if r in sections]
                    for r in sections:
                        if r not in active_roles:
                            active_roles.append(r)

                    if active_roles:
                        tab_labels = [
                            f"{OUTFIT_ROLE_META.get(r, ('🏷️',''))[0]} {OUTFIT_ROLE_META.get(r, ('','Items'))[1]}"
                            for r in active_roles
                        ]
                        tabs = st.tabs(tab_labels)

                        # CHANGE 2 & 3: Uniform image height using contain; Bags & Layer get up to 4 items
                        OUTFIT_IMG_HEIGHT = 260  # fixed height for all outfit cards

                        for tab, role in zip(tabs, active_roles):
                            with tab:
                                type_list = sections[role]
                                # CHANGE 3: Bags and Layer sections show up to 4 items
                                max_items = 4 if role in ("Bag", "Layer") else 4
                                cols_data = []
                                for atype in type_list[:8]:  # scan more types to fill slots
                                    picked = pick_outfit_item(
                                        meta, atype, top_colour,
                                        has_price, top_gender)
                                    if len(picked) > 0:
                                        cols_data.append((atype, picked.iloc[0]))
                                    if len(cols_data) >= max_items:
                                        break

                                if cols_data:
                                    oc = st.columns(len(cols_data))
                                    for ci, (atype, item_row) in enumerate(cols_data):
                                        with oc[ci]:
                                            st.markdown(
                                                f"<div style='font-family:DM Mono,monospace;"
                                                f"font-size:9px;letter-spacing:2px;color:#6a6860;"
                                                f"text-transform:uppercase;margin-bottom:4px;"
                                                f"text-align:center;'>{atype}</div>",
                                                unsafe_allow_html=True)
                                            product_card(
                                                oc[ci], item_row, 0.0,
                                                has_price=has_price,
                                                fixed_img_height=OUTFIT_IMG_HEIGHT)
                                else:
                                    st.info(
                                        f"No {role.lower()} items found in the dataset "
                                        f"for this outfit. Try a different search.")

    # ── RECENTLY VIEWED ──────────────────────────────────────
    if show_history and st.session_state.recently_viewed:
        st.markdown('<div style="height:1px;background:#2a2a28;margin:36px 0 28px;"></div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:20px;">
          <span style="font-family:'DM Mono',monospace;font-size:10px;letter-spacing:3px;
                       color:#6a6860;text-transform:uppercase;white-space:nowrap;">Your History</span>
          <div style="flex:1;height:1px;background:#2a2a28;"></div>
          <span style="font-family:'Playfair Display',serif;font-size:20px;
                       font-weight:700;color:#f0ede8;">Recently Viewed</span>
        </div>""", unsafe_allow_html=True)
        rv = st.session_state.recently_viewed[:6]
        rc = st.columns(min(6, len(rv)))
        for col, item in zip(rc, rv):
            with col:
                if os.path.exists(item["path"]):
                    st.image(item["path"], use_container_width=True)
                # CHANGE 4: Show brand, name, price AND similarity score in history
                brand_html = (
                    f'<div style="font-family:DM Mono,monospace;font-size:9px;'
                    f'letter-spacing:1px;color:#6a6860;text-transform:uppercase;'
                    f'margin-top:6px;">{item["brand"]}</div>'
                )
                name_html = (
                    f'<div style="font-size:12px;color:#9a9890;line-height:1.3;">'
                    f'{item["name"][:28]}</div>'
                )
                price_score_parts = []
                if item.get("price"):
                    price_score_parts.append(
                        f'<span style="font-family:DM Mono,monospace;color:#c8a96e;'
                        f'font-size:13px;">₹{int(item["price"]):,}</span>'
                    )
                if item.get("score", 0) > 0:
                    price_score_parts.append(
                        f'<span style="font-family:DM Mono,monospace;color:#6a6860;'
                        f'font-size:11px;margin-left:6px;">{item["score"]:.2f}</span>'
                    )
                price_score_html = (
                    f'<div style="margin-top:2px;display:flex;align-items:baseline;gap:4px;">'
                    + "".join(price_score_parts)
                    + '</div>'
                ) if price_score_parts else ""

                st.markdown(brand_html + name_html + price_score_html, unsafe_allow_html=True)

        st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)
        if st.button("Clear History"):
            st.session_state.recently_viewed = []
            st.rerun()

    # ── FOOTER ───────────────────────────────────────────────
    st.markdown("""
    <div style="height:1px;background:#2a2a28;margin:48px 0 0;"></div>
    <footer style="display:flex;align-items:center;justify-content:space-between;
                   padding:20px 0 32px;flex-wrap:wrap;gap:20px;">
      <div style="font-family:'DM Mono',monospace;font-size:10px;
                  color:#6a6860;letter-spacing:1px;">
        StyleSense AI · CNN+RNN Multimodal System · Myntra Fashion Dataset
      </div>
      <div style="display:flex;gap:32px;">
        <div style="text-align:right;">
          <div style="font-family:'Playfair Display',serif;font-size:18px;color:#c8a96e;">44K+</div>
          <div style="font-family:'DM Mono',monospace;font-size:9px;color:#6a6860;
                      letter-spacing:1px;text-transform:uppercase;">Products</div>
        </div>
        <div style="text-align:right;">
          <div style="font-family:'Playfair Display',serif;font-size:18px;color:#c8a96e;">142</div>
          <div style="font-family:'DM Mono',monospace;font-size:9px;color:#6a6860;
                      letter-spacing:1px;text-transform:uppercase;">Categories</div>
        </div>
        <div style="text-align:right;">
          <div style="font-family:'Playfair Display',serif;font-size:18px;color:#c8a96e;">256-D</div>
          <div style="font-family:'DM Mono',monospace;font-size:9px;color:#6a6860;
                      letter-spacing:1px;text-transform:uppercase;">Joint Space</div>
        </div>
      </div>
    </footer>""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()