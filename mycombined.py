# right_side_zonal_extractor.py
import os
import json
import uuid
import logging
from pathlib import Path
from typing import Tuple, Dict

import fitz  # PyMuPDF
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pytesseract
from pyzbar.pyzbar import decode
from rembg import remove
import re
import random


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("zonal-right")

# ---------------- CONFIG ----------------
CONFIG = {
    # If tesseract not on PATH, set full path:
    "TESSERACT_CMD": r"C:\Program Files\Tesseract-OCR\tesseract.exe",

    # Input/Output
    "OUT_DIR": "out_right",
    "DPI": 300,  # rendering DPI (300 is faster; increase to 600 if needed)

    # Template image to place extracted fields on (optional)
    # If missing, a blank 1024x640 white image will be used.
    "TEMPLATE_IMAGE":"template0.jpg",  # e.g. "my_template_1024x640.png"
    "PHOTO_POS": (53, 111), "PHOTO_SIZE": (174, 230),
    "PHOTO_POS_SMALL": (522, 307), "PHOTO_SIZE_SMALL": (56, 65),
    "QR_POS": (933, 9), "QR_SIZE": (339, 336),
    "BARCODE_POS": (296, 313), "BARCODE_SIZE": (167, 62),

    # Where to place each field on output template (x, y, max_width, font_size)
    # You can change these target positions to match your design.
    "FIELD_TARGETS": {
        "Name":       (252, 108, 300, 20),
        "DOB":        (252, 186, 300, 20),
        "Sex":        (252, 227, 300, 20),
        "DOE":        (252, 273, 300, 20),
        "Phone":      (669, 40, 148, 18),
        "Address":    (669, 141, 200, 18),
        "Fin":        (757, 316, 200, 18),
        "Doing":      (2, 80, 40, 13),  # vertical field (will be drawn bottom-to-top)
        "Serial":     (1171, 353, 100, 16),
    },

    # Your supplied right-side zones (x1,y1,x2,y2) measured on the PNG produced at DPI above
    # (I used the values you provided)
    "ZONES": {
        "Name":   (698, 907, 1224, 998),
        "DOB":    (1727, 1043, 2019, 1079),
        "Sex":    (224, 1300, 442, 1395),
        "DOE":    (1727, 1148, 2069, 1197),     # date of expiry maybe
        "Phone":  (1707, 2065, 1867, 2101),
        "Address":(1694, 2199, 2208, 2386),
        "Fin":    (2067, 2053, 2252, 2090),
        "Doing":  (2232, 535, 2278, 878),       # narrow vertical region
        #"Sn":    (288, 939, , 18),         # seems out-of-right-area but included
    },

    # Debug: save crops & variant images
    "SAVE_DEBUG_CROPS": True,
    "DEBUG_DIR": "debug_vertical123",

    # Tesseract configs
    "OCR_LANG_AMH_ENG": "amh+eng",
    "OCR_LANG_ENG": "eng",
    "OCR_CONFIG_NUM": "--oem 1 --psm 7 -c tessedit_char_whitelist=0123456789/|-abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",
    "OCR_CONFIG_DEFAULT": "--oem 1 --psm 6",
    "OCR_CONFIG_ADD": "--oem 3 --psm 7" 
    
}

# set tesseract path if provided
if CONFIG["TESSERACT_CMD"]:
    pytesseract.pytesseract.tesseract_cmd = CONFIG["TESSERACT_CMD"]

os.makedirs(CONFIG["OUT_DIR"], exist_ok=True)
if CONFIG["SAVE_DEBUG_CROPS"]:
    os.makedirs(CONFIG["DEBUG_DIR"], exist_ok=True)

# ---------------- util helpers ----------------
#def generate_sn():
    #return f"{random.randint(0,9999999):07d}"

def pdf_to_png(pdf_path: str, out_path: str, dpi: int = 300) -> str:
    """Render first page of PDF to PNG and return path"""
    logger.info("Rendering PDF to PNG at %d DPI...", dpi)
    doc = fitz.open(pdf_path)
    if doc.page_count == 0:
        raise ValueError("Empty PDF")
    page = doc.load_page(0)
    mat = fitz.Matrix(dpi/72.0, dpi/72.0)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    pix.save(out_path)
    logger.info("Saved rendered page to %s", out_path)
    doc.close()
    return out_path

def crop_from_cv(img_cv, box: Tuple[int,int,int,int]) -> np.ndarray:
    x1,y1,x2,y2 = box
    h, w = img_cv.shape[:2]
    # clamp
    x1c = max(0, min(w-1, int(x1)))
    y1c = max(0, min(h-1, int(y1)))
    x2c = max(0, min(w, int(x2)))
    y2c = max(0, min(h, int(y2)))
    if x2c <= x1c or y2c <= y1c:
        raise ValueError(f"Invalid crop box {box} for image size {(w,h)}")
    return img_cv[y1c:y2c, x1c:x2c]

def preprocess_crop(cv_crop: np.ndarray) -> Image.Image:
    """Preprocess a crop (cv image) into PIL grayscale/binarized image for OCR."""
    # Convert to gray
    gray = cv2.cvtColor(cv_crop, cv2.COLOR_BGR2GRAY) if len(cv_crop.shape)==3 else cv_crop

    # quick resize if very small (makes OCR more reliable)
    h,w = gray.shape[:2]
    if h < 50 or w < 50:
        gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        # CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)

    # small blur to reduce textured backgrounds
    blur = cv2.GaussianBlur(enhanced, (3,3), 0)

    # Otsu binarization
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # optionally dilate a bit to thicken strokes
    kernel = np.ones((2,2), np.uint8)
    thick = cv2.dilate(th, kernel, iterations=1)

    return Image.fromarray(thick)

def ocr_try_variants(pil_img: Image.Image, langs=("eng","amh+eng"), configs=None, debug_prefix=None):
    """
    Try OCR on pil_img using different languages/configs, return best text (simple heuristic)
    """
    best_text = ""
    best_score = -1.0
    best_meta = None

    if configs is None:
        configs = [CONFIG["OCR_CONFIG_DEFAULT"], CONFIG["OCR_CONFIG_NUM"]]

    for lang in langs:
        for cfg in configs:
            try:
                txt = pytesseract.image_to_string(pil_img, lang=lang, config=cfg).strip()
                # score: prefer longer alnum content; prefer digits if expecting dates/nums (we may boost later)
                alnum = sum(1 for c in txt if c.isalnum())
                total = max(1, len(txt))
                ratio = alnum/total
                score = ratio * 0.6 + (len(txt)/200.0) * 0.4
            except Exception as e:
                txt = ""
                score = 0
            # store best
            if score > best_score:
                best_score = score
                best_text = txt
                best_meta = {"lang": lang, "config": cfg, "score": score}
    # debug save
    if debug_prefix and CONFIG["SAVE_DEBUG_CROPS"]:
        try:
            dpath = os.path.join(CONFIG["DEBUG_DIR"], debug_prefix + ".txt")
            with open(dpath, "w", encoding="utf-8") as f:
                f.write(best_text + "\n\nMETA:\n" + json.dumps(best_meta, ensure_ascii=False, indent=2))
        except Exception:
            pass
    return best_text, best_meta

def ocr_vertical_crop(pil_crop: Image.Image, debug_name: str):
    # Convert to grayscale and rotate to horizontal for OCR
    cv_img = cv2.cvtColor(np.array(pil_crop), cv2.COLOR_RGB2GRAY)
    rotated = cv2.rotate(cv_img, cv2.ROTATE_90_CLOCKWISE)
    
    # Standard preprocessing
    upscaled = cv2.resize(rotated, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    _, thresh = cv2.threshold(upscaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    prep_pil = Image.fromarray(thresh)
    
    # Use whitelist to ensure '|' is recognized
    custom_config = r'--oem 1 --psm 7 -c tessedit_char_whitelist=0123456789JanFebMarAprMayJunJulAugSepOctNovDec/|- '
    text = pytesseract.image_to_string(prep_pil, lang="eng", config=custom_config).strip()
    
    return text, None
def deskew_image(cv_img):
    """Detects text angle and straightens the image."""
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    # Use Canny to find edges
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    # Use HoughLinesP to find lines
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
    
    angles = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            # Filter out near-vertical lines
            if -45 < angle < 45:
                angles.append(angle)
    
    if len(angles) > 0:
        median_angle = np.median(angles)
        (h, w) = cv_img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        rotated = cv2.warpAffine(cv_img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        logger.info(f"Deskewed image by {median_angle:.2f} degrees")
        return rotated
    
    return cv_img
def format_sex(text):
    """Robustly separates Amharic and English regardless of spacing or noise."""

    amh_part = "".join(re.findall(r'[\u1200-\u137F]+', text))
    # Find all English letters
    eng_part = "".join(re.findall(r'[a-zA-Z0-9]+', text))
    
    #if amh_part and eng_part:
    return f"{amh_part} | {eng_part}"
    # If regex fails, return the original text so we don't lose data
   # return text


def format_date(text):
    """Converts 12/Dec/2000 to 12/12/2000."""
    months = {
        'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04', 'May': '05', 'Jun': '06',
        'Jul': '07', 'Aug': '08', 'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'
    }
    # Standardize separators to /
    clean_date = text.replace('*', '/').replace('-', '/')
    for name, num in months.items():
        if name in clean_date:
            return clean_date.replace(name, num)
    return clean_date
    

# ---------------- core extraction ----------------

def extract_right_side_fields(pdf_path: str) -> Dict[str, str]:
    uid = uuid.uuid4().hex[:8]
    out_png = os.path.join(CONFIG["OUT_DIR"], f"render_{uid}.png")
    png_path = pdf_to_png(pdf_path, out_path=out_png, dpi=CONFIG["DPI"])

    img_cv = cv2.imread(png_path)
    if img_cv is None:
        raise RuntimeError("Failed to read rendered PNG: " + png_path)
    
    # Straighten the page before cropping
    img_cv = deskew_image(img_cv)

    results = {}
    zones = CONFIG["ZONES"]

    for fname, box in zones.items():
        try:
            crop_cv = crop_from_cv(img_cv, box)
        except Exception as e:
            results[fname] = ""
            continue

        text = ""
        # --- 1. SPECIAL CASE: ADDRESS ---
        if fname.lower() == "address":
            gray = cv2.cvtColor(crop_cv, cv2.COLOR_BGR2GRAY)
            # Higher contrast for addresses
            adjusted = cv2.convertScaleAbs(gray, alpha=1.7, beta=-30)
            # Upscale 2x for clearer Amharic/English characters
            upscaled = cv2.resize(adjusted, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            pil_prep = Image.fromarray(upscaled)
            
            # Using PSM 6 (Assume a single uniform block of text) and OEM 3 (LSTM)
            # We use image_to_string directly here to apply specific config
            text = pytesseract.image_to_string(pil_prep, lang="amh+eng", config="--oem 1 --psm 6").strip()

        # --- 2. SPECIAL CASE: DOE (Date of Expiry) ---
        elif fname.lower() == "doe":
            pil_prep = preprocess_crop(crop_cv)
            # FIX: Deny the letter 'w' and whitelist date characters/months
            # This forces 'Now' to become 'Nov'
            date_cfg = "--oem 1 --psm 7 -c tessedit_char_whitelist=0123456789JanFebMarAprMayJunJulAugSepOctNovDec/|- "
            text = pytesseract.image_to_string(pil_prep, lang="eng", config=date_cfg).strip()

        # --- 3. VERTICAL FIELDS ---
        elif "doing" in fname.lower():
            pil_crop = Image.fromarray(cv2.cvtColor(crop_cv, cv2.COLOR_BGR2RGB))
            text, _ = ocr_vertical_crop(pil_crop, debug_name=f"{fname}_{uid}")


        # --- 4. ALL OTHER FIELDS ---
      
        else:
            pil_prep = preprocess_crop(crop_cv)
            lang = CONFIG["OCR_LANG_ENG"] if fname.lower() in ("fin","dob", "phone") else CONFIG["OCR_LANG_AMH_ENG"]
            cfg = CONFIG["OCR_CONFIG_NUM"] if fname.lower() in ("fin","dob" "phone") else CONFIG["OCR_CONFIG_DEFAULT"]
            
            text, _ = ocr_try_variants(pil_prep, langs=(lang,), configs=[cfg])

        # Save debug crops
        if CONFIG["SAVE_DEBUG_CROPS"]:
            cv2.imwrite(os.path.join(CONFIG["DEBUG_DIR"], f"{fname}_{uid}.png"), crop_cv)

        cleaned = text.strip().replace("\n", " ").replace("\r", " ").strip()
        if fname.lower()== "Sex":
            results[fname] = format_sex(cleaned)
        elif fname.lower() == "dob":
            results[fname] = format_date(cleaned)
        else:
            results[fname] = cleaned
    results["Serial"]= str(random.randint(1000000, 9999999))

    # ... (Save JSON/TXT logic remains the same)
    return results

    # save results JSON & text
    json_path = os.path.join(CONFIG["OUT_DIR"], f"fields_{uid}.json")
    txt_path = os.path.join(CONFIG["OUT_DIR"], f"fields_{uid}.txt")
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(results, jf, ensure_ascii=False, indent=2)
    with open(txt_path, "w", encoding="utf-8") as tf:
        for k,v in results.items():
            tf.write(f"{k}: {v}\n")

    logger.info("Saved results to %s and %s", json_path, txt_path)
    return results

# ---------------- optional template placer ----------------
def draw_vertical_bottom_up(draw: ImageDraw.ImageDraw, x: int, y_bottom: int, text: str, font: ImageFont.ImageFont):
    for ch in text:
        bbox = draw.textbbox((0,0), ch, font=font)
        h = bbox[3] - bbox[1]
        y_bottom -= h
        draw.text((x, y_bottom), ch, font=font, fill="black")


def get_stacked_lines(text: str):
    """
    Groups alternating Amharic and English/Numeric blocks into a list of lines.
    Input: "አዲስ አበባ Addis Ababa ወረዳ 09"
    Output: ["አዲስ አበባ", "Addis Ababa", "ወረዳ", "09"] -> actually we group 09 with English
    """
    # Pattern 1: Matches one or more Ethiopic characters (including spaces between them)
    # Pattern 2: Matches Latin letters, numbers, and symbols
    pattern = r'([\u1200-\u137F0-9\s]+)|([a-zA-Z0-9\s.,/-]+)'
    
    matches = re.findall(pattern, text)
    lines = []
    for amh, eng in matches:
        if amh.strip():
            lines.append(amh.strip())
        if eng.strip():
            lines.append(eng.strip())
    return lines
def extract_and_paste_images(doc, template_path):
    """Strict Index-Based Extraction: 0=Photo, 1=QR, 2=Barcode"""
    page = doc[0]
    images = page.get_images(full=True)
    template = Image.open(template_path).convert("RGBA")
    
    # --- STABLE CONVERSION HELPER ---
    def get_pil_from_pdf(index):
        if len(images) <= index:
            logger.warning(f"⚠️ Index {index} not found in PDF images.")
            return None
        try:
            pix = fitz.Pixmap(doc, images[index][0])
            # Handle grayscale or CMYK by forcing RGB conversion
            if pix.n != 3:
                pix = fitz.Pixmap(fitz.csRGB, pix)
            
            # Use frombytes which is more stable than frombuffer for PDF samples
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            return img
        except Exception as e:
            logger.error(f"❌ Failed to extract image at index {index}: {e}")
            return None

    # 1. USER PHOTO (Strictly Index 0)
    photo_pil = get_pil_from_pdf(0)
    if photo_pil:
        # LOGGING ONLY: Check if it's colored for your info
        img_np = np.array(photo_pil)
        color_score = np.std(img_np[:,:,0]-img_np[:,:,1]) + np.std(img_np[:,:,0]-img_np[:,:,2])
        logger.info(f"✅ Extracted Index 0 (Color Score: {color_score:.2f})")
        
        # Remove background and paste
        photo_no_bg = remove(photo_pil)
        template.paste(photo_no_bg.resize(CONFIG["PHOTO_SIZE"]), CONFIG["PHOTO_POS"], photo_no_bg.resize(CONFIG["PHOTO_SIZE"]))
        template.paste(photo_no_bg.resize(CONFIG["PHOTO_SIZE_SMALL"]), CONFIG["PHOTO_POS_SMALL"], photo_no_bg.resize(CONFIG["PHOTO_SIZE_SMALL"]))

    # 2. QR CODE (Strictly Index 1)
    qr_pil = get_pil_from_pdf(1)
    if qr_pil:
        template.paste(qr_pil.resize(CONFIG["QR_SIZE"]), CONFIG["QR_POS"])
        logger.info("✅ Extracted Index 1 (QR)")

    # 3. BARCODE (Strictly Index 2)
    bc_pil = get_pil_from_pdf(2)
    if bc_pil:
        img_cv = cv2.cvtColor(np.array(bc_pil), cv2.COLOR_RGB2BGR)
        detected = decode(img_cv)
        
        if detected:
            for obj in detected:
                if obj.type != "QRCODE":
                    x, y, w, h = obj.rect
                    # Use your expansion values
                    cropped = bc_pil.crop((max(0, x-25), max(0, y-80), min(bc_pil.width, x+w+60), min(bc_pil.height, y+h+20)))
                    template.paste(cropped.resize(CONFIG["BARCODE_SIZE"], Image.LANCZOS), CONFIG["BARCODE_POS"])
                    logger.info("✅ Extracted & Cropped Index 2 (Barcode)")
                    break
        else:
            # Failsafe: Paste raw if decoding fails
            template.paste(bc_pil.resize(CONFIG["BARCODE_SIZE"]), CONFIG["BARCODE_POS"])
            logger.info("✅ Extracted Index 2 (Pasted raw barcode image)")
                
    return template
def add_watermark(img: Image.Image, text="PREVIEW", opacity=60):
    watermark = Image.new("RGBA", img.size, (0,0,0,0))
    draw = ImageDraw.Draw(watermark)

    try:
        font = ImageFont.truetype("arial.ttf", 80)
    except:
        font = ImageFont.load_default()

    w, h = img.size
    for y in range(0, h, 200):
        for x in range(0, w, 400):
            draw.text((x, y), text, fill=(255, 0, 0, opacity), font=font)

    return Image.alpha_composite(img.convert("RGBA"), watermark)

def place_on_template(pdf_path: str, extracted: Dict[str,str],out_image_path: str, watermark=False):
    doc = fitz.open(pdf_path)
    tpl = extract_and_paste_images(doc, CONFIG["TEMPLATE_IMAGE"])
    draw = ImageDraw.Draw(tpl)
    try:
        font_path = "NotoSansEthiopic_ExtraCondensed-Black.ttf"
        base_font = ImageFont.truetype(font_path, 24)
    except:
        base_font = ImageFont.load_default()

    for field, target in CONFIG["FIELD_TARGETS"].items():
        x, y, maxw, font_size = target
        text = extracted.get(field, "").strip()

        if not text:
            continue

        try:
            font = ImageFont.truetype(font_path, font_size)
        except:
            font = base_font

        # 1. Handle Vertical Field
        if field.lower() == "doing":
            if 'O0ct' in text:
                 text = text.replace('O0ct','Oct')
            parts = [p.strip() for p in text.split('|')]
            parts.reverse()
            
            # Define starting Y for each part
            # Part 1 (Top): starts at the base 'y'
            # Part 2 (Bottom): starts 150 pixels lower (adjust as needed)
            y_offsets = [0, 169] 
            
            for i, part in enumerate(parts):
                if not part or i >= len(y_offsets): 
                    continue
                
                # Measure and create canvas
                txt_len = draw.textlength(part, font=font)
                temp_txt_img = Image.new("RGBA", (int(txt_len) + 10, font_size + 10), (0, 0, 0, 0))
                temp_draw = ImageDraw.Draw(temp_txt_img)
                temp_draw.text((0, 0), part, fill="black", font=font)
                
                # Rotate 90 degrees (Bottom-to-Top orientation)
                rotated_part = temp_txt_img.rotate(90, expand=True)
                
                # Calculate the specific Y for this part
                part_y = y + y_offsets[i]
                
                # Paste onto template
                tpl.alpha_composite(rotated_part, (x, part_y))
            continue
        # 2. Apply Stacked Logic ONLY for Name and Address
        if field.lower() in ["name", "address"]:
            lines_to_draw = get_stacked_lines(text)
            curr_y = y
            for block in lines_to_draw:
                words = block.split()
                line = ""
                for w in words:
                    trial = (line + " " + w).strip()
                    if draw.textlength(trial, font=font) <= maxw:
                        line = trial
                    else:
                        draw.text((x, curr_y), line, fill="black", font=font)
                        curr_y += font_size + 2
                        line = w
                if line:
                    draw.text((x, curr_y), line, fill="black", font=font)
                    curr_y += font_size + 4 

        # 3. Simple Wrap for all other fields (DOB, Sex, Phone, etc.)
        elif field.lower() == "sex":
            words = text.split()
            separate = ""
            line = ""
            curr_y = y
            for w in words:
                trial = (line + " " + w).strip()
                if draw.textlength(trial, font=font) <= maxw:
                    line = trial
                else:
                    draw.text((x, curr_y), line, fill="black", font=font)
                    curr_y += font_size + 4
                    line = w
            if line:
                separate = format_sex(line)
                draw.text((x, curr_y), separate, fill="black", font=font)
        elif field.lower() == ["dob", "doe"]:
            if '0ct' in text:
                 text = text.replace('0ct','Oct')
        else:
            words = text.split()
            line = ""
            curr_y = y
            for w in words:
                trial = (line + " " + w).strip()
                if draw.textlength(trial, font=font) <= maxw:
                    line = trial
                else:
                    draw.text((x, curr_y), line, fill="black", font=font)
                    curr_y += font_size + 4
                    line = w
            if line:
                draw.text((x, curr_y), line, fill="black", font=font)
    if watermark:
        tpl=add_watermark(tpl,"PREVIEW NOT-PAID")
        
        tpl.save(out_image_path)
        #tpl = tpl.transpose(Image.FLIP_LEFT_RIGHT)
   
    os.makedirs(os.path.dirname(out_image_path),exist_ok=True)
    tpl.save(out_image_path)
    print(f"[TEMPLATE] Processed {field} fields and saved: {out_image_path}")

# ---------------- CLI runner ----------------

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python right_side_zonal_extractor.py <Thomas.pdf> [--place-template]")
        sys.exit(1)
    pdf_in = sys.argv[1]
    place = "--place-template" in sys.argv
    results = extract_right_side_fields(pdf_in)
    if place:
        out_img = os.path.join(CONFIG["OUT_DIR"], "placed_" + Path(pdf_in).stem + ".png")
        place_on_template(results, out_img)
    print("Done. Check folder:", CONFIG["OUT_DIR"])

