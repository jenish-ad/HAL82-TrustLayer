# import re
# from rapidfuzz import process

# class KycCleaner:
#     def __init__(self):
#         self.template_keywords = ["Municipality", "Muncipality", "Ward No", "District", "Province", "Year", "Month", "Day"]
#         self.months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec","JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"]
        
#         self.nepal_districts = [
#             "Achham", "Arghakhanchi", "Baglung", "Baitadi", "Bajhang", "Bajura", "Banke", "Bara", "Bardiya",
#             "Bhaktapur", "Bhojpur", "Chitwan", "Dadeldhura", "Dailekh", "Dang", "Darchula", "Dhading", 
#             "Dhankuta", "Dhanusa", "Dolakha", "Dolpa", "Doti", "Gorkha", "Gulmi", "Humla", "Ilam", "Jajarkot",
#             "Jhapa", "Jumla", "Kailali", "Kalikot", "Kanchanpur", "Kapilvastu", "Kaski", "Kathmandu", 
#             "Kavrepalanchok", "Khotang", "Lalitpur", "Lamjung", "Mahottari", "Makwanpur", "Manang", "Morang",
#             "Mugu", "Mustang", "Myagdi", "Nawalpur", "Nuwakot", "Okhaldhunga", "Palpa", "Panchthar", "Parbat", 
#             "Parsa", "Parasi", "Pyuthan", "Ramechhap", "Rasuwa", "Rautahat", "Rolpa", "Rukum East", "Rukum West",
#             "Rupandehi", "Salyan", "Sankhuwasabha", "Saptari", "Sarlahi", "Sindhuli", "Sindhupalchok", "Siraha", 
#             "Solukhumbu", "Sunsari", "Surkhet", "Syangja", "Tanahu", "Taplejung", "Terhathum", "Udayapur"
#         ]

#     def clean(self, field, text, conf=1.0):
#         if not text:
#             if field == "permanent-address":
#                 return "[PARTIAL]"
#             return "[NOT_DETECTED]" if field != "officer_signature" else "[MISSING]"

#         text = text.strip()

#         # 1. Signature Logic
#         if field == "officer-signature":
#             return "[SIGNED]"

#         # 2. Citizenship Number
#         if field == "citisenship-number":
#             text = text.upper()
#             text = (
#                 text.replace('O', '0')
#                     .replace('I', '1')
#                     .replace('S', '5')
#                     .replace('(', '1')
#                     .replace('L', '1')
#                     .replace('Z', '2')
#             )

#             digits = re.findall(r'\d+', text)
#             number = "".join(digits)
            
#             if 10 <= len(number) <= 14:
#                 return number

#             return "[INVALID]"
            
#         # 3. Date of Birth Logic
#         if field == "DOB":
#             # PREPROCESSING: Fix common OCR errors FIRST
#             text = text.replace('_', '').replace('|', '')  # Remove noise
            
#             # Fix common year OCR errors
#             text = text.replace('L', '1').replace('l', '1').replace('O', '0') 
#             text = text.replace('/9', '19')  # /977 -> 1977
#             text = text.replace('l9', '19')  # l977 -> 1977
#             text = text.replace('I9', '19')  # I977 -> 1977
#             #text = text.replace('/0', '20')  # /007 -> 2007
#             text = text.replace('~', '')     # Remove ~ symbol
            
#             # Try to parse structured format first (Year: XXXX Month: XXX Day: XX)
#             structured_match = re.search(r'Year:\s*(\d{4})?\s*Month:\s*(\w+)?\s*Day:\s*(\d{1,2})?', text, re.IGNORECASE)
            
#             if structured_match:
#                 res_year = structured_match.group(1) if structured_match.group(1) else "????"
#                 month_text = structured_match.group(2) if structured_match.group(2) else None
#                 res_day = structured_match.group(3) if structured_match.group(3) else "??"
                
#                 # Match month name
#                 if month_text:
#                     # Clean month text of any non-alphabetic characters
#                     month_text = re.sub(r'[^A-Za-z]', '', month_text)
#                     month_match = process.extractOne(month_text, self.months, score_cutoff=60)
#                     res_month = month_match[0] if month_match else "???"
#                 else:
#                     res_month = "???"
#             else:
#                 # Fallback to your original pattern matching
#                 year = re.search(r'(?:19|20)\d{2}', text)
#                 day = re.search(r'\b([1-9]|[0-2]\d|3[01])\b', text)
#                 month_match = process.extractOne(text, self.months, score_cutoff=60)

#                 res_year = year.group() if year else "????"
#                 res_month = month_match[0] if month_match else "???"
#                 res_day = day.group() if day else "??"

#             return f"Year: {res_year} Month: {res_month} Day: {res_day}"
                
        
#         # 4. Permanent Address Logic (FIXED)
#         if field == "permanent-address":
#             text = text.replace("Muncipality", "Municipality")
            
#             # WARD - Handle multiple colon patterns
#             ward = "?"
#             # Clean up multiple colons before matching
#             clean_text = re.sub(r':+', ':', text)  # Replace :: with :
            
#             # Try multiple ward patterns
#             ward_patterns = [
#                 r'Ward\s*No\.?\s*:?\s*(\d{1,2})',  # Ward No: 8 or Ward No:: 8
#                 r'Ward\s+(\d{1,2})',                # Ward 8
#                 r'No\.?\s*:?\s*(\d{1,2})',          # No: 8 or No:: 8
#             ]
            
#             for pattern in ward_patterns:
#                 ward_match = re.search(pattern, clean_text, re.I)
#                 if ward_match:
#                     ward = ward_match.group(1)
#                     break

#             # DISTRICT - Check each word individually with strict matching
#             district = "UNKNOWN"
            
#             # First try to find district with "District:" pattern
#             district_pattern = r'District\s*[:\-]?\s*([A-Za-z]{3,})'
#             district_match = re.search(district_pattern, text, re.I)
            
#             if district_match:
#                 potential_district = re.sub(r'[^A-Za-z]', '', district_match.group(1))
#                 # Verify it's actually a district name
#                 verify_match = process.extractOne(
#                     potential_district, self.nepal_districts, score_cutoff=75
#                 )
#                 if verify_match:
#                     district = verify_match[0]
            
#             # If not found with pattern, check individual words
#             if district == "UNKNOWN":
#                 words = text.split()
                
#                 # First pass: Look for exact or very close matches
#                 for word in words:
#                     # Clean the word
#                     clean_word = re.sub(r'[^A-Za-z]', '', word)
#                     if len(clean_word) >= 4:
#                         # Try exact match first
#                         if clean_word in self.nepal_districts:
#                             district = clean_word
#                             break
#                         # Try fuzzy match with high threshold
#                         word_match = process.extractOne(
#                             clean_word, self.nepal_districts, score_cutoff=85
#                         )
#                         if word_match:
#                             district = word_match[0]
#                             break
                
#                 # If still not found, try with lower threshold
#                 if district == "UNKNOWN":
#                     for word in words:
#                         clean_word = re.sub(r'[^A-Za-z]', '', word)
#                         if len(clean_word) >= 4 and clean_word.lower() not in ['municipality', 'muncipality', 'ward', 'province', 'district']:
#                             word_match = process.extractOne(
#                                 clean_word, self.nepal_districts, score_cutoff=75
#                             )
#                             if word_match:
#                                 district = word_match[0]
#                                 break

#             # MUNICIPALITY - Handle both Municipality and R.M (Rural Municipality)
#             municipality = "UNKNOWN"
            
#             muni_patterns = [
#                 # Match R.M: Name or RM: Name
#                 r'R\.?M\.?\s*[:\-]?\s*([A-Za-z]{3,})',
#                 # Match Municipality: Name (but not District: Name)
#                 r'(?:Municipality|Muncipality)\s*[:\-]?\s*([A-Za-z]{3,})',
#                 # Match Name R.M or Name RM
#                 r'([A-Za-z]{4,})\s+R\.?M\.?(?:\s|$)',
#                 # Match Name Municipality
#                 r'([A-Za-z]{4,})\s+(?:Municipality|Muncipality)',
#             ]
            
#             for pattern in muni_patterns:
#                 muni_match = re.search(pattern, text, re.I)
#                 if muni_match:
#                     potential_muni = muni_match.group(1)
#                     # Make sure we didn't capture "District" or the district name
#                     if potential_muni.lower() not in ['district', 'province', 'ward'] and potential_muni != district:
#                         municipality = potential_muni
#                         break
            
#             # Fallback: first substantial word that's not a district or keyword
#             if municipality == "UNKNOWN":
#                 keywords = ['ward', 'district', 'municipality', 'province', 'muncipality']
#                 words = text.split()
                
#                 for word in words:
#                     clean_word = re.sub(r'[^A-Za-z]', '', word)
#                     # Make sure it's not a district name or keyword
#                     if (len(clean_word) >= 4 and 
#                         clean_word.lower() not in keywords and 
#                         clean_word != district and
#                         clean_word not in self.nepal_districts):
#                         municipality = clean_word
#                         break

#             return f"Municipality: {municipality} District: {district} Ward No: {ward}"
        
#         # 5. Name Cleaning
#         if field == "full-name":
#             for label in self.template_keywords:
#                 text = re.sub(rf'\b{label}\b', '', text, flags=re.IGNORECASE)
#             text = re.sub(r'[^a-zA-Z\s]', '', text)
#             return " ".join(text.split())

#         return text
#     #utils.py


# this is the second version of the utils.py file 

import re
from rapidfuzz import process

class KycCleaner:
    def __init__(self):
        self.template_keywords = ["Municipality", "Muncipality", "Ward No", "District", "Province", "Year", "Month", "Day"]
        self.months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
                       "JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"]
        
        self.nepal_districts = [
            "Achham", "Arghakhanchi", "Baglung", "Baitadi", "Bajhang", "Bajura", "Banke", "Bara", "Bardiya",
            "Bhaktapur", "Bhojpur", "Chitwan", "Dadeldhura", "Dailekh", "Dang", "Darchula", "Dhading",
            "Dhankuta", "Dhanusa", "Dolakha", "Dolpa", "Doti", "Gorkha", "Gulmi", "Humla", "Ilam", "Jajarkot",
            "Jhapa", "Jumla", "Kailali", "Kalikot", "Kanchanpur", "Kapilvastu", "Kaski", "Kathmandu",
            "Kavrepalanchok", "Khotang", "Lalitpur", "Lamjung", "Mahottari", "Makwanpur", "Manang", "Morang",
            "Mugu", "Mustang", "Myagdi", "Nawalpur", "Nuwakot", "Okhaldhunga", "Palpa", "Panchthar", "Parbat",
            "Parsa", "Parasi", "Pyuthan", "Ramechhap", "Rasuwa", "Rautahat", "Rolpa", "Rukum East", "Rukum West",
            "Rupandehi", "Salyan", "Sankhuwasabha", "Saptari", "Sarlahi", "Sindhuli", "Sindhupalchok", "Siraha",
            "Solukhumbu", "Sunsari", "Surkhet", "Syangja", "Tanahu", "Taplejung", "Terhathum", "Udayapur"
        ]

        # ── All Nepali address unit types ──────────────────────────────────────
        self.address_unit_types = {
            # Type label         : possible OCR variations
            "Metropolitan":      ["metropolitan", "mahanagarpalika", "mahanagar"],
            "Sub-Metropolitan":  ["sub-metropolitan", "sub metropolitan", "upa-mahanagarpalika",
                                  "upamahanagarpalika", "sub-metro", "submeta"],
            "Municipality":      ["municipality", "muncipality", "municipaliy", "nagarpalika"],
            "Rural Municipality":["rural municipality", "gaupalika", "gaunpalika", "r.m", "rm",
                                  "rural", "gaun palika"],
        }

        # Canonical label for output
        self.unit_canonical = {
            "Metropolitan":      "Metropolitan City",
            "Sub-Metropolitan":  "Sub-Metropolitan City",
            "Municipality":      "Municipality",
            "Rural Municipality":"Rural Municipality",
        }

    # ── HELPER: detect address unit type ──────────────────────────────────────
    def _detect_unit_type(self, text):
        """Returns (canonical_label, matched_substring) or (None, None)"""
        text_lower = text.lower()
        ordered_types = [
            "Sub-Metropolitan",   # must come before Metropolitan
            "Metropolitan",
            "Rural Municipality", # must come before Municipality
            "Municipality",
    ]
        # Check in priority order (most specific first)
        for unit_type in ordered_types:
            for variant in self.address_unit_types[unit_type]:
                if variant in text_lower:
                    return self.unit_canonical[unit_type], variant
        return None, None

    # ── HELPER: extract unit name ──────────────────────────────────────────────
    def _extract_unit_name(self, text, unit_variant, district):
        """Extracts the name before or after the unit type keyword."""
        keywords_to_skip = ['ward', 'district', 'province', 'municipality',
                            'muncipality', 'metropolitan', 'rural', 'sub']

        # Pattern 1: "Name UnitType" e.g. "Phalgunanda R.M" or "Biratnagar Metropolitan"
        pattern_before = rf'([A-Za-z]{{3,}})\s+{re.escape(unit_variant)}'
        match = re.search(pattern_before, text, re.I)
        if match:
            name = match.group(1)
            if name.lower() not in keywords_to_skip and name != district:
                return name

        # Pattern 2: "UnitType: Name" or "UnitType Name"
        pattern_after = rf'{re.escape(unit_variant)}\s*[:\-]?\s*([A-Za-z]{{3,}})'
        match = re.search(pattern_after, text, re.I)
        if match:
            name = match.group(1)
            if name.lower() not in keywords_to_skip and name != district:
                return name

        return "UNKNOWN"

    def clean(self, field, text, conf=1.0):
        if not text:
            if field == "permanent-address":
                return "[PARTIAL]"
            return "[NOT_DETECTED]" if field != "officer-signature" else "[MISSING]"
        
        # ── BUG FIX: was checking "officer_signature" (underscore) but field
        #    comes in as "officer-signature" (hyphen) — was never returning [MISSING]
        text = text.strip()

        # 1. Signature
        if field == "officer-signature":
            return "[SIGNED]"

        # 2. Citizenship Number
        if field == "citisenship-number":
            text = text.upper()
            text = (
                text.replace('O', '0')
                    .replace('I', '1')
                    .replace('S', '5')
                    .replace('(', '1')
                    .replace('L', '1')
                    .replace('Z', '2')
            )
            # ── BUG FIX: preserve hyphens for Nepali format XX-XX-XX-XXXXX ──
            cleaned = re.sub(r'[^0-9\-]', '', text)
            # if re.match(r'^\d{2}-\d{2}-\d{2}-\d{4,6}$', cleaned):
            #     return cleaned  # return with hyphens preserved

            # Fallback: just digits
            digits = re.findall(r'\d+', cleaned)
            number = "".join(digits)
            if 10 <= len(number) <= 14:
                return number
            return "[INVALID]"

        # 3. DOB
        if field == "DOB":
            text = text.replace('_', '').replace('|', '').replace('~', '')
            text = text.replace('L', '1').replace('l', '1').replace('O', '0')
            text = text.replace('/9', '19').replace('I9', '19')

            text = re.sub(r'\bYcar\b', 'Year', text, flags=re.I)   # Ycar → Year
            text = re.sub(r'Month([A-Z]{3})', r'Month: \1', text)
            # ── BUG FIX: handle "Year2004" with no separator ──
            text = re.sub(r'Year\s*:?\s*(\d{4})', r'Year: \1', text, flags=re.I)
            text = re.sub(r'Month\s*:?\s*([A-Za-z]+)', r'Month: \1', text, flags=re.I)
            text = re.sub(r'Day\s*[.:]\s*([A-Za-z0-9]{1,2})', r'Day: \1', text, flags=re.I)

            structured_match = re.search(
                r'Year:\s*(\d{4})?\s*Month:\s*(\w+)?\s*Day:\s*([A-Za-z0-9]{1,2})?',
                text, re.IGNORECASE
            )

            if structured_match:
                res_year  = structured_match.group(1) or "????"
                month_text = structured_match.group(2)
                res_day   = structured_match.group(3) or "??"

                # ── BUG FIX: replace L/l in day AFTER extraction ──
                res_day = res_day.replace('L', '1').replace('l', '1').replace('O', '0')

                if month_text:
                    month_text = re.sub(r'[^A-Za-z]', '', month_text)
                    month_match = process.extractOne(month_text, self.months, score_cutoff=60)
                    res_month = month_match[0] if month_match else "???"
                else:
                    res_month = "???"
            else:
                year        = re.search(r'(?:19|20)\d{2}', text)
                day         = re.search(r'\b([1-9]|[0-2]\d|3[01])\b', text)
                month_match = process.extractOne(text, self.months, score_cutoff=60)
                res_year  = year.group() if year else "????"
                res_month = month_match[0] if month_match else "???"
                res_day   = day.group() if day else "??"

            return f"Year: {res_year} Month: {res_month} Day: {res_day}"

        # 4. Permanent Address
        if field == "permanent-address":
            text = text.replace("Muncipality", "Municipality")
            clean_text = re.sub(r':+', ':', text)

            # ── WARD ──────────────────────────────────────────────────────────
            ward = "?"
            ward_patterns = [
                r'Ward\s*No\.?\s*:?\s*(\d{1,2})',
                r'Ward\s+(\d{1,2})',
                r'No\.?\s*:?\s*(\d{1,2})',
            ]
            for pattern in ward_patterns:
                ward_match = re.search(pattern, clean_text, re.I)
                if ward_match:
                    ward = ward_match.group(1)
                    break

            # ── DISTRICT ──────────────────────────────────────────────────────
            district = "UNKNOWN"
            district_pattern = r'District\s*[:\-]?\s*([A-Za-z]{3,})'
            district_match = re.search(district_pattern, text, re.I)
            if district_match:
                potential = re.sub(r'[^A-Za-z]', '', district_match.group(1))
                verify = process.extractOne(potential, self.nepal_districts, score_cutoff=75)
                if verify:
                    district = verify[0]

            if district == "UNKNOWN":
                for word in text.split():
                    clean_word = re.sub(r'[^A-Za-z]', '', word)
                    if len(clean_word) >= 4:
                        if clean_word in self.nepal_districts:
                            district = clean_word
                            break
                        word_match = process.extractOne(clean_word, self.nepal_districts, score_cutoff=85)
                        if word_match:
                            district = word_match[0]
                            break

                if district == "UNKNOWN":
                    for word in text.split():
                        clean_word = re.sub(r'[^A-Za-z]', '', word)
                        skip = ['municipality', 'muncipality', 'ward', 'province',
                                'district', 'metropolitan', 'rural', 'sub']
                        if len(clean_word) >= 4 and clean_word.lower() not in skip:
                            word_match = process.extractOne(clean_word, self.nepal_districts, score_cutoff=75)
                            if word_match:
                                district = word_match[0]
                                break

            # ── ADDRESS UNIT TYPE + NAME ──────────────────────────────────────
            unit_type, unit_variant = self._detect_unit_type(text)

            if unit_type and unit_variant:
                unit_name = self._extract_unit_name(text, unit_variant, district)
                return f"{unit_type}: {unit_name} District: {district} Ward No: {ward}"

            # ── FALLBACK: old Municipality logic ─────────────────────────────
            municipality = "UNKNOWN"
            muni_patterns = [
                r'R\.?M\.?\s*[:\-]?\s*([A-Za-z]{3,})',
                r'(?:Municipality|Muncipality)\s*[:\-]?\s*([A-Za-z]{3,})',
                r'([A-Za-z]{4,})\s+R\.?M\.?(?:\s|$)',
                r'([A-Za-z]{4,})\s+(?:Municipality|Muncipality)',
            ]
            for pattern in muni_patterns:
                muni_match = re.search(pattern, text, re.I)
                if muni_match:
                    potential_muni = muni_match.group(1)
                    skip = ['district', 'province', 'ward']
                    if potential_muni.lower() not in skip and potential_muni != district:
                        municipality = potential_muni
                        break

            if municipality == "UNKNOWN":
                skip_words = ['ward', 'district', 'municipality', 'province',
                              'muncipality', 'metropolitan', 'rural', 'sub']
                for word in text.split():
                    clean_word = re.sub(r'[^A-Za-z]', '', word)
                    if (len(clean_word) >= 4 and
                            clean_word.lower() not in skip_words and
                            clean_word != district and
                            clean_word not in self.nepal_districts):
                        municipality = clean_word
                        break

            return f"Municipality: {municipality} District: {district} Ward No: {ward}"

        # 5. Full Name
        if field == "full-name":
            for label in self.template_keywords:
                text = re.sub(rf'\b{label}\b', '', text, flags=re.IGNORECASE)
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            return " ".join(text.split())

        return text