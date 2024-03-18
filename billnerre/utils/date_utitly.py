import re

def extract_month(*dts):
    for txt in dts:
        m = re.search("\d+[\.\-/年](\d+)[\.\-/月]\d+日?",txt)
        if m:
            return int(m.group(1))
    else:
        return None