from typing import Dict, List, Union

import easyocr
import numpy as np
from gladia_api_utils.io import _open

codes_as_string = """abq	Abaza	abq
ady	Adyghe	ady
afr	Afrikaans	af
anp	Angika	ang
arb	Arabic	ar
asm	Assamese	as
ava	Avar	ava
aze	Azerbaijani	az
bel	Belarusian	be
bul	Bulgarian	bg
bih	Bihari	bh
bho	Bhojpuri	bho
ben	Bengali	bn
bos	Bosnian	bs
zho	Traditional Chinese	ch_tra
zho	Simplified Chinese	ch_sim
che	Chechen	che
cze	Czech	cs
wel	Welsh	cy
dan	Danish	da
dar	Dargwa	dar
ger	German	de
eng	English	en
spa	Spanish	es
est	Estonian	et
per	Persian (Farsi)	fa
fre	French	fr
gle	Irish	ga
gom	Goan Konkani	gom
hin	Hindi	hi
hrv	Croatian	hr
hun	Hungarian	hu
ind	Indonesian	id
inh	Ingush	inh
ice	Icelandic	is
ita	Italian	it
jpn	Japanese	ja
kdb	Kabardian	kbd
kan	Kannada	kn
kor	Korean	ko
kur	Kurdish	ku
lat	Latin	la
lbe	Lak	lbe
lez	Lezghian	lez
lit	Lithuanian	lt
lvs	Latvian	lv
mah	Magahi	mah
mai	Maithili	mai
mri	Maori	mi
mon	Mongolian	mn
mar	Marathi	mr
may	Malay	ms
mlt	Maltese	mt
nep	Nepali	ne
new	Newari	new
dut	Dutch	nl
nno	Norwegian	no
pro	Occitan	oc
pli	Pali	pi
pol	Polish	pl
por	Portuguese	pt
ron	Romanian	ro
rus	Russian	ru
srp_cyr	Serbian (cyrillic)	rs_cyrillic
srp_lat	Serbian (latin)	rs_latin
sck	Nagpuri	sck
slo	Slovak	sk
slv	Slovenian	sl
alb	Albanian	sq
swe	Swedish	sv
swa	Swahili	sw
tam	Tamil	ta
tab	Tabassaran	tab
tal	Telugu	te
tha	Thai	th
tgk	Tajik	tjk
tlg	Tagalog	tl
crh	Turkish	tr
uig	Uyghur	ug
ukr	Ukranian	uk
urd	Urdu	ur
uzb	Uzbek	uz
vie	Vietnamese	vi"""
codes_as_string = codes_as_string.split("\n")

easy_ocr_codes_mapping = {}
for code in codes_as_string:
    iso_3, lang, lang_code = code.split("\t")
    easy_ocr_codes_mapping[iso_3] = lang_code
    easy_ocr_codes_mapping[lang] = lang_code
    easy_ocr_codes_mapping[lang_code] = lang_code


def predict(
    image: bytes, source_language: str = "eng"
) -> Dict[str, Union[str, List[str]]]:
    """
    Call the EasyOcr package and return the text detected in the image by the ocr

    Args:
        image (bytes): The image to be processed
        source_language (str): The language of the image

    Returns:
        Dict[str, Union[str, List[str]]]: The text detected in the image by the ocr
    """

    if source_language not in easy_ocr_codes_mapping:
        plain_text = "Unknown language"
        text = plain_text
    else:
        image = _open(image)
        image = np.array(image)
        reader = easyocr.Reader([easy_ocr_codes_mapping[source_language]], gpu=True)
        text = reader.readtext(image, detail=False)
        plain_text = "\n".join(text)

    return {"prediction": plain_text, "prediction_raw": text}
